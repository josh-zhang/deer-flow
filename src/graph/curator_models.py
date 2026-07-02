from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  领域模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Relevance(str, Enum):
    DIRECT = "直接相关"
    INDIRECT = "间接相关"
    UNCERTAIN = "存疑"


class EvidenceItem(BaseModel):
    relevance: Relevance = Relevance.UNCERTAIN
    body: str = ""
    id: str = "0"

    # ── 从 body 解析的结构化字段 ──
    source_document: str = ""
    referenced_chunks: list[str] = Field(default_factory=list)
    is_expired: bool = False
    is_tool_extracted: bool = False

    @field_validator("relevance", mode="before")
    @classmethod
    def _fuzzy_relevance(cls, v: object) -> str:
        if not isinstance(v, str):
            return Relevance.UNCERTAIN.value
        v = v.strip()
        for member in Relevance:
            if v == member.value:
                return v
        FUZZY_MAP = {"直接": Relevance.DIRECT.value, "间接": Relevance.INDIRECT.value}
        for keyword, target in FUZZY_MAP.items():
            if keyword in v:
                return target
        return Relevance.UNCERTAIN.value


class CuratorOutput(BaseModel):
    evaluation_basis: str = ""
    evidences: list[EvidenceItem] = Field(default_factory=list)
    contradictions: str = "未发现跨片段矛盾。"
    discarded: str = "无丢弃片段。"
    coverage: str = ""

    # ── 代码计算 ──
    total_input_count: int = 0
    retained_count: int = 0
    direct_count: int = 0
    indirect_count: int = 0
    uncertain_count: int = 0
    discarded_count: int = 0
    expired_count: int = 0
    tool_extracted_count: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  解析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _enrich_evidence(evidence: EvidenceItem) -> None:
    """从 body 文本解析结构化字段，就地更新。"""
    for line in evidence.body.split("\n"):
        stripped = line.strip()

        if stripped.startswith("来源文档：") or stripped.startswith("来源文档:"):
            evidence.source_document = _extract_after_colon(stripped)

        elif stripped.startswith("段落编号：") or stripped.startswith("段落编号:") or stripped.startswith("引用段落：") or stripped.startswith("引用段落:"):
            raw = _extract_after_colon(stripped)
            # "1, 3, 7, 15" → ["1", "3", "7", "15"]
            evidence.referenced_chunks = [
                t.strip() for t in re.split(r"[,，\s]+", raw) if t.strip()
            ]

        elif stripped.startswith("信息质量：") or stripped.startswith("信息质量:"):
            quality = _extract_after_colon(stripped)
            if "已过期" in quality:
                evidence.is_expired = True
            if "工具截取" in quality:
                evidence.is_tool_extracted = True

        elif stripped.startswith("信息完整性：") or stripped.startswith("信息完整性:"):
            completeness = _extract_after_colon(stripped)
            if "工具截取" in completeness:
                evidence.is_tool_extracted = True


def _extract_after_colon(line: str) -> str:
    for sep in ("：", ":"):
        if sep in line:
            return line.split(sep, 1)[1].strip()
    return line.strip()


def _load_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    try:
        import json_repair
        return json_repair.loads(text)
    except Exception:
        pass
    raise CuratorParseError("JSON 解析失败")


def _count_discarded(discarded_text: str) -> int:
    if "无丢弃" in discarded_text or not discarded_text.strip():
        return 0
    lines = discarded_text.strip().split("\n")
    return sum(
        1 for line in lines
        if "|" in line and "---" not in line and "序号" not in line
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  段落解析：从 document_chunk_maps 提取原文
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ResolvedEvidence:
    """一条 evidence 解析后的完整数据，供 Rule Splitter / Analyst 使用。"""
    evidence_id: str
    relevance: str
    source_document: str
    body_metadata: str            # body 中"段落编号"之前的部分
    body_content: str             # body 中"段落编号"及之后的部分
    referenced_chunks: list[str]  # Curator 引用的段落编号
    resolved_chunks: list[dict]   # 从 chunk_map 解析的原文 [{chunk_index, chunk_content}]
    is_expired: bool
    is_tool_extracted: bool


_STRIP_CHARS_PATTERN = re.compile(r"[《》〈〉<>【】\[\]「」『』\s]")


def normalize_doc_title(title: str) -> str:
    """去除书名号、括号、空白等装饰字符，返回纯净文档名。"""
    return _STRIP_CHARS_PATTERN.sub("", title).strip()


def _extract_doc_name_candidates(raw_title: str) -> list[str]:
    """
    从可能包含多个文档名的字符串中提取候选文档名。

    Curator 偶尔违反合并规则，在来源文档字段拼接多个文档名，例如：
    - "《信用卡分期业务管理办法》《白金卡产品说明书》"
    - "《文档A》/《文档B》"
    - "文档A、文档B"（无书名号）
    - "「文档A」「文档B」"

    此函数用于防御性解析：拆出候选名取第一个匹配的 chunk_map。

    策略：
    1. 优先从各种括号对中提取（《》「」『』〈〉【】）
    2. 如无括号对，按常见分隔符拆分（、/；;和）
    3. 过滤掉过短的碎片（≤2字符，可能是拆分噪音）

    返回候选名列表（不含装饰符）。如果只能提取到一个候选，返回空列表（单文档无需拆分）。
    """
    # 策略 1: 从括号对中提取
    bracket_patterns = [
        r"《([^》]+)》",
        r"「([^」]+)」",
        r"『([^』]+)』",
        r"〈([^〉]+)〉",
        r"【([^】]+)】",
    ]
    for pattern in bracket_patterns:
        found = re.findall(pattern, raw_title)
        if len(found) >= 2:
            return [name.strip() for name in found if name.strip()]

    # 策略 2: 无括号对时，按分隔符拆分
    # 常见分隔符：中文顿号、斜杠、分号、"和"/"与"
    parts = re.split(r"[、/；;]|(?<=[\u4e00-\u9fff])和(?=[\u4e00-\u9fff])", raw_title)
    candidates = [normalize_doc_title(p.strip()) for p in parts if p.strip()]
    # 过滤过短碎片（≤2字符可能是拆分噪音如"和""与"）
    candidates = [c for c in candidates if len(c) > 2]
    if len(candidates) >= 2:
        return candidates

    return []


def fuzzy_match_chunk_map(
    curator_doc_title: str,
    document_chunk_maps: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    """
    四级匹配策略查找 chunk_map：
    1. 精确匹配
    2. 归一化后精确匹配
    3. 归一化后包含匹配（一方包含另一方）
    4. 多文档名拆分匹配（防御性：取第一个命中的）

    返回 chunk_map 或 None。

    注意：每条 evidence 设计上只对应一个文档。Level 4 仅用于防御
    LLM 偶尔违反合并规则（在来源文档字段拼接多个文档名）的情况。
    """
    # Level 1: 精确匹配
    if curator_doc_title in document_chunk_maps:
        return document_chunk_maps[curator_doc_title]

    # Level 2: 归一化匹配
    norm_query = normalize_doc_title(curator_doc_title)
    if not norm_query:
        return None

    for key, chunk_map in document_chunk_maps.items():
        if normalize_doc_title(key) == norm_query:
            return chunk_map

    # Level 3: 包含匹配
    for key, chunk_map in document_chunk_maps.items():
        norm_key = normalize_doc_title(key)
        if not norm_key:
            continue
        if norm_query in norm_key or norm_key in norm_query:
            logger.info(
                "文档名包含匹配：Curator '%s' ↔ chunk_map '%s'",
                curator_doc_title, key,
            )
            return chunk_map

    # Level 4: 多文档名拆分匹配（防御性：取第一个命中的）
    candidates = _extract_doc_name_candidates(curator_doc_title)
    if candidates:
        for candidate in candidates:
            result = fuzzy_match_chunk_map(candidate, document_chunk_maps)
            if result is not None:
                logger.warning(
                    "来源文档字段包含多个文档名（违反合并规则），取第一个匹配：'%s' → '%s'",
                    curator_doc_title, candidate,
                )
                return result

    return None


# ── 在 resolve_all_evidence_chunks 中替换原有的 dict.get ──

def resolve_all_evidence_chunks(
    curator_output: "CuratorOutput",
    document_chunk_maps: dict[str, dict[str, str]],
) -> list["ResolvedEvidence"]:
    results = []

    for evidence in curator_output.evidences:
        metadata, content = _find_content_boundary(evidence.body)
        doc_title = evidence.source_document

        # ── 模糊匹配查找 chunk_map ──
        chunk_map = fuzzy_match_chunk_map(doc_title, document_chunk_maps)

        if chunk_map is None:
            logger.warning(
                "Evidence %s 的文档 '%s' 在 chunk_maps 中未找到（含模糊匹配）。"
                "回退至 body 原始内容。chunk_maps 现有 keys: %s",
                evidence.id, doc_title, list(document_chunk_maps.keys()),
            )
            chunk_map = {}

        resolved = []
        missing = []
        for idx in evidence.referenced_chunks:
            if idx in chunk_map:
                resolved.append({"chunk_index": idx, "chunk_content": chunk_map[idx]})
            else:
                missing.append(idx)

        if missing and chunk_map:
            logger.warning(
                "Evidence %s 引用段落 %s 在文档 '%s' 的 chunk_map 中未找到。",
                evidence.id, missing, doc_title,
            )

        results.append(ResolvedEvidence(
            evidence_id=evidence.id,
            relevance=evidence.relevance.value,
            source_document=doc_title,
            body_metadata=metadata,
            body_content=content,
            referenced_chunks=evidence.referenced_chunks,
            resolved_chunks=resolved,
            is_expired=evidence.is_expired,
            is_tool_extracted=evidence.is_tool_extracted,
        ))

    return results


def _find_content_boundary(body: str) -> tuple[str, str]:
    """将 body 分为 metadata / content 两段（以"段落编号"行为界）。

    LLM 输出 body 模板中，段落编号是最后一个一级字段，其后不再有
    "具体内容"包装层。此函数匹配"段落编号："作为分界线。
    """
    lines = body.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("段落编号") and ("：" in stripped or ":" in stripped):
            metadata = "\n".join(lines[:i])
            colon_pos = stripped.find("：")
            if colon_pos == -1:
                colon_pos = stripped.find(":")
            after = stripped[colon_pos + 1:].strip()
            content = (after + "\n" + "\n".join(lines[i + 1:])) if after else "\n".join(lines[i + 1:])
            return metadata.strip(), content.strip()
    return body.strip(), ""


def build_rule_splitter_view_with_resolved(
    output: CuratorOutput,
    resolved_list: list[ResolvedEvidence],
) -> str:
    """
    用 resolved_chunks（chunk_map 原文）替换 body 中的具体内容，
    确保 Rule Splitter 拿到的是未经 Curator 精简的完整段落原文。
    """
    parts = [f"\n## 评估基准\n{output.evaluation_basis}", "\n## 相关业务依据清单"]

    for r in resolved_list:
        parts.append("")

        if r.resolved_chunks:
            # 用 chunk_map 原文重建具体内容
            content_lines = []
            for chunk in r.resolved_chunks:
                content_lines.append(f"**[{chunk['chunk_index']}]**\n{chunk['chunk_content']}")
            full_content = "\n\n".join(content_lines)
            parts.append(f"### 业务依据 {r.evidence_id}\n\n{r.body_metadata}\n具体内容：\n{full_content}")
        else:
            # 回退：使用 Curator body 原始内容
            evidence = output.evidences[int(r.evidence_id) - 1]
            parts.append(f"### 业务依据 {r.evidence_id}\n\n{evidence.body}")

    parts.append(f"\n## 矛盾提示\n{output.contradictions}")
    return "\n".join(parts)