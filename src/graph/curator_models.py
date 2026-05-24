from __future__ import annotations

import json
import re
import logging
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

class CuratorParseError(Exception):
    """Curator 解析失败"""


def parse_curator_output(text: str) -> CuratorOutput:
    text = text.strip()
    data = _load_json(text)

    try:
        output = CuratorOutput.model_validate(data)
    except Exception as e:
        raise CuratorParseError(f"Pydantic 校验失败: {e}") from e

    for idx, evidence in enumerate(output.evidences):
        evidence.id = str(idx + 1)
        _enrich_evidence(evidence)

    output.retained_count = len(output.evidences)
    output.direct_count = sum(1 for e in output.evidences if e.relevance == Relevance.DIRECT)
    output.indirect_count = sum(1 for e in output.evidences if e.relevance == Relevance.INDIRECT)
    output.uncertain_count = sum(1 for e in output.evidences if e.relevance == Relevance.UNCERTAIN)
    output.expired_count = sum(1 for e in output.evidences if e.is_expired)
    output.tool_extracted_count = sum(1 for e in output.evidences if e.is_tool_extracted)
    output.discarded_count = _count_discarded(output.discarded)
    output.total_input_count = output.retained_count + output.discarded_count

    return output


def _enrich_evidence(evidence: EvidenceItem) -> None:
    """从 body 文本解析结构化字段，就地更新。"""
    for line in evidence.body.split("\n"):
        stripped = line.strip()

        if stripped.startswith("来源：") or stripped.startswith("来源:"):
            evidence.source_document = _extract_after_colon(stripped)

        elif stripped.startswith("引用段落：") or stripped.startswith("引用段落:"):
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
    body_metadata: str            # body 中"具体内容"之前的部分
    body_content: str             # body 中"具体内容"之后的部分（Curator 精简后的文本）
    referenced_chunks: list[str]  # Curator 引用的段落编号
    resolved_chunks: list[dict]   # 从 chunk_map 解析的原文 [{chunk_index, chunk_content}]
    is_expired: bool
    is_tool_extracted: bool


from dataclasses import dataclass


def resolve_all_evidence_chunks(
    curator_output: CuratorOutput,
    document_chunk_maps: dict[str, dict[str, str]],
) -> list[ResolvedEvidence]:
    """
    遍历 Curator 输出的每条 evidence，
    利用 State.document_chunk_maps 将引用段落解析为原文。

    Args:
        curator_output:       parse_curator_output() 的返回值
        document_chunk_maps:  State 中的全局文档段落映射
                              {document_title: {chunk_index: chunk_content}}

    Returns:
        每条 evidence 对应一个 ResolvedEvidence，
        其中 resolved_chunks 为从 chunk_map 解析出的原文段落列表。
    """
    results: list[ResolvedEvidence] = []

    for evidence in curator_output.evidences:
        metadata, content = _find_content_boundary(evidence.body)

        # 查找该文档的 chunk_map
        doc_title = evidence.source_document
        chunk_map = document_chunk_maps.get(doc_title, {})

        resolved = []
        missing = []
        for idx in evidence.referenced_chunks:
            if idx in chunk_map:
                resolved.append({
                    "chunk_index": idx,
                    "chunk_content": chunk_map[idx],
                })
            else:
                missing.append(idx)

        if missing:
            logger.warning(
                "Evidence %s 引用段落 %s 在文档 '%s' 的 chunk_map 中未找到 "
                "(chunk_map 含 %d 个段落)。回退至 body 中的具体内容。",
                evidence.id, missing, doc_title, len(chunk_map),
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
    """将 body 分为 metadata / content 两段（以"具体内容"行为界）。"""
    lines = body.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("具体内容") and ("：" in stripped or ":" in stripped):
            metadata = "\n".join(lines[:i])
            colon_pos = stripped.find("：")
            if colon_pos == -1:
                colon_pos = stripped.find(":")
            after = stripped[colon_pos + 1:].strip()
            content = (after + "\n" + "\n".join(lines[i + 1:])) if after else "\n".join(lines[i + 1:])
            return metadata.strip(), content.strip()
    return body.strip(), ""


def _build_rule_splitter_view_with_resolved(
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