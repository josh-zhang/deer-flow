from __future__ import annotations

import json
import re
import logging
import unicodedata
from typing import TypeVar, Any

from pydantic import BaseModel, ValidationError

from src.graph.curator_models import CuratorOutput, Relevance, EvidenceItem
from src.rag.retriever import ToolCallRecord

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class CuratorParseError(Exception):
    """Curator 解析失败"""


def parse_curator_output(text: str) -> CuratorOutput:
    """
    将 Evidence Curator 的 JSON 文本解析为 CuratorOutput。

    鲁棒性保障链：
    1. json.loads() — 确定性 JSON 解析
    2. _try_repair() — 截断修复兜底
    3. Pydantic model_validate() — 类型校验 + 默认值 + 模糊枚举匹配
    4. 计数字段由代码统计
    """
    text = text.strip()

    # ── Step 1: JSON 反序列化 ──
    data = _load_json(text)

    # ── Step 2: Pydantic 校验 ──
    try:
        output = CuratorOutput.model_validate(data)
    except Exception as e:
        raise CuratorParseError(f"Pydantic 校验失败: {e}") from e

    # ── Step 3: 填充 ID 和计数 ──
    for idx, evidence in enumerate(output.evidences):
        evidence.id = str(idx + 1)
        _enrich_evidence(evidence)

    output.retained_count = len(output.evidences)
    output.direct_count = sum(
        1 for e in output.evidences if e.relevance == Relevance.DIRECT
    )
    output.indirect_count = sum(
        1 for e in output.evidences if e.relevance == Relevance.INDIRECT
    )
    output.uncertain_count = sum(
        1 for e in output.evidences if e.relevance == Relevance.UNCERTAIN
    )
    output.expired_count = sum(1 for e in output.evidences if e.is_expired)
    output.tool_extracted_count = sum(1 for e in output.evidences if e.is_tool_extracted)
    # discarded_count：尝试从 discarded 文本中计数表格行
    output.discarded_count = _count_discarded(output.discarded)
    output.total_input_count = output.retained_count + output.discarded_count

    return output

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
    """尝试解析 JSON，失败时进行截断修复。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 兜底 1：去除 markdown 代码块包裹
    cleaned = text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # 去首行 ```json 和末行 ```
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # 兜底 2：json-repair 截断修复
    try:
        import json_repair
        return json_repair.loads(text)
    except Exception:
        pass

    raise CuratorParseError("JSON 解析失败，所有修复策略均未成功")


def _count_discarded(discarded_text: str) -> int:
    if "无丢弃" in discarded_text or not discarded_text.strip():
        return 0
    lines = discarded_text.strip().split("\n")
    return sum(
        1 for line in lines
        if "|" in line and "---" not in line and "序号" not in line
    )


def normalize_doc_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title)

    NONE_WORD_PATTERN = re.compile(r"[^\w]")

    return NONE_WORD_PATTERN.sub("", normalized).strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 从 tool_returns_cache 提取 citations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_citations_from_cache(
    cache: list[ToolCallRecord],
) -> list[dict[str, Any]]:
    """
    从 Searcher 的 tool_returns_cache 中提取去重后的 citation 列表。

    每个 citation 结构:
    {
        "title":       str,   # 文档标题
        "url":         str,   # 文档 URL（可能为空）
        "description": str,   # 文档分类描述
        "source_tool": str,   # 来源工具 "local_search_tool" | "crawl_tool" | "fetch_tool"
        "is_extracted": bool, # 是否经过截取
    }
    """
    citations: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for record in cache:
        if record.artifact is None:
            continue

        for doc in record.artifact.documents:
            # 去重 key: 优先用 url，其次用归一化 title
            normalize_title = normalize_doc_title(doc.document_title)

            key = _citation_dedup_key(normalize_title, doc.document_url)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            citations.append({
                "title": normalize_title,
                "original_title": doc.document_title,
                "url": doc.document_url or "",
                "description": doc.description or "",
                "source_tool": record.tool_name,
                "is_extracted": doc.is_extracted,
            })

    return citations


def _citation_dedup_key(title: str, url: str | None) -> str:
    """生成 citation 去重 key。优先用 url，无 url 则用归一化 title。"""
    if url:
        return f"url:{url.strip()}"
    return f"title:{normalize_doc_title(title)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. 跨步骤 merge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def merge_citations(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    将新 citations 合并到 existing，保持插入顺序，URL/title 去重。

    合并策略:
    - 已存在的 citation 保留原始位置
    - 新 citation 追加到末尾
    - 同一文档出现在多个工具中时，优先保留 crawl/fetch 的元数据
      （因为 crawl/fetch 有更完整的文档信息）
    """
    result = list(existing)
    seen_keys: set[str] = set()

    for c in result:
        seen_keys.add(_citation_dedup_key(c.get("title", ""), c.get("url")))

    for c in new:
        key = _citation_dedup_key(c.get("title", ""), c.get("url"))
        if key in seen_keys:
            # 已存在：如果新的来自 crawl/fetch 且旧的来自 local_search，用新的更新
            if c.get("source_tool") in ("crawl_tool", "fetch_tool"):
                for i, existing_c in enumerate(result):
                    existing_key = _citation_dedup_key(
                        existing_c.get("title", ""), existing_c.get("url")
                    )
                    if existing_key == key and existing_c.get("source_tool") == "local_search_tool":
                        # 保留原位置，更新元数据
                        result[i] = {**existing_c, **c}
                        break
            continue
        seen_keys.add(key)
        result.append(c)

    return result




# ══════════════════════════════════════════════════════════════════════
# 通用 JSON 提取 + Pydantic 解析
# ══════════════════════════════════════════════════════════════════════

def extract_json_str(text: str) -> str | None:
    """
    从 LLM 输出文本中提取 JSON 字符串。
    尝试策略：直接全文 → 代码块 → 首个 {/[ 到末尾匹配。
    """
    text = text.strip()

    # 策略 1：整段即 JSON
    if (text.startswith("{") and text.endswith("}")) or \
            (text.startswith("[") and text.endswith("]")):
        return text

    # 策略 2：代码块提取
    pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for m in matches:
        m = m.strip()
        if m.startswith(("{", "[")):
            return m

    # 策略 3：首尾字符定位
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        end = text.rfind(end_char)
        if end <= start:
            continue
        return text[start:end + 1]

    return None


def parse_model_from_text(text: str, model_class: type[T]) -> T:
    """
    从 LLM 输出文本中提取 JSON 并解析为 Pydantic 模型。
    利用 Pydantic v2 的 model_validate 自动执行字段校验和类型转换。

    Raises:
        ValueError: JSON 提取或模型校验失败
    """
    json_str = extract_json_str(text)
    if json_str is None:
        raise ValueError(
            f"无法从文本中提取 JSON。model={model_class.__name__}, "
            f"text_preview={text[:300]}"
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON 解析失败: {e}。model={model_class.__name__}, "
            f"json_preview={json_str[:300]}"
        )

    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        logger.warning(
            f"Pydantic 校验有问题，尝试宽松解析。errors={e.error_count()}"
        )
        # 宽松重试：移除 None 值字段
        if isinstance(data, dict):
            cleaned = {k: v for k, v in data.items() if v is not None}
            try:
                return model_class.model_validate(cleaned)
            except ValidationError:
                pass
        raise ValueError(
            f"模型校验失败: {e}。model={model_class.__name__}"
        )