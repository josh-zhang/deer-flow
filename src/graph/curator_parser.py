from __future__ import annotations

import json
import re
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from planner_model import CPPlannerOutput, CPPointAnalystOutput
from .curator_models import CuratorOutput, Relevance, normalize_doc_title

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
        evidence.id = idx + 1

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

    # discarded_count：尝试从 discarded 文本中计数表格行
    output.discarded_count = _count_discarded(output.discarded)
    output.total_input_count = output.retained_count + output.discarded_count

    return output


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
    """从 discarded 自由文本中估算丢弃数量（尽力而为）。"""
    if "无丢弃" in discarded_text or not discarded_text.strip():
        return 0
    # 计算表格数据行数（含 | 但不含 --- 的行，排除表头）
    lines = discarded_text.strip().split("\n")
    data_rows = 0
    for line in lines:
        line = line.strip()
        if "|" in line and "---" not in line and "序号" not in line:
            data_rows += 1
    return max(data_rows, 0)



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 从 tool_returns_cache 提取 citations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_citations_from_cache(
    cache: list["ToolCallRecord"],
) -> list[dict[str, Any]]:
    """
    从 Searcher 的 tool_returns_cache 中提取去重后的 citation 列表。

    每个 citation 结构:
    {
        "title":       str,   # 文档标题
        "url":         str,   # 文档 URL（可能为空）
        "file_id":     str,   # 知识库文档编号（可能为空）
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
            key = _citation_dedup_key(doc.document_title, doc.document_url)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            citations.append({
                "title": doc.document_title,
                "url": doc.document_url or "",
                "file_id": doc.file_id or "",
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


# ══════════════════════════════════════════════════════════════════════
# 类型化解析快捷函数
# ══════════════════════════════════════════════════════════════════════

def parse_planner_output(raw_text: str) -> CPPlannerOutput:
    """解析 Planner 输出"""
    return parse_model_from_text(raw_text, CPPlannerOutput)


def parse_point_analyst_output(raw_text: str) -> CPPointAnalystOutput:
    """解析 Point Analyst 输出"""
    return parse_model_from_text(raw_text, CPPointAnalystOutput)



