from __future__ import annotations

import logging
import re
from typing import Any

from src.cg.types import (
    ChannelCopy,
    MasterCopy,
    ProductFacts,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  数值交叉校验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_NUMERIC_PATTERN = re.compile(
    r"\d+(?:\.\d+)?%?|[\d,]+(?:\.\d+)?元|[\d,]+(?:\.\d+)?倍"
)


def verify_fact_claims(
    product_facts: ProductFacts | dict,
    chunk_maps: dict[str, dict[str, str]],
) -> list[str]:
    """
    数值交叉校验：检查产品事实中的数值是否能在 chunk_maps 原文中找到支撑。

    返回警告消息列表（空列表表示无问题）。

    策略：对 ProductFacts.benefits 中每个 benefit 的 numeric_value，
    尝试在其 source_document 的对应 source_chunks 中找到该数值字面量。
    """
    warnings: list[str] = []

    if isinstance(product_facts, dict):
        benefits = product_facts.get("benefits", [])
        product_name = product_facts.get("product_name", "")
    else:
        benefits = [b.model_dump() for b in product_facts.benefits]
        product_name = product_facts.product_name

    for benefit in benefits:
        numeric_value = benefit.get("numeric_value", "").strip()
        if not numeric_value or numeric_value == "未检索到":
            continue

        source_doc = benefit.get("source_document", "")
        source_chunks = benefit.get("source_chunks", [])
        benefit_name = benefit.get("name", "")

        if not source_doc or not source_chunks:
            warnings.append(
                f"[数值无法验证] 权益「{benefit_name}」的数值「{numeric_value}」"
                f"缺少来源文档或段落引用。"
            )
            continue

        # 在 chunk_maps 中查找对应段落
        chunk_map = _fuzzy_match_chunk_map(source_doc, chunk_maps)
        if chunk_map is None:
            warnings.append(
                f"[文档未找到] 权益「{benefit_name}」的来源文档「{source_doc}」"
                f"在 chunk_maps 中未找到。"
            )
            continue

        # 从声明的段落中提取文本并搜索数值
        found = False
        for chunk_ref in source_chunks:
            # chunk_ref 格式如 "[3]" 或 "3"
            chunk_idx = chunk_ref.strip("[]")
            chunk_content = chunk_map.get(chunk_idx, "")
            if numeric_value in chunk_content:
                found = True
                break

        if not found:
            warnings.append(
                f"[数值未验证] 产品「{product_name}」权益「{benefit_name}」"
                f"声称数值「{numeric_value}」在引用段落 {source_chunks} 中未找到。"
                f"可能是 Fact Miner 的误提取或段落引用有误。"
            )

    return warnings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  文案一致性校验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_consistency(
    master: MasterCopy | dict,
    channel_copy: ChannelCopy | dict,
) -> list[str]:
    """
    母版与渠道版文案一致性校验。

    检查渠道版文案中是否存在与母版不一致的数值，返回不一致提示列表。
    """
    issues: list[str] = []

    if isinstance(master, dict):
        master_text = " ".join([
            master.get("headline", ""),
            master.get("body", ""),
            master.get("compliance_footer", ""),
        ])
    else:
        master_text = " ".join([master.headline, master.body, master.compliance_footer])

    if isinstance(channel_copy, dict):
        channel_text = channel_copy.get("copy_text", "")
        channel_name = channel_copy.get("channel_name", "")
    else:
        channel_text = channel_copy.copy_text
        channel_name = channel_copy.channel_name

    # 提取母版中的数值
    master_numerics = set(_NUMERIC_PATTERN.findall(master_text))

    # 提取渠道版中的数值
    channel_numerics = set(_NUMERIC_PATTERN.findall(channel_text))

    # 检查渠道版出现的数值是否都来自母版
    extra_numerics = channel_numerics - master_numerics
    for num in sorted(extra_numerics):
        issues.append(
            f"[数值偏差] 渠道版「{channel_name}」出现了数值「{num}」，"
            f"该数值在母版中未找到。可能是 Channel Adapter 引入了新数值。"
        )

    return issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  事实校验层
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fact_check_master_copy(
    master: MasterCopy,
    product_facts: ProductFacts,
) -> MasterCopy:
    """
    事实校验层：检查母版文案中引用的权益是否能在 ProductFacts 中找到。

    返回带有校验注释的 MasterCopy（不修改文案内容，仅在 benefit_references 中追加警告）。
    """
    if not master.benefit_references:
        return master

    fact_benefit_names = {b.name for b in product_facts.benefits}

    updated_refs = []
    for ref in master.benefit_references:
        # ref 格式: "权益名称→文案中的表述"
        if "→" in ref:
            benefit_name, _ = ref.split("→", 1)
            benefit_name = benefit_name.strip()
            if benefit_name and benefit_name not in fact_benefit_names:
                updated_refs.append(f"{ref} [⚠️ 权益名称在ProductFacts中未找到]")
                logger.warning(
                    "母版文案引用了 ProductFacts 中不存在的权益名称：「%s」", benefit_name
                )
            else:
                updated_refs.append(ref)
        else:
            updated_refs.append(ref)

    return master.model_copy(update={"benefit_references": updated_refs})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  chunk_maps 合并
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def merge_chunk_maps(
    existing: dict[str, dict[str, str]],
    new: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """
    合并两个 document_chunk_maps，同一文档的 chunk 取并集。

    新数据中 chunk_index 与旧数据冲突时，新数据优先（假设新的更完整）。
    """
    result = {k: dict(v) for k, v in existing.items()}
    for doc_title, chunk_map in new.items():
        if doc_title in result:
            result[doc_title].update(chunk_map)
        else:
            result[doc_title] = dict(chunk_map)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  引用池格式化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def format_dual_citations(
    fact_citations: list[dict[str, Any]],
    rule_citations: list[dict[str, Any]],
) -> str:
    """
    将业务知识库引用和消保知识库引用合并格式化为 Markdown 字符串。

    供下游节点注入提示词。
    """
    parts: list[str] = []

    if fact_citations:
        parts.append("### 业务知识库引用来源")
        for i, c in enumerate(fact_citations, 1):
            title = c.get("title", "Untitled")
            url = c.get("url", "")
            file_id = c.get("file_id", "")
            parts.append(f"{i}. **{title}**")
            if url:
                parts.append(f"   - URL: `{url}`")
            if file_id:
                parts.append(f"   - 编号: `{file_id}`")

    if rule_citations:
        parts.append("\n### 消保审查知识库引用来源")
        for i, c in enumerate(rule_citations, 1):
            title = c.get("title", "Untitled")
            url = c.get("url", "")
            file_id = c.get("file_id", "")
            parts.append(f"{i}. **{title}**")
            if url:
                parts.append(f"   - URL: `{url}`")
            if file_id:
                parts.append(f"   - 编号: `{file_id}`")

    return "\n".join(parts) if parts else "（无知识库引用）"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  工具缓存 citations 提取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_citations_from_tool_cache(cache: list[Any]) -> list[dict[str, Any]]:
    """
    从工具缓存（list[ToolCallRecord]）中提取去重后的 citation 列表。

    每个 citation 结构:
    {
        "title":       str,
        "url":         str,
        "file_id":     str,
        "description": str,
        "source_tool": str,
        "is_extracted": bool,
    }

    与 src/graph/curator_parser.py 中 extract_citations_from_cache 保持一致的接口。
    """
    citations: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for record in cache:
        artifact = getattr(record, "artifact", None)
        if artifact is None:
            continue

        documents = getattr(artifact, "documents", [])
        for doc in documents:
            title = getattr(doc, "document_title", "") or ""
            url = getattr(doc, "document_url", "") or ""
            file_id = getattr(doc, "file_id", "") or ""
            description = getattr(doc, "description", "") or ""
            is_extracted = getattr(doc, "is_extracted", False)
            tool_name = getattr(record, "tool_name", "unknown")

            # 去重 key：优先用 url，其次用归一化 title
            key = f"url:{url.strip()}" if url else f"title:{_normalize_title(title)}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            citations.append({
                "title": title,
                "url": url,
                "file_id": file_id,
                "description": description,
                "source_tool": tool_name,
                "is_extracted": is_extracted,
            })

    return citations


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  内部工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

