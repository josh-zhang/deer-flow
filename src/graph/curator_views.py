from __future__ import annotations

import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from src.graph.curator_models import CuratorOutput, Relevance, ResolvedEvidence
from src.graph.curator_parser import _citation_dedup_key, normalize_doc_title
from src.rag.retriever import ToolCallRecord

logger = logging.getLogger(__name__)

_INDIRECT_CONTENT_PREVIEW_LIMIT = 300

TOOL_NAME_DISPLAY = {
    "local_search_tool": "语义检索",
    "crawl_tool": "全文获取(url)",
    "fetch_tool": "全文获取(文档名)",
}


def _find_content_boundary(body: str) -> tuple[str, str]:
    """将 evidence body 文本分为 metadata 部分和 content 部分。
    约定：body 中“具体内容：”行以下为 content。
    如果找不到，整个 body 视为 metadata（存疑-低等无 content 的情况）。
    返回 (metadata_text, content_text)，content_text 可能为空字符串。
    """
    # 查找“具体内容：”或“具体内容:”
    lines = body.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("段落编号") and (":" in stripped or "：" in stripped):
            metadata = "\n".join(lines[:i])
            # 具体内容可能紧跟在冒号后，也可能在下一行
            colon_pos = stripped.find(":")
            if colon_pos == -1:
                colon_pos = stripped.find("：")
            after_colon = stripped[colon_pos + 1 :].strip()
            if after_colon:
                content = after_colon + "\n" + "\n".join(lines[i + 1 :])
            else:
                content = "\n".join(lines[i + 1 :])
            return metadata.strip(), content.strip()
    return body.strip(), ""


def _format_evidence_for_analyst(resolved_list: list[ResolvedEvidence]) -> str:
    """分析视图（改造后）：
    - 直接相关 -> 完整 body
    - 间接相关 / 存疑 -> metadata + 具体内容截断预览
    """
    parts = []
    for r in resolved_list:
        parts.append(f"**业务依据 {r.evidence_id}**\n\n{r.body_metadata}")
        if r.resolved_chunks:
            # 用 chunk_map 原文重建具体内容
            content_lines = []
            for chunk in r.resolved_chunks:
                content_lines.append(
                    f"**段落[{chunk['chunk_index']}]**\n{chunk['chunk_content']}"
                )
            full_content = "\n\n".join(content_lines)
            if r.relevance == Relevance.DIRECT:
                parts.append(f"具体内容：\n{full_content}\n")
            else:
                preview = full_content[:_INDIRECT_CONTENT_PREVIEW_LIMIT]
                if len(full_content) > _INDIRECT_CONTENT_PREVIEW_LIMIT:
                    preview += "\n...[具体内容已截断，以上为前部预览]"
                parts.append(f"具体内容预览：\n{preview}\n")
        else:
            parts.append("*具体内容查询失败*\n")
    return "\n".join(parts)


def generate_analysis_view(
    output: CuratorOutput, resolved_list: list[ResolvedEvidence]
) -> str:
    """Analyst 视图：直接相关含全文，其余仅 metadata + 统计概览。"""
    parts: list[str] = []
    # 一、统计概览（新增）
    stats = (
        f"共保留 {output.retained_count} 条依据"
        f"（直接相关 {output.direct_count}，"
        f"间接相关 {output.indirect_count}，"
        f"存疑 {output.uncertain_count}）"
    )
    if output.expired_count > 0:
        stats += f"，其中 {output.expired_count} 条已过期"
    if output.tool_extracted_count > 0:
        stats += f"，{output.tool_extracted_count} 条为工具截取"
    parts.append(f"【依据统计】{stats}。")
    parts.append(f"\n【评估基准】\n{output.evaluation_basis}")
    parts.append("\n【相关业务依据清单】")
    # for e in output.evidences:
    # parts.append("")
    parts.append(_format_evidence_for_analyst(resolved_list))
    parts.append(f"\n【矛盾提示】\n{output.contradictions}")
    parts.append(f"\n【丢弃清单】\n{output.discarded}")
    parts.append(f"\n【检索覆盖度自评】\n{output.coverage}")
    return "\n".join(parts)


def generate_rule_splitter_view(
    output: CuratorOutput,
    resolved_list: list[ResolvedEvidence] | None = None,
) -> str:
    """Rule Splitter 完整视图。
    两种模式：
    - 有 resolved_list：用 chunk_map 原文替换 Curator 精简后的具体内容 -> Rule Splitter 拿到的是未经 Curator 精简的完整段落原文
    - 无 resolved_list：回退到 Curator body 原始内容（向后兼容）
    """
    parts: list[str] = []
    # 一、评估基准
    parts.append(f"## 评估基准\n\n{output.evaluation_basis}")
    # 相关业务依据清单
    parts.append("\n## 相关业务依据清单")
    if resolved_list:
        # 一、有 resolved_list：逐条用 chunk_map 原文重建
        resolved_map = {r.evidence_id: r for r in resolved_list}
        for evidence in output.evidences:
            parts.append("")
            resolved = resolved_map.get(evidence.id)
            if resolved and resolved.resolved_chunks:
                # 用 chunk_map 原文重建具体内容
                content_lines = []
                for chunk in resolved.resolved_chunks:
                    content_lines.append(
                        f"**段落[{chunk['chunk_index']}]**\n{chunk['chunk_content']}"
                    )
                rebuilt_content = "\n\n".join(content_lines)
                parts.append(
                    f"### 业务依据 {evidence.id}\n\n"
                    f"{resolved.body_metadata}\n"
                    f"具体内容：\n{rebuilt_content}"
                )
            else:
                # 回退：chunk_map 未命中，使用 Curator body 原始内容
                parts.append(f"### 业务依据 {evidence.id}\n\n{evidence.body}")
    else:
        # 一、无 resolved_list：直接使用 Curator body (向后兼容)
        for evidence in output.evidences:
            parts.append("")
            parts.append(f"### 业务依据 {evidence.id}\n\n{evidence.body}")
    # 一、矛盾提示
    parts.append(f"\n## 矛盾提示\n\n{output.contradictions}")
    return "\n".join(parts)


def format_citations_for_reporter(
    citations: list[dict[str, Any]],
) -> str:
    """将 citations 格式化为 Reporter 提示词中的"可用参考来源"段落。
    输出示例：
    # 可用参考来源
    正文中需要引用时，在句末使用 `[[n]](#ref-n)`，n 为下方序号。
    仅限引用下方列出的来源，严禁编造 URL。

    [1] 信用卡分期业务管理办法 - http://xxx
    [2] 白金卡产品说明书 - http://yyy
    """
    ordered = _dedupe_citations_preserve_order(citations)
    if not ordered:
        return (
            "# 可用参考来源\n\n"
            "当前无可参考来源。报告中请勿使用 `[[n]](#ref-n)` 引用标记。\n"
            "如需引用文档，仅用《文档名》指代。"
        )
    lines = [
        "# 可用参考来源",
        "",
        "正文中需要引用时，在句末使用 `[[n]](#ref-n)`，n 为下方序号。",
        "仅限引用下方列出的来源，严禁编造未列出的来源或 URL。",
        "",
    ]
    for i, c in enumerate(ordered, 1):
        # title = (c.get("title") or "未知文档").strip()
        original_title = (c.get("original_title") or "未知文档").strip()
        url = (c.get("url") or "").strip()
        if url:
            lines.append(f"[{i}] {original_title} - {url}")
        else:
            lines.append(f"[{i}] {original_title}")
    return "\n".join(lines)


def _dedupe_citations_preserve_order(
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """去重并保持插入顺序。"""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        key = _citation_dedup_key(c.get("title", ""), c.get("url"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def format_tool_cache_for_curator(
    cache: list[ToolCallRecord],
    step_title: str = "",
) -> str:
    """将缓存的全部工具调用格式化为 Curator 的 "原始工具返回" 输入。
    输出格式（Markdown）：
    === 第1次调用 (local_search_tool / 语义检索) ===
    [content_md from tool]
    === 第2次调用 (crawl_tool / 全文获取) ===
    [content_md from tool]
    """
    if not cache:
        return "(本步骤未执行任何工具调用)"
    parts: list[str] = []
    if step_title:
        parts.append(f"## {step_title} 的原始工具返回\n")
    for record in cache:
        display_name = TOOL_NAME_DISPLAY.get(record.tool_name, record.tool_name)
        parts.append(
            f"=== 第{record.call_index}次调用 ({record.tool_name} / {display_name}) ==="
        )
        parts.append("")
        parts.append(record.content_md)
        parts.append("")
    return "\n".join(parts).strip()


def extract_chunk_maps_from_cache(
    cache: list[ToolCallRecord],
) -> dict[str, dict[str, str]]:
    """从缓存中提取所有文档的 chunk_map。
    返回: {document_title: {chunk_index: chunk_content}}
    合并策略：
    - crawl/fetch 的 chunk_map（全量）优先
    - local_search 的 chunks 作为补充
    - 同一文档多次出现时取并集
    """
    result: dict[str, dict[str, str]] = defaultdict(dict)
    # Pass 1: 收集 crawl/fetch 的完整 chunk_map (优先级高)
    for record in cache:
        if record.artifact is None:
            continue
        for doc in record.artifact.documents:
            if doc.chunk_map:
                normalized_title = normalize_doc_title(doc.document_title)
                # 完整 chunk_map 直接写入（覆盖 local_search 的部分数据）
                result[normalized_title].update(doc.chunk_map)
    # Pass 2: 补充 local_search 的 chunks (不覆盖已有的)
    for record in cache:
        if record.artifact is None:
            continue
        for doc in record.artifact.documents:
            if doc.chunk_map:
                continue  # 已在 Pass 1 处理
            normalized_title = normalize_doc_title(doc.document_title)
            for chunk in doc.chunks:
                if chunk.chunk_index not in result[normalized_title]:
                    result[normalized_title][chunk.chunk_index] = chunk.chunk_content
    return dict(result)


def extract_document_metadata_from_cache(
    cache: list[ToolCallRecord],
) -> dict[str, dict]:
    """从缓存中提取所有文档的元数据（去重）。
    返回: {document_title: {"url": ..., "is_extracted": ...}}
    """
    metadata: dict[str, dict] = {}
    for record in cache:
        if record.artifact is None:
            continue
        for doc in record.artifact.documents:
            title = doc.document_title
            normalized_title = normalize_doc_title(doc.document_title)
            if normalized_title not in metadata:
                metadata[normalized_title] = {
                    "original_title": title,
                    "document_url": doc.document_url,
                    "description": doc.description,
                    "is_extracted": doc.is_extracted,
                    "source_tools": [],
                }
            metadata[normalized_title]["source_tools"].append(record.tool_name)
            # crawl/fetch 的 is_extracted 优先（它比 local_search 更明确）
            if doc.is_extracted:
                metadata[normalized_title]["is_extracted"] = True
    return metadata


def build_rule_splitter_view_with_resolved(
    output: CuratorOutput,
    resolved_list: list[ResolvedEvidence],
) -> str:
    """用 resolved_chunks (chunk_map 原文) 替换 body 中的具体内容，确保 Rule Splitter 拿到的是未经 Curator 精简的完整段落原文。"""
    parts = [f"\n## 评估基准\n{output.evaluation_basis}", "\n## 相关业务依据清单"]
    for r in resolved_list:
        parts.append("")
        if r.resolved_chunks:
            # 用 chunk_map 原文重建具体内容
            content_lines = []
            for chunk in r.resolved_chunks:
                content_lines.append(
                    f"**段落[{chunk['chunk_index']}]**\n{chunk['chunk_content']}"
                )
            full_content = "\n\n".join(content_lines)
            parts.append(
                f"### 业务依据 {r.evidence_id}\n\n{r.body_metadata}\n具体内容：\n{full_content}"
            )
        else:
            # 回退：使用 Curator body 原始内容
            evidence = output.evidences[int(r.evidence_id) - 1]
            parts.append(
                f"### 业务依据 {r.evidence_id}\n\n{evidence.body}\n（根据段落编号查询段落原文失败）"
            )
    parts.append(f"\n## 矛盾提示\n{output.contradictions}")
    return "\n".join(parts)


def fuzzy_match_chunk_map(
    curator_doc_title: str,
    document_chunk_maps: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    """四级匹配策略查找 chunk_map：
    1. 精确匹配
    2. 归一化后精确匹配
    3. 归一化后包含匹配（一方包含另一方）
    4. 相似度兜底匹配（ratio ≥ 0.9，取最高分）
    返回 chunk_map 或 None。
    注意：每条 evidence 设计上只对应一个文档。Level 4 仅用于防御 LLM 偶尔违反合并规则（在来源字段拼接多个文档名）的情况。
    """
    # Level 1: 精确匹配
    if curator_doc_title in document_chunk_maps:
        return document_chunk_maps[curator_doc_title]

    # Level 2: 归一化匹配
    norm_query = normalize_doc_title(curator_doc_title)
    if not norm_query:
        return None
    if norm_query in document_chunk_maps:
        return document_chunk_maps[norm_query]

    # Level 3: 包含匹配
    for norm_key, chunk_map in document_chunk_maps.items():
        if norm_query in norm_key or norm_key in norm_query:
            logger.info(
                "文档名包含匹配：Curator '%s' ↔ chunk_map '%s'",
                curator_doc_title,
                norm_key,
            )
            return chunk_map

    # Level 4: 相似度兜底匹配（ratio ≥ 0.8，取最高分）
    best_score = 0.0
    best_match = None
    best_key = ""
    for norm_key, chunk_map in document_chunk_maps.items():
        score = SequenceMatcher(None, norm_query, norm_key).ratio()
        if score > best_score:
            best_score = score
            best_match = chunk_map
            best_key = norm_key
    if best_score >= 0.9 and best_match is not None:
        logger.info(
            "文档名相似度匹配(ratio=%.2f)：查询[%s] ↔ chunk_map[%s]",
            best_score,
            curator_doc_title,
            best_key,
        )
        return best_match
    return None


def resolve_all_evidence_chunks(
    curator_output: CuratorOutput,
    document_chunk_maps: dict[str, dict[str, str]],
) -> list[ResolvedEvidence]:
    results = []
    for evidence in curator_output.evidences:
        metadata, content = _find_content_boundary(evidence.body)
        doc_title = evidence.source_document
        # —— 模糊匹配替代精确查询 ——
        chunk_map = fuzzy_match_chunk_map(doc_title, document_chunk_maps)
        if chunk_map is None:
            logger.error(
                "Evidence %s 的文档 '%s' 在 chunk_maps 中未找到（含模糊匹配）。"
                "回退至 body 中的具体内容。chunk_maps 现有 keys: %s",
                evidence.id,
                doc_title,
                list(document_chunk_maps.keys()),
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
            logger.error(
                "Evidence %s 引用段落 %s 在文档 '%s' 的 chunk_map 中未找到。",
                evidence.id,
                missing,
                doc_title,
            )
        results.append(
            ResolvedEvidence(
                evidence_id=evidence.id,
                relevance=evidence.relevance.value,
                source_document=doc_title,
                body_metadata=metadata,
                body_content=content,
                referenced_chunks=evidence.referenced_chunks,
                resolved_chunks=resolved,
                is_expired=evidence.is_expired,
                is_tool_extracted=evidence.is_tool_extracted,
            )
        )
    return results
