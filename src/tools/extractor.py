"""
Target-question-based chunk extraction module.

Given a pre-chunked document and a list of target questions,
uses an LLM to identify relevant chunk indices, then deterministically
extracts the corresponding original text with context expansion.

Produces three output views:
1. full_text_md      – Markdown-formatted full document (all chunks)
2. extracted_text_md – Markdown-formatted extraction result (selected + context chunks)
3. chunk_map         – Dict mapping chunk_index → chunk_content for programmatic access
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ━━ Data Structures ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Chunk:
    """A single document chunk produced by the upstream segmentation function."""
    chunk_index: str        # str type — e.g. "1", "2a", "S3"
    chunk_content: str


class ChunkRole(str, Enum):
    """Role of a chunk in the extraction result."""
    SELECTED = "selected"
    CONTEXT = "context"
    FALLBACK = "fallback"


@dataclass
class ExtractedChunk:
    """A chunk in the extraction result with its role annotation."""
    chunk_index: str
    chunk_content: str
    role: ChunkRole


@dataclass
class ExtractionResult:
    """Output of the extraction module, containing three views of the data."""

    # ── View 1: Full document in Markdown ──
    full_text_md: str

    # ── View 2: Extracted chunks in Markdown (with watermark) ──
    extracted_text_md: str

    # ── View 3: Structured chunk data for programmatic access ──
    chunk_map: dict[str, str]                # {chunk_index: chunk_content} for ALL chunks
    selected_indices: list[str]              # Indices selected by LLM
    extracted_chunks: list[ExtractedChunk]   # Ordered extracted chunks with roles

    # ── Metadata ──
    document_title: str
    total_chunks: int
    is_extracted: bool


# ━━ Prompt Template ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """\
你是一个文档段落相关性标注工具。你的唯一任务是：判断哪些编号段落与 target_questions 相关，并输出相关段落的编号。

【铁律——严格遵守，不得违反】
1. 只输出相关段落的编号，用英文逗号分隔（如 1,3,7,8）。
2. 绝对不要输出任何段落内容。
3. 绝对不要回答 target_questions。
4. 绝对不要输出编号列表以外的任何文字（不要解释、不要总结、不要前言后语）。
5. 如果没有任何段落与 target_questions 相关，只输出一个字：无
6. 宁多选勿漏选——不确定是否相关时，选入。
7. 输出的编号必须是下方【编号段落】中实际出现的编号。合法编号列表：{valid_indices}"""

USER_PROMPT = """\
【target_questions】
{target_questions}

【编号段落（共 {total_chunks} 段）】
{numbered_chunks}

【相关段落编号】"""


# ━━ Internal Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_numbered_text(chunks: list[Chunk]) -> str:
    lines = []
    for c in chunks:
        content = c.chunk_content.strip()
        lines.append(f"[{c.chunk_index}] {content}")
    return "\n\n".join(lines)


def _parse_llm_output(raw_output: str, valid_indices: set[str]) -> list[str]:
    """
    Parse LLM output into a validated list of chunk index strings.

    Strategy: extract all tokens that match a valid index.
    This handles arbitrary index formats (numeric, alphanumeric, etc.)
    """
    text = raw_output.strip()

    if not text or text in ("无", "无。", "None", "none", "N/A", "[]", "空"):
        return []

    # Split on common delimiters: comma, Chinese comma, whitespace, newline
    tokens = re.split(r"[,，\s\n]+", text)
    tokens = [t.strip().strip("[]()（）") for t in tokens if t.strip()]

    result = []
    seen: set[str] = set()
    invalid_count = 0

    for token in tokens:
        if token in valid_indices:
            if token not in seen:
                seen.add(token)
                result.append(token)
        else:
            invalid_count += 1

    if invalid_count:
        logger.warning(
            "Filtered %d invalid indices from LLM output. Valid set: %s",
            invalid_count,
            valid_indices,
        )

    # Return in original chunk order
    index_order = {idx: i for i, idx in enumerate(
        sorted(valid_indices, key=lambda x: list(valid_indices).index(x)
               if x in valid_indices else float("inf"))
    )}
    # Actually, preserve insertion order from chunks
    return result


def _get_neighbors(
    chunk_index: str,
    ordered_indices: list[str],
    window: int,
) -> list[str]:
    """Get neighboring indices within window based on positional order."""
    try:
        pos = ordered_indices.index(chunk_index)
    except ValueError:
        return []

    neighbors = []
    for offset in range(-window, window + 1):
        neighbor_pos = pos + offset
        if 0 <= neighbor_pos < len(ordered_indices):
            neighbors.append(ordered_indices[neighbor_pos])
    return neighbors


def _expand_with_context(
    selected_indices: list[str],
    ordered_indices: list[str],
    context_window: int,
) -> list[tuple[str, ChunkRole]]:
    """Expand selected indices with context from positional neighbors."""
    selected_set = set(selected_indices)
    result_map: dict[str, ChunkRole] = {}

    for idx in selected_indices:
        result_map[idx] = ChunkRole.SELECTED
        neighbors = _get_neighbors(idx, ordered_indices, context_window)
        for n in neighbors:
            if n not in result_map:
                result_map[n] = ChunkRole.CONTEXT

    # Return in original document order
    return [
        (idx, result_map[idx])
        for idx in ordered_indices
        if idx in result_map
    ]


# ━━ Markdown Formatters ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _format_full_text_md(document_title: str, chunks: list[Chunk]) -> str:
    lines = [f"# {document_title}", ""]
    for c in chunks:
        content = c.chunk_content.strip()
        lines.append(f"**[{c.chunk_index}]**\n{content}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_extracted_text_md(
    document_title: str,
    chunk_map: dict[str, str],
    expanded: list[tuple[str, ChunkRole]],
    total_chunks: int,
    selected_count: int,
) -> str:
    lines = [
        (
            f"【截取结果】以下内容从《{document_title}》的"
            f" {total_chunks} 个段落中选取了 {selected_count} 个相关段落"
            f"（含上下文共 {len(expanded)} 个段落）。"
        ),
        "",
        f"# {document_title}（截取）",
        "",
    ]

    for idx, role in expanded:
        content = chunk_map.get(idx, "").strip()
        if not content:
            continue
        if role == ChunkRole.SELECTED:
            lines.append(f"**[{idx}]**\n{content}")
        elif role == ChunkRole.CONTEXT:
            lines.append(f"**[{idx}]** `[上下文]`\n{content}")
        elif role == ChunkRole.FALLBACK:
            lines.append(f"**[{idx}]** `[兜底]`\n{content}")
        lines.append("")

    return "\n".join(lines).strip()


def _format_fallback_md(
    document_title: str,
    chunks: list[Chunk],
    total_chunks: int,
) -> str:
    first = chunks[0]
    last = chunks[-1] if len(chunks) > 1 else None

    lines = [
        (
            f"【截取结果】基于 target_questions 未在《{document_title}》中"
            f"找到直接相关段落。以下返回文档首尾段落供参考。"
        ),
        "",
        f"# {document_title}（截取 — 无直接相关段落）",
        "",
        f"**[{first.chunk_index}]** `[兜底]`\n{first.chunk_content.strip()}",
    ]

    if last and last.chunk_index != first.chunk_index:
        lines.append("")
        lines.append(f"**[{last.chunk_index}]** `[兜底]`\n{last.chunk_content.strip()}")

    return "\n".join(lines).strip()


# ━━ Public API ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_relevant_chunks(
    chunks: list[Chunk],
    target_questions: list[str],
    llm_call: callable,
    document_title: str = "未知文档",
    context_window: int = 1,
    full_text_threshold: int = 5000,
) -> ExtractionResult:
    """
    Main entry point.

    Args:
        chunks:              Pre-segmented document chunks (chunk_index is str).
        target_questions:    List of questions to select relevant chunks.
        llm_call:            Callable: (system: str, user: str) -> str.
        document_title:      Document title for Markdown headers.
        context_window:      Context chunks before/after each selected chunk. Default 1.
        full_text_threshold: If total chars <= threshold, return full text. Default 5000.

    Returns:
        ExtractionResult with three views.
    """
    # ── Edge case: empty ──
    if not chunks:
        return ExtractionResult(
            full_text_md=f"# {document_title}\n\n（文档内容为空）",
            extracted_text_md=f"# {document_title}\n\n（文档内容为空）",
            chunk_map={},
            selected_indices=[],
            extracted_chunks=[],
            document_title=document_title,
            total_chunks=0,
            is_extracted=False,
        )

    total_chunks = len(chunks)
    ordered_indices = [c.chunk_index for c in chunks]
    chunk_map = {c.chunk_index: c.chunk_content for c in chunks}
    total_chars = sum(len(c.chunk_content) for c in chunks)

    # ── View 1: Always build full text ──
    full_text_md = _format_full_text_md(document_title, chunks)

    # ── Short document: return full text, no extraction ──
    if total_chars <= full_text_threshold:
        all_extracted = [
            ExtractedChunk(
                chunk_index=c.chunk_index,
                chunk_content=c.chunk_content,
                role=ChunkRole.SELECTED,
            )
            for c in chunks
        ]
        return ExtractionResult(
            full_text_md=full_text_md,
            extracted_text_md=full_text_md,
            chunk_map=chunk_map,
            selected_indices=ordered_indices,  # all chunk indices as-is
            extracted_chunks=all_extracted,
            document_title=document_title,
            total_chunks=total_chunks,
            is_extracted=False,
        )

    # ── Build prompt ──
    valid_indices_set = set(ordered_indices)
    tq_text = "\n".join(f"- {q}" for q in target_questions)
    numbered_text = _build_numbered_text(chunks)
    valid_indices_display = ", ".join(ordered_indices)

    system = SYSTEM_PROMPT.format(
        valid_indices=valid_indices_display,
    )
    user = USER_PROMPT.format(
        target_questions=tq_text,
        total_chunks=total_chunks,
        numbered_chunks=numbered_text,
    )

    # ── Call LLM ──
    raw_output = llm_call(system, user)
    logger.info("LLM chunk selection output: %r", raw_output)

    # ── Parse ──
    selected_indices = _parse_llm_output(raw_output, valid_indices_set)

    # Restore document order
    index_position = {idx: pos for pos, idx in enumerate(ordered_indices)}
    selected_indices.sort(key=lambda x: index_position.get(x, float("inf")))

    # ── Fallback: no selection ──
    if not selected_indices:
        first = chunks[0]
        last = chunks[-1] if total_chunks > 1 else None
        fallback_chunks = [
            ExtractedChunk(first.chunk_index, first.chunk_content, ChunkRole.FALLBACK)
        ]
        if last and last.chunk_index != first.chunk_index:
            fallback_chunks.append(
                ExtractedChunk(last.chunk_index, last.chunk_content, ChunkRole.FALLBACK)
            )

        return ExtractionResult(
            full_text_md=full_text_md,
            extracted_text_md=_format_fallback_md(document_title, chunks, total_chunks),
            chunk_map=chunk_map,
            selected_indices=[],
            extracted_chunks=fallback_chunks,
            document_title=document_title,
            total_chunks=total_chunks,
            is_extracted=True,
        )

    # ── Expand with context ──
    expanded = _expand_with_context(selected_indices, ordered_indices, context_window)

    extracted_chunks = [
        ExtractedChunk(idx, chunk_map[idx], role)
        for idx, role in expanded
    ]

    # ── View 2: Extracted Markdown ──
    extracted_text_md = _format_extracted_text_md(
        document_title, chunk_map, expanded, total_chunks, len(selected_indices),
    )

    return ExtractionResult(
        full_text_md=full_text_md,
        extracted_text_md=extracted_text_md,
        chunk_map=chunk_map,
        selected_indices=selected_indices,
        extracted_chunks=extracted_chunks,
        document_title=document_title,
        total_chunks=total_chunks,
        is_extracted=True,
    )


# ━━ Utility: Resolve chunks by indices from downstream output ━━━

def resolve_chunks_by_indices(
    result: ExtractionResult,
    indices: list[str],
) -> list[dict]:
    """
    Given an ExtractionResult and a list of chunk indices (e.g. from Curator),
    return corresponding chunk contents for Rule Splitter / Analyst.

    Args:
        result:  The ExtractionResult.
        indices: Chunk index strings selected by Curator.

    Returns:
        List of {"chunk_index": str, "chunk_content": str} in document order.
    """
    # Build order map from original chunks
    all_ordered = list(result.chunk_map.keys())
    order_map = {idx: pos for pos, idx in enumerate(all_ordered)}

    resolved = []
    for idx in indices:
        idx_str = str(idx)
        if idx_str in result.chunk_map:
            resolved.append({
                "chunk_index": idx_str,
                "chunk_content": result.chunk_map[idx_str],
            })
        else:
            logger.warning(
                "Curator referenced chunk [%s] not found in '%s' (total: %d). Skipping.",
                idx_str, result.document_title, result.total_chunks,
            )

    resolved.sort(key=lambda x: order_map.get(x["chunk_index"], float("inf")))
    return resolved