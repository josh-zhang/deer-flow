from dataclasses import dataclass
from typing import Optional

from src.rag.llm_reranker import get_reranked_chunks
from src.rag.retriever import Chunk, ChunkRole


# —— Data Structures ——————————————————————————————————————

@dataclass
class ExtractedChunk:
    """A chunk in the extraction result with its role annotation."""
    chunk_index: str
    chunk_content: str
    role: ChunkRole


@dataclass
class ExtractionResult:
    """Output of the extraction module, containing three views of the data."""

    # —— View 1: Full document in Markdown ——
    full_text_md: str

    # —— View 2: Extracted chunks in Markdown (with watermark) ——
    extracted_text_md: str

    # —— View 3: Structured chunk data for programmatic access ——
    chunk_map: dict[str, str]  # {chunk_index: chunk_content} for ALL chunks
    selected_indices: list[str]  # Indices selected by LLM
    extracted_chunks: list[ExtractedChunk]  # Ordered extracted chunks with roles

    # —— Metadata ——
    document_title: str
    total_chunks: int
    is_extracted: bool


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


def _format_full_text_md(document_title: str, chunks: list[Chunk], document_category: str) -> str:
    lines = [f"# {document_title}", ""]
    if document_category:
        lines += [f"知识库类型: {document_category}", ""]
    for c in chunks:
        content = c.chunk_content.strip()
        lines.append(f"**段落[{c.chunk_index}]**\n{content}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_extracted_text_md(
    document_title: str,
    chunk_map: dict[str, str],
    expanded: list[tuple[str, ChunkRole]],
    total_chunks: int,
    selected_count: int,
    document_category: str
) -> str:
    lines = [
        (
            f"【截取结果】以下内容从《{document_title}》的"
            f"{total_chunks} 个段落中选取了 {selected_count} 个相关段落"
            f"（含上下文共 {len(expanded)} 个段落）。"
        ),
        "",
        f"# {document_title} (截) ",
        "",
        f"知识库类型: {document_category}",
        "",
    ]
    for idx, role in expanded:
        content = chunk_map.get(idx, "").strip()
        if not content:
            continue
        if role == ChunkRole.SELECTED:
            lines.append(f"**段落[{idx}]**\n{content}")
        elif role == ChunkRole.CONTEXT:
            lines.append(f"**段落[{idx}]** `[上下文]`\n{content}")
        elif role == ChunkRole.FALLBACK:
            lines.append(f"**段落[{idx}]** `[兜底]`\n{content}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_fallback_md(
    document_title: str,
    target_questions: str,
    chunks: list[Chunk]
) -> str:
    first = chunks[0]
    last = chunks[-1] if len(chunks) > 1 else None
    lines = [
        (
            f"【截取结果】基于以下问题未在《{document_title}》中找到直接相关段落。返回文档首尾段落供参考。\n{target_questions}\n"
        ),
        "",
        f"# {document_title} (截 - 无直接相关段落) ",
        "",
        f"**段落[{first.chunk_index}]** `[兜底]`\n{first.chunk_content.strip()}",
    ]

    if last and last.chunk_index != first.chunk_index:
        lines.append("")
        lines.append(f"**段落[{last.chunk_index}]** `[兜底]`\n{last.chunk_content.strip()}")

    return "\n".join(lines).strip()


# —— Public API ——————————————————————————————————————————

def extract_relevant_chunks(
    chunks: list[Chunk],
    target_questions: list[str],
    document_title: str = "未知文档",
    context_window: int = 1,
    full_text_threshold: int = 8000,
    document_category: str = "",
) -> Optional[ExtractionResult]:
    """
    Main entry point.

    Args:
        chunks: Pre-segmented document chunks (chunk_index is str).
        target_questions: List of questions to select relevant chunks.
        document_title: Document title for Markdown headers.
        context_window: Context chunks before/after each selected chunk. Default 1.
        full_text_threshold: If total chars <= threshold, return full text. Default 8000.
        document_category: Document category

    Returns:
        ExtractionResult with three views.
    """
    # —— Edge case: empty ——
    if not chunks:
        return ExtractionResult(
            full_text_md=f"# {document_title}\n\n (文档内容为空) ",
            extracted_text_md=f"# {document_title}\n\n (文档内容为空) ",
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

    # —— View 1: Always build full text ——
    full_text_md = _format_full_text_md(document_title, chunks, document_category)

    # —— Short document: return full text, no extraction ——
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

    # —— Parse ——
    tq_text = "\n".join(f"- {q}" for q in target_questions)

    unsorted_list = list()
    for c in chunks:
        unsorted_list.append(c.chunk_content)

    ranked_docs = get_reranked_chunks(tq_text, unsorted_list, need_sorting=False)

    selected_indices = list()
    for doc in ranked_docs:
        score = doc['score']
        if score > 0:
            # 如果文档 相关 或 不确定
            original_index = doc['original_index']
            chunk = chunks[original_index]
            selected_indices.append(chunk.chunk_index)

    # Restore document order
    index_position = {idx: pos for pos, idx in enumerate(ordered_indices)}
    selected_indices.sort(key=lambda x: index_position.get(x, float("inf")))

    # —— Fallback: no selection ——
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
            extracted_text_md=_format_fallback_md(document_title, tq_text, chunks),
            chunk_map=chunk_map,
            selected_indices=[],
            extracted_chunks=fallback_chunks,
            document_title=document_title,
            total_chunks=total_chunks,
            is_extracted=True,
        )

    # —— Expand with context ——
    expanded = _expand_with_context(selected_indices, ordered_indices, context_window)
    extracted_chunks = [
        ExtractedChunk(idx, chunk_map[idx], role)
        for idx, role in expanded
    ]

    # —— View 2: Extracted Markdown ——
    extracted_text_md = _format_extracted_text_md(
        document_title, chunk_map, expanded, total_chunks, len(selected_indices), document_category
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
