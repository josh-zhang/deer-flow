import re
from dataclasses import dataclass

from src.rag.llm_reranker import get_reranked_chunks
from src.rag.retriever import Chunk, ChunkRole


@dataclass
class ExtractedChunk:
    """A chunk in the extraction result with its role annotation."""

    chunk_index: str
    chunk_content: str
    role: ChunkRole


@dataclass
class ExtractionResult:
    """Output of the extraction module — containing three views of the extraction result."""

    # — View 1: Full document in Markdown —
    full_text_md: str
    # — View 2: Extracted chunks in Markdown (with watermark) —
    extracted_text_md: str
    # — View 3: Structured chunk data for programmatic access —
    chunk_map: dict[str, dict[str, str]]

    selected_indices: list[str]
    extracted_chunks: list[ExtractedChunk]

    document_title: str
    total_chunks: int
    is_extracted: bool


def _get_neighbors(
    chunk_index: str,
    ordered_indices: list[str],
    window: int
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


def _expand_with_context():
    pass


def expand_selected_indices_with_context_from_positions(
    selected: list,
    ordered_indices: list,
    window: int,
    n: int,
) -> list[tuple[str, ChunkRole]]:
    """Expand selected indices with context from positional neighbors."""
    expanded: list[tuple[str, ChunkRole]] = []
    sel_set = set(selected)
    for pos, idx in enumerate(ordered_indices):
        if idx in sel_set:
            expanded.append((idx, ChunkRole.SELECTED))
        elif any(
            abs(pos - p) <= window
            for p, s in enumerate(ordered_indices)
            if s in sel_set
        ):
            expanded.append((idx, ChunkRole.CONTEXT))
    return expanded


def get_neighbors_with_context(
    chunk_map: dict[str, Chunk],
    ordered_indices: list[str],
    window: int,
    n: int,
) -> dict[str, str]:
    """Return original document context for the selected indices."""
    result_map: dict[str, str] = {}
    for idx in ordered_indices:
        if idx in chunk_map:
            result_map[idx] = chunk_map[idx]
    return result_map


def _format_full_text_md(
    document_title: str,
    chunks: list[Chunk],
    document_category: str,
) -> str:
    lines = [f"# {document_title}"]
    if document_category:
        lines.append(f"知识库类型: {document_category}")
    for c in chunks:
        content = c.chunk_content.strip()
        lines.append(f"**段落[{c.chunk_index}]**\n{content}")
        lines.append("")

    return "\n".join(lines)


def _format_extracted_text_md(
    document_title: str,
    chunk_map: dict[str, str],
    expanded: list[tuple[str, ChunkRole]],
    total_chunks: int,
    selected_count: int,
    document_category: str,
) -> str:
    lines = [f"【截取结果】以下内容从《{document_title}》的",
             f" {total_chunks} 个段落中选取了 {selected_count} 个相关段落。", "", f"# {document_title}（截取）", "",
             f"知识库类型：{document_category}", ""]

    for idx, role in expanded:
        content = chunk_map.get(idx, "").strip()
        if not content:
            continue
        if role == ChunkRole.SELECTED:
            lines.append(f"**段落[{idx}]**\n{content}")
        elif role == ChunkRole.CONTEXT:
            lines.append(f"**段落[{idx}]**\n【上下文】\n{content}")
        else:
            lines.append(f"**段落[{idx}]**\n【兜底】\n{content}")

        lines.append("")

    return "\n".join(lines)


def _format_fallback_md(
    document_title: str,
    target_questions: str,
    chunks: list[Chunk],
) -> str:
    pass


def extract_qa_windows(
    chunks: list[Chunk],
    target_questions: list[str],
    document_title: str = "",
    context_window: int = 1,
    full_text_threshold: int = 8000,
    document_category: str = "",
) -> ExtractionResult:
    pass
