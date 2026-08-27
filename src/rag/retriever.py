import abc
import logging
import hashlib
import re
from typing import Final, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 预编译正则表达式以提升多次调用时的性能
_PATTERN: Final[re.Pattern[str]] = re.compile(r'^#### ', re.MULTILINE)


class ToolChunk(BaseModel):
    """工具返回中的单个文档段落。"""
    chunk_index: str
    chunk_content: str


class ToolDocumentReturn(BaseModel):
    """单个文档维度的工具返回数据。三种工具共用此结构。"""
    document_title: str
    document_url: str | None = None
    description: str | None = None  # local_search 的 description 元数据
    chunks: list[ToolChunk] = Field(default_factory=list)
    # —— 截取元数据（仅 crawl/fetch 有值）——
    is_extracted: bool = False  # True = 执行了截取
    chunk_map: dict[str, str] | None = None  # 全量 chunk_index -> content 映射


class ToolCallArtifact(BaseModel):
    """单次工具调用的结构化 artifact，存入 ToolMessage.artifact。"""
    tool_type: str  # "local_search" | "crawl" | "fetch"
    documents: list[ToolDocumentReturn] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """Hook 缓存中的单条记录。"""
    call_index: int
    tool_name: str
    content_md: str  # Searcher 可见的 Markdown
    artifact: ToolCallArtifact | None = None


@dataclass
class Chunk:
    chunk_index: str
    chunk_content: str
    similarity: float

    def __init__(self, chunk_id: str, content: str, similarity: float = 0):
        self.chunk_index = chunk_id
        self.chunk_content = content
        self.similarity = similarity


class ChunkRole(str, Enum):
    """Role of a chunk in the extraction result."""
    SELECTED = "selected"
    CONTEXT = "context"
    FALLBACK = "fallback"


class Document:
    """ Document is a class that represents a document. """
    id: str
    url: str = ""
    title: str = ""
    description: str = ""
    chunks: list[Chunk] = []

    def __init__(
        self,
        id: str,
        url: str,
        title: str,
        description: str,
        chunks: list[Chunk] = [],
    ):
        self.id = id
        self.url = url
        self.title = title
        self.description = description
        self.chunks = chunks


class Resource(BaseModel):
    """ Resource is a class that represents a resource. """
    uri: str = Field(..., description="The URI of the resource")
    title: str = Field(..., description="The title of the resource")
    description: str | None = Field("", description="The description of the resource")


class Retriever(abc.ABC):
    @abc.abstractmethod
    def list_resources(self, query: str | None = None) -> list[Resource]:
        pass

    @abc.abstractmethod
    async def list_resources_async(self, query: str | None = None) -> list[Resource]:
        pass

    @abc.abstractmethod
    def query_relevant_documents(
        self,
        query: str,
        top_k: int,
        background: str,
        resources: list[Resource] = [],
    ) -> dict[str, dict]:
        """ Query relevant documents from the resources (synchronous version). """
        pass

    @abc.abstractmethod
    async def query_relevant_documents_async(
        self,
        query: str,
        top_k: int,
        background: str,
        resources: list[Resource] = [],
    ) -> dict[str, dict]:
        pass

    def ingest_file(self, file_content: bytes, filename: str, **kwargs) -> Resource:
        raise NotImplementedError("ingest_file is not implemented")


def generate_checksum(text: str, algorithm: str = 'sha256') -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    try:
        hash_func = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm '{algorithm}'.") from e
    # Encode the string to bytes and update the hash object
    hash_func.update(text.encode('utf-8'))
    # Return the hexadecimal digest
    return hash_func.hexdigest()


def format_crawl_fetch_return(
    extraction_result,
    tool_type: str = "crawl",
    document_url: str | None = None,
    description: str | None = None,
) -> Tuple[str, Dict]:
    """ 将 ExtractionResult 格式化为 (content_md, artifact_dict)。
    Args:
        extraction_result: 截取模块的输出。
        tool_type: "crawl" 或 "fetch"。
        document_url: 文档 URL。
        description: 文档分类描述。
    """
    # —— content_md: Searcher 可见 ——
    if extraction_result.is_extracted:
        content_md = extraction_result.extracted_text_md
    else:
        content_md = extraction_result.full_text_md

    # —— artifact: 结构化数据 ——
    tool_chunks = [
        ToolChunk(
            chunk_index=ec.chunk_index,
            chunk_content=ec.chunk_content
        )
        for ec in extraction_result.extracted_chunks
    ]
    doc = ToolDocumentReturn(
        document_title=extraction_result.document_title,
        document_url=document_url,
        description=description,
        chunks=tool_chunks,
        is_extracted=extraction_result.is_extracted,
        chunk_map=extraction_result.chunk_map if extraction_result.chunk_map else None,
    )
    artifact = ToolCallArtifact(tool_type=tool_type, documents=[doc])
    return content_md, artifact.model_dump()


def format_local_search_return(
    doc_groups: dict[str, dict]
) -> Tuple[str, Dict]:
    # —— 构建 content_md (Searcher 可见) ——
    md_parts: list[str] = []
    artifact_docs: list[ToolDocumentReturn] = []
    for doc_idx, (title, doc_data) in enumerate(doc_groups.items(), 1):
        # Markdown header
        url_part = f" | url: {doc_data['document_url']}" if doc_data["document_url"] else ""
        desc_part = f" | {doc_data['description']}" if doc_data["description"] else ""
        md_parts.append(f"**文档{doc_idx}** - 《{title}》{desc_part}{url_part}")
        md_parts.append("")
        category = doc_data["description"].strip()
        if category:
            md_parts.append(f"知识库类型: {category}")
            md_parts.append("")
        chunks = doc_data["chunks"]
        tool_chunks: list[ToolChunk] = []
        for chunk in chunks:
            idx = chunk["chunk_index"]
            content = chunk["chunk_content"].strip()
            md_parts.append(f"**段落[{idx}]**\n{content}")
            md_parts.append("")
            tool_chunks.append(ToolChunk(chunk_index=idx, chunk_content=content))
        md_parts.append("---")
        md_parts.append("")
        artifact_docs.append(ToolDocumentReturn(
            document_title=title,
            document_url=doc_data["document_url"],
            description=doc_data["description"],
            chunks=tool_chunks,
            is_extracted=False,
            chunk_map=None,
        ))
    content_md = "\n".join(md_parts).strip()
    artifact = ToolCallArtifact(tool_type="local_search", documents=artifact_docs)
    return content_md, artifact.model_dump()


def remove_line_start_hashes(ttt: str) -> str:
    if not isinstance(ttt, str):
        raise TypeError(f"Expected a string, got {type(ttt).__name__}")
    # re.MULTILINE 使 ^ 匹配每一行的开头，而非仅整个字符串的开头
    return _PATTERN.sub('', ttt)
