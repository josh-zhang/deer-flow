from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

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
