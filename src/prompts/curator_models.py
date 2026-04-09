"""Evidence Curator 输出的领域模型（极简版）。

设计哲学：只有影响代码分支的字段（relevance）才建模为枚举，
其他所有字段保持为自由文本字符串，由视图生成代码原样传递。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Relevance(str, Enum):
    DIRECT = "直接相关"
    INDIRECT = "间接相关"
    UNCERTAIN = "存疑"


class EvidenceItem(BaseModel):
    relevance: Relevance = Relevance.UNCERTAIN
    body: str = ""
    id: int = 0

    @field_validator("relevance", mode="before")
    @classmethod
    def _fuzzy_relevance(cls, v: object) -> str:
        if not isinstance(v, str):
            return Relevance.UNCERTAIN.value

        v = v.strip()

        # 精确匹配
        for member in Relevance:
            if v == member.value:
                return v

        # 模糊匹配
        FUZZY_MAP = {
            "直接": Relevance.DIRECT.value,
            "间接": Relevance.INDIRECT.value,
        }
        for keyword, target in FUZZY_MAP.items():
            if keyword in v:
                return target

        # 兜底：保守归为存疑（宁多勿漏）
        return Relevance.UNCERTAIN.value


class CuratorOutput(BaseModel):
    evaluation_basis: str = ""
    evidences: list[EvidenceItem] = Field(default_factory=list)
    contradictions: str = "未发现跨片段矛盾。"
    discarded: str = "无丢弃片段。"
    coverage: str = ""
    searcher_reference: str = ""

    # 代码计算
    total_input_count: int = 0
    retained_count: int = 0
    direct_count: int = 0
    indirect_count: int = 0
    uncertain_count: int = 0
    discarded_count: int = 0
