"""
Evidence Curator 结构化输出模型
用于 JSON 解析、视图构建和跨 Agent 信息路由
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ────────────────────────── 枚举类型 ──────────────────────────


class Relevance(str, Enum):
    """五级相关性等级"""
    DIRECT = "直接相关"
    INDIRECT = "间接相关"
    UNCERTAIN_HIGH = "存疑-高"
    UNCERTAIN_LOW = "存疑-低"
    IRRELEVANT = "明确无关"  # 仅用于丢弃清单标注，不出现在保留依据中


class ContentType(str, Enum):
    REGULATION = "条款规则"
    FEE_TABLE = "费率表"
    PROCEDURE = "操作流程"
    FAQ = "FAQ"
    PRODUCT_SPEC = "产品说明"
    OTHER = "其他"


class CoverageStatus(str, Enum):
    DIRECT_EVIDENCE = "有直接依据"
    INDIRECT_ONLY = "仅有间接依据"
    NO_EVIDENCE = "无依据"


class OverallCoverage(str, Enum):
    FULL = "充分覆盖"
    PARTIAL = "部分覆盖"
    NONE = "未覆盖"


# ────────────────────────── 子模型 ──────────────────────────


class EvaluationBasis(BaseModel):
    """评估基准：从当前步骤的 description 中解析"""
    search_content: str = Field(
        description="检索内容（复述当前步骤 description 原文）"
    )
    specific_targets: list[str] = Field(
        description="具体标的列表，从 description 中逐项枚举"
    )
    related_info_needs: str = Field(
        description="关联信息需求（例外条款、禁止规则、前置条件、适用范围限制等）"
    )


class EvidenceItem(BaseModel):
    """单条保留的业务依据"""
    id: int = Field(
        description="依据序号，从 1 开始连续编号"
    )
    source: str = Field(
        description="来源文档名（与 Searcher 输出中的文档名一致）"
    )
    merged_from: Optional[list[int]] = Field(
        default=None,
        description="如果该依据由多个同源片段合并产生，记录原始片段的 Searcher 输出序号；否则为 null"
    )
    relevance: Relevance = Field(
        description="相关性等级（五级）"
    )
    relevance_reason: str = Field(
        description="相关性判定理由（1-2 句话）"
    )
    completeness: str = Field(
        description=(
            "信息完整性：完整 / 片段截断，已获取全文 / 片段截断，无法获取全文。"
            "如执行了段落级精简则追加'（已精简）'"
        )
    )
    quality_tags: list[str] = Field(
        description=(
            "信息质量标签列表。无质量问题时为 ['清晰']；"
            "有问题时可多选：模糊指代 / 适用范围不明 / 疑似过时 / 残缺 / 内部矛盾"
        )
    )
    content_type: ContentType = Field(
        description="内容类型"
    )
    has_conditional_branch: bool = Field(
        description="是否包含条件分支（if-then / 当...时 / 除...外 等条件结构）"
    )
    summary: str = Field(
        description=(
            "要点概述。写作深度按相关性分层：\n"
            "- 直接相关：1-2 句，覆盖了哪个标的 + 信息类型。\n"
            "- 间接相关：2-3 句，须用「」引用原文中最核心的 1-2 条业务规则/数值。\n"
            "- 存疑-高：2-3 句，须说明存疑原因并用「」引用原文关键陈述。\n"
            "- 存疑-低：2-3 句，须说明存疑原因并用「」引用原文关键陈述。"
        )
    )
    content: Optional[str] = Field(
        default=None,
        description=(
            "具体内容（保留段落的原始文本，逐字保持原文，无关段落用省略标记替代）。\n"
            "- 直接相关 / 间接相关 / 存疑-高：必须提供。\n"
            "- 存疑-低：必须为 null（不输出具体内容）。"
        )
    )


class DiscardedItem(BaseModel):
    """被丢弃的片段"""
    id: int = Field(description="序号，从 1 开始")
    source: str = Field(description="来源文档名")
    content_summary: str = Field(description="内容摘要（≤30字）")
    discard_reason: str = Field(description="判定为明确无关的具体理由")


class ContradictionNote(BaseModel):
    """跨片段矛盾标注"""
    id: int = Field(description="矛盾编号，从 1 开始")
    evidence_ids: list[int] = Field(
        description="涉及的业务依据序号列表（对应 EvidenceItem.id）"
    )
    description: str = Field(
        description="矛盾描述：分别说了什么，矛盾点是什么"
    )


class TargetCoverage(BaseModel):
    """单个标的覆盖情况"""
    target: str = Field(description="标的名称（与 evaluation_basis.specific_targets 中的对应项一致）")
    status: CoverageStatus = Field(description="覆盖状态")


class CoverageAssessment(BaseModel):
    """检索覆盖度自评"""
    overall: OverallCoverage = Field(description="整体覆盖度")
    target_details: list[TargetCoverage] = Field(
        description="逐标的覆盖详情（与 evaluation_basis.specific_targets 一一对应）"
    )
    negative_rules_covered: bool = Field(
        description="是否有否定性规则/例外条款的相关依据"
    )
    uncovered_aspects: str = Field(
        description="未覆盖方面（列出无依据的标的，全部覆盖则填'无'）"
    )
    possible_reasons: str = Field(
        description=(
            "可能原因：检索语句未命中 / 信息可能分散在多个文档中 / "
            "知识库可能未收录（无法确认） / 无法判断"
        )
    )


class DiscoveredDocument(BaseModel):
    """已发现的文档"""
    name: str = Field(description="文档名")
    topic: str = Field(description="该文档涉及的主要业务主题（≤15字）")


class SearcherReference(BaseModel):
    """供后续 Searcher 使用的精简摘要"""
    search_topic: str = Field(
        description="本步骤检索主题（一句话概括）"
    )
    covered_targets: list[str] = Field(
        description="已覆盖标的列表（有直接依据的标的）"
    )
    uncovered_targets: list[str] = Field(
        description="未覆盖标的列表（无依据的标的，全部覆盖则为空列表）"
    )
    discovered_documents: list[DiscoveredDocument] = Field(
        description="已发现的不重复文档清单"
    )
    pending_leads: str = Field(
        description=(
            "待追踪线索：片段中引用但本步骤未获取的文档名称或编号；"
            "发现的与调查原始问题相关的重要业务别名或术语。无则填'无'"
        )
    )


# ────────────────────────── 顶层输出模型 ──────────────────────────


class CuratorOutput(BaseModel):
    """Evidence Curator 完整结构化输出"""

    # ── 评估基准 ──
    evaluation_basis: EvaluationBasis

    # ── 统计摘要 ──
    total_input_count: int = Field(
        description="Searcher 筛选后输入的总片段数"
    )
    retained_count: int = Field(description="保留的总依据数")
    direct_count: int = Field(description="直接相关依据数")
    indirect_count: int = Field(description="间接相关依据数")
    uncertain_high_count: int = Field(description="存疑-高依据数")
    uncertain_low_count: int = Field(description="存疑-低依据数")
    discarded_count: int = Field(description="丢弃的片段数")

    # ── 业务依据清单（按相关性降序排列）──
    evidences: list[EvidenceItem] = Field(
        description="保留的业务依据列表，按相关性降序排列：直接相关 → 间接相关 → 存疑-高 → 存疑-低"
    )

    # ── 矛盾提示 ──
    contradictions: list[ContradictionNote] = Field(
        description="跨片段矛盾列表。无矛盾时为空列表"
    )

    # ── 丢弃清单 ──
    discarded_items: list[DiscardedItem] = Field(
        description="被丢弃的片段列表。无丢弃时为空列表"
    )

    # ── 覆盖度自评 ──
    coverage: CoverageAssessment

    # ── 后续步骤检索参考 ──
    searcher_reference: SearcherReference