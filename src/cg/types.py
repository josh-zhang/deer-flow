from __future__ import annotations

import operator
from dataclasses import field
from enum import Enum
from typing import Annotated, Any

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  State 字段 Reducer（用于并行分支合并）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _merge_dict_maps(
    existing: dict[str, dict],
    new: dict[str, dict],
) -> dict[str, dict]:
    """将两个嵌套 dict 合并，内层 dict 取并集（新值覆盖旧值）。

    用于 document_chunk_maps 和 document_metadata 的并行分支合并。
    """
    result = {k: dict(v) if isinstance(v, dict) else v for k, v in existing.items()}
    for key, val in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = {**result[key], **val}
        else:
            result[key] = val if isinstance(val, dict) else val
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CG 输入模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CGInput(BaseModel):
    """CG 流水线输入参数"""
    product_name: str = Field(description="产品名称")
    product_type: str = Field(description="产品类型，如联名信用卡、分期产品")
    campaign_name: str = Field(default="", description="营销活动名称（如有）")
    channels: list[str] = Field(default_factory=list, description="渠道列表，如企业微信、全民生活APP、短信")
    personas: list[str] = Field(default_factory=list, description="客群列表，如山姆会员/家庭消费群体")
    scene_empathy: bool = Field(default=True, description="是否启用场景化共情")
    relationship_temperature: str = Field(default="warm", description="关系温度: cold/warm/hot/rm_known")
    privacy_boundary: str = Field(default="standard", description="隐私边界: strict/standard/relaxed")
    ab_test: bool = Field(default=False, description="是否输出A/B测试变体（启用后 Copy Writer 为有 AB 方案的组合生成两版母版）")
    historical_ab_summary: str = Field(default="", description="历史 AB 实验结论（自然语言，按客群×渠道分组，由运营人员填写）")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 1 产品事实模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProductBenefit(BaseModel):
    """单条产品权益"""
    name: str = Field(default="", description="权益名称")
    description: str = Field(default="", description="权益描述（逐字复制原文）")
    numeric_value: str = Field(default="", description="数值（逐字复制原文）")
    conditions: list[str] = Field(default_factory=list, description="限制条件列表")
    validity_period: str = Field(default="", description="有效期（如有）")
    source_document: str = Field(default="", description="来源文档名")
    source_chunks: list[str] = Field(default_factory=list, description="引用段落编号列表，如['[3]', '[5]']")


class CampaignInfo(BaseModel):
    """活动信息"""
    name: str = Field(default="", description="活动名称")
    period: str = Field(default="", description="活动期间")
    details: str = Field(default="", description="活动详情")


class ProductFacts(BaseModel):
    """产品事实完整数据（来自 Fact Miner）"""
    product_name: str = Field(default="", description="知识库中的产品标准全称")
    product_type: str = Field(default="", description="产品类型")
    card_level: str = Field(default="", description="卡等级（如有）")
    annual_fee_policy: str = Field(default="", description="年费政策原文")
    annual_fee_source: str = Field(default="", description="年费信息来源段落编号")
    benefits: list[ProductBenefit] = Field(default_factory=list, description="权益列表")
    risk_disclosures: list[str] = Field(default_factory=list, description="法定披露文本列表")
    campaign_info: CampaignInfo = Field(default_factory=CampaignInfo, description="营销活动信息")
    customer_service_hotline: str = Field(default="", description="客服热线")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 1 合规规则模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ComplianceLevel(str, Enum):
    L1_PROHIBITED = "L1-红线禁止"
    L2_MANDATORY = "L2-强制披露"
    L3_CAUTIOUS = "L3-审慎表达"
    L4_FREE = "L4-自由创作"


class ComplianceRule(BaseModel):
    """单条合规规则"""
    rule_id: str = Field(default="", description="规则ID，如L1-001")
    level: str = Field(default="", description="严格度等级")
    category: str = Field(default="", description="规则类别，如绝对化用语/收益承诺/信息披露")
    description: str = Field(default="", description="规则描述")
    source_type: str = Field(default="", description="外部法规 或 行内规定")
    source_document: str = Field(default="", description="来源文档名")
    source_chunks: list[str] = Field(default_factory=list, description="引用段落编号")
    compliant_templates: list[str] = Field(default_factory=list, description="合规表达模板（L2/L3适用）")
    violation_examples: list[str] = Field(default_factory=list, description="违规示例")


class ComplianceRuleSet(BaseModel):
    """完整合规规则集（来自 Rule Miner）"""
    l1_rules: list[ComplianceRule] = Field(default_factory=list, description="L1 红线禁止规则")
    l2_rules: list[ComplianceRule] = Field(default_factory=list, description="L2 强制披露规则")
    l3_rules: list[ComplianceRule] = Field(default_factory=list, description="L3 审慎表达规则")
    channel_rules: dict[str, list[str]] = Field(default_factory=dict, description="渠道特有合规要求")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 2 文案策略模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PersonaStrategy(BaseModel):
    """客群差异化策略"""
    persona_name: str = Field(default="", description="客群名称")
    preferred_benefits: list[str] = Field(default_factory=list, description="优先权益列表")
    marketing_hooks: list[str] = Field(default_factory=list, description="推荐营销钩子类型")
    scene_direction: str = Field(default="", description="场景化方向")
    tone: str = Field(default="", description="语气调性")
    conflict_resolution: str = Field(default="", description="权益冲突解决策略")
    taste_profile: str = Field(default="", description="品味特征描述")
    oily_sensitivity: str = Field(default="medium", description="对促销感的耐受度: high/medium/low")
    formality_range: str = Field(default="", description="正式度区间")
    forbidden_expressions: list[str] = Field(default_factory=list, description="禁用表达")
    preferred_expressions: list[str] = Field(default_factory=list, description="偏好表达")


class ChannelSpec(BaseModel):
    """渠道规范"""
    channel_name: str = Field(default="", description="渠道名称")
    style: str = Field(default="", description="语言风格")
    max_chars: int = Field(default=200, description="字数限制")
    structure_template: str = Field(default="", description="结构模板")
    action_call: str = Field(default="", description="CTA模板")
    mandatory_elements: list[str] = Field(default_factory=list, description="必含元素列表")


class PersonaChannelStyleCard(BaseModel):
    """客群×渠道交叉风格卡"""
    persona_name: str = Field(default="", description="客群名称")
    channel_name: str = Field(default="", description="渠道名称")
    familiar_register: str = Field(default="", description="该客群在该渠道中熟悉的语言层级")
    pressure_tolerance: str = Field(default="medium", description="对催促/稀缺话术的耐受度: high/medium/low/zero")
    service_tone: str = Field(default="", description="服务感基调")
    max_hooks: int = Field(default=2, description="该组合允许的最大营销钩子数")
    forbidden_phrases: list[str] = Field(default_factory=list, description="禁用表达")
    preferred_phrases: list[str] = Field(default_factory=list, description="偏好表达")
    cta_style: str = Field(default="", description="CTA风格")
    example_good: list[str] = Field(default_factory=list, description="正样本")
    example_bad: list[str] = Field(default_factory=list, description="负样本")
    privacy_creepiness_notes: str = Field(default="", description="隐私冒犯风险提示")
    delivery_recommendation: str = Field(default="send", description="投放建议: send/soft_send/service_first/app_passive/no_send")


class CopyStrategy(BaseModel):
    """文案策略（来自 Copy Strategist）"""
    universal_selling_points: list[str] = Field(default_factory=list, description="普世卖点")
    persona_strategies: list[PersonaStrategy] = Field(default_factory=list, description="客群策略列表")
    channel_specs: list[ChannelSpec] = Field(default_factory=list, description="渠道规范列表")
    copy_structure: str = Field(default="", description="统一文案结构模板")
    hooks_library: dict[str, list[str]] = Field(default_factory=dict, description="钩子库，按类型分组")
    persona_channel_cards: list[PersonaChannelStyleCard] = Field(default_factory=list, description="客群×渠道交叉风格卡")
    channel_orchestration_notes: str = Field(default="", description="渠道协同策略说明")
    ab_plans: list[dict[str, Any]] = Field(default_factory=list, description="AB 实验方案列表，每条含 persona_name/channel_name/variable_tested/group_a/group_b")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 3-4 文案模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MasterCopy(BaseModel):
    """母版文案（来自 Copy Writer）"""
    variant_type: str = Field(default="universal", description="版本类型: universal 或客群名称")
    headline: str = Field(default="", description="标题")
    sub_headline: str = Field(default="", description="副标题（如有）")
    body: str = Field(default="", description="正文")
    cta: str = Field(default="", description="行动号召")
    compliance_footer: str = Field(default="", description="合规尾注")
    hooks_used: list[str] = Field(default_factory=list, description="使用的钩子类型")
    benefit_references: list[str] = Field(default_factory=list, description="权益引用追踪")
    ab_group: str = Field(default="", description="AB 分组标记: A/B（空=非 AB）")
    ab_variable: str = Field(default="", description="AB 实验变量名（如 hook_type）")
    ab_label: str = Field(default="", description="AB 分组标签（如 损失规避）")


class ChannelCopy(BaseModel):
    """渠道版文案（来自 Channel Adapter 或 Copy Rewriter）"""
    variant_type: str = Field(default="universal", description="版本类型")
    channel_name: str = Field(default="", description="渠道名称")
    copy_text: str = Field(default="", description="完整的渠道版文案文本")
    char_count: int = Field(default=0, description="文案字数")
    compression_notes: str = Field(default="", description="压缩说明")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  审计与举证模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AtomicFactClaim(BaseModel):
    """文案中的业务原子事实声明"""
    claim_id: str = Field(default="", description="声明ID，如F-001")
    claim_text: str = Field(default="", description="文案中的原文表述")
    fact_source_document: str = Field(default="", description="业务知识库文档名")
    fact_source_chunks: list[str] = Field(default_factory=list, description="引用段落编号")
    fact_source_excerpt: str = Field(default="", description="知识库原文段落（逐字引用）")
    is_faithful: bool = Field(default=True, description="是否忠实于知识库")
    deviation_note: str = Field(default="", description="偏差说明")


class ComplianceEvidence(BaseModel):
    """合规举证条目"""
    evidence_id: str = Field(default="", description="举证ID，如C-001")
    claim_id: str = Field(default="", description="关联的事实声明ID")
    claim_text: str = Field(default="", description="文案中的原文表述")
    compliance_rule_id: str = Field(default="", description="适用的合规规则ID")
    compliance_rule_description: str = Field(default="", description="合规规则描述")
    rule_source_document: str = Field(default="", description="消保知识库文档名")
    rule_source_chunks: list[str] = Field(default_factory=list, description="引用段落编号")
    rule_source_excerpt: str = Field(default="", description="消保知识库原文段落（逐字引用）")
    compliance_reasoning: str = Field(default="", description="合规推理说明")
    verdict: str = Field(default="合规", description="合规/需关注/违规")


class OverComplianceCheck(BaseModel):
    """过度合规检测结果"""
    over_compliance_score: float = Field(default=7.0, description="过度合规评分 0-10，越高越好（越不过度合规）")
    disclaimer_ratio: float = Field(default=0.0, description="合规尾注占比")
    scene_density: float = Field(default=0.0, description="场景/生活语言占比")
    value_expression_density: float = Field(default=0.0, description="客户可感知收益表达占比")
    has_emotion: bool = Field(default=False, description="是否包含情感/共情表达")
    has_cta_variety: bool = Field(default=False, description="CTA是否非模板化")
    issues: list[str] = Field(default_factory=list, description="检测到的问题列表")
    rewrite_direction: str = Field(default="", description="创意恢复方向")


class CreativityPreservationCard(BaseModel):
    """创意保全卡"""
    expression: str = Field(default="", description="原表达")
    creative_intent: str = Field(default="", description="营销意图")
    compliance_risk: str = Field(default="", description="合规风险描述")
    mitigation: str = Field(default="", description="风险化解方式")
    l4_justification: str = Field(default="", description="L4自由创作区理由")
    conservative_alternative: str = Field(default="", description="稳妥备选表达")
    do_not_delete_reason: str = Field(default="", description="不建议删除的理由")


class AuditReport(BaseModel):
    """审计报告（来自 Copy Auditor）"""
    variant_type: str = Field(default="universal", description="版本类型")
    channel_name: str = Field(default="", description="渠道名称")
    overall_verdict: str = Field(default="通过", description="通过 / 需改写 / 需创意恢复 / 需人工介入")
    atomic_fact_claims: list[AtomicFactClaim] = Field(default_factory=list, description="事实声明列表")
    compliance_evidences: list[ComplianceEvidence] = Field(default_factory=list, description="合规举证列表")
    l2_disclosure_check: dict[str, Any] = Field(default_factory=dict, description="L2强制披露完整性检查")
    consistency_check: dict[str, Any] = Field(default_factory=dict, description="母版一致性校验")
    over_compliance_check: OverComplianceCheck = Field(default_factory=OverComplianceCheck, description="过度合规检测")
    creativity_cards: list[CreativityPreservationCard] = Field(default_factory=list, description="创意保全卡列表")
    issues: list[str] = Field(default_factory=list, description="审计问题列表")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  品味评分模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TasteIssue(BaseModel):
    """品味问题条目"""
    dimension: str = Field(default="", description="问题维度: oily/ad/premium/friend/familiarity")
    text: str = Field(default="", description="问题文案原文")
    suggestion: str = Field(default="", description="改写建议")


class TasteScore(BaseModel):
    """品味评分（来自 Taste Guardian）"""
    oily_score: float = Field(default=7.0, description="油腻度评分 0-10，越高越好")
    ad_score: float = Field(default=7.0, description="广告感评分 0-10，越高越好")
    premium_score: float = Field(default=7.0, description="高级感评分 0-10")
    friend_score: float = Field(default=7.0, description="朋友感评分 0-10")
    familiarity_score: float = Field(default=7.0, description="熟悉感评分 0-10")
    weighted_total: float = Field(default=7.0, description="加权总分")
    threshold: float = Field(default=7.0, description="通过门槛")
    overall_pass: bool = Field(default=True, description="是否通过品味评分")
    issues: list[TasteIssue] = Field(default_factory=list, description="品味问题列表")
    rewrite_direction: str = Field(default="", description="如未通过，整体改写方向提示")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  模拟受众模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SilenceReason(BaseModel):
    """沉默理由"""
    reason: str = Field(default="", description="沉默理由（第一人称）")
    probability: str = Field(default="中", description="可能性: 高/中/低")
    role_type: str = Field(default="", description="角色类型: 理性犹豫/广告免疫/隐私敏感")
    suggestion: str = Field(default="", description="微调建议")
    is_strategic: bool = Field(default=False, description="是否为策略层问题（无法通过文案微调解决）")


class AudienceReaction(BaseModel):
    """模拟受众反应（来自 Simulated Audience）"""
    variant_type: str = Field(default="universal", description="版本类型")
    channel_name: str = Field(default="", description="渠道名称")
    silence_reasons: list[SilenceReason] = Field(default_factory=list, description="沉默理由列表")
    click_willingness: float = Field(default=7.0, description="点击意愿 0-10")
    cta_suggestion: str = Field(default="", description="CTA优化建议（如有）")
    final_tweaks: list[str] = Field(default_factory=list, description="可采纳的微调指令列表")
    annoyance_risk: float = Field(default=2.0, description="反感风险 0-10")
    unsubscribe_risk: float = Field(default=1.0, description="退订/拉黑风险 0-10")
    complaint_risk: float = Field(default=0.5, description="投诉/质疑风险 0-10")
    privacy_creepiness: float = Field(default=1.0, description="隐私冒犯感 0-10")
    net_score: float = Field(default=5.0, description="综合净分")
    delivery_override: str = Field(default="", description="投放策略建议覆盖")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CG 流水线 State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CGState(MessagesState):
    """CG 合规营销文案生成流水线 State"""

    # ═══════════════════════════════════════════════════
    #  Pipeline 模式标识
    # ═══════════════════════════════════════════════════
    pipeline_mode: str = "cg"

    # ═══════════════════════════════════════════════════
    #  CG 输入参数
    # ═══════════════════════════════════════════════════
    cg_input: dict[str, Any] = field(default_factory=dict)

    # ═══════════════════════════════════════════════════
    #  Phase 1：双库检索产出
    # ═══════════════════════════════════════════════════
    product_facts: dict[str, Any] = field(default_factory=dict)
    # ProductFacts JSON，来自 Fact Miner
    compliance_rule_set: dict[str, Any] = field(default_factory=dict)
    # ComplianceRuleSet JSON，来自 Rule Miner
    fact_citations: Annotated[list[dict[str, Any]], operator.add] = field(default_factory=list)
    # 业务知识库引用，从 fact_miner 工具缓存提取
    rule_citations: Annotated[list[dict[str, Any]], operator.add] = field(default_factory=list)
    # 消保知识库引用，从 rule_miner 工具缓存提取

    # ═══════════════════════════════════════════════════
    #  Phase 2：文案策略
    # ═══════════════════════════════════════════════════
    copy_strategy: dict[str, Any] = field(default_factory=dict)
    # CopyStrategy JSON，来自 Copy Strategist

    # ═══════════════════════════════════════════════════
    #  Phase 3：母版文案（1+n 个版本）
    # ═══════════════════════════════════════════════════
    master_copies: list[dict[str, Any]] = field(default_factory=list)
    # MasterCopy JSON 列表，来自 Copy Writer

    # ═══════════════════════════════════════════════════
    #  Phase 4：渠道版文案及审核结果
    # ═══════════════════════════════════════════════════
    channel_copies: list[dict[str, Any]] = field(default_factory=list)
    # ChannelCopy JSON 列表，来自 Channel Adapter + Rewriter
    audit_reports: list[dict[str, Any]] = field(default_factory=list)
    # AuditReport JSON 列表，来自 Copy Auditor
    evidence_reports: list[str] = field(default_factory=list)
    # Markdown 格式举证报告，来自 Evidence Assembler
    taste_scores: list[dict[str, Any]] = field(default_factory=list)
    # TasteScore JSON 列表，来自 Taste Guardian
    audience_reactions: list[dict[str, Any]] = field(default_factory=list)
    # AudienceReaction JSON 列表，来自 Simulated Audience

    # ═══════════════════════════════════════════════════
    #  审改闭环控制
    # ═══════════════════════════════════════════════════
    rewrite_iterations: int = 0
    max_rewrite_iterations: int = 2

    # ═══════════════════════════════════════════════════
    #  AB 实验与反馈闭环
    # ═══════════════════════════════════════════════════
    historical_ab_summary: str = ""
    # 历史 AB 实验结论（自然语言，注入 Copy Strategist）
    ab_plans: list[dict[str, Any]] = field(default_factory=list)
    # Copy Strategist 输出的本批次 AB 方案
    # [{"persona_name": "...", "channel_name": "...", "variable_tested": "hook_type",
    #   "group_a": "获得感", "group_b": "损失规避"}]
    loaded_skills: list[str] = field(default_factory=list)
    # 已加载的经验 Skill 文本（来自历史投放验证的稳定经验）

    # ═══════════════════════════════════════════════════
    #  与 BI/CP 共用的文档共享字段
    # ═══════════════════════════════════════════════════
    document_chunk_maps: Annotated[dict[str, dict[str, str]], _merge_dict_maps] = field(default_factory=dict)
    # {document_title: {chunk_index: chunk_content}}，跨知识库共享，并行节点结果自动合并
    document_metadata: Annotated[dict[str, dict], _merge_dict_maps] = field(default_factory=dict)
    # {document_title: {"url", "file_id", "description", "is_extracted", "source_tools"}}
    citations: Annotated[list[dict[str, Any]], operator.add] = field(default_factory=list)
    # 全局引用累积（fact + rule 合并后）
