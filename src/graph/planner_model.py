from __future__ import annotations

from typing import List
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

class StepType(str, Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    # PROCESSING = "processing"


class Step(BaseModel):
    need_search: bool = Field(..., description="Must be explicitly set for each step")
    title: str
    background: str = Field(..., description="Context and why this step is needed")
    description: str = Field(..., description="Specify exactly what data to collect")
    step_type: StepType = Field(..., description="Indicates the nature of the step")
    execution_res: Optional[str] = Field(
        default=None, description="The Step execution result"
    )


class Plan(BaseModel):
    locale: str = "zh_CN"
    has_enough_context: bool = True
    thought: str = Field(default="", description="Thinking process for the plan")
    title: str = ""
    workflow_type: str = Field(
        default="A",
        description="Investigation workflow A/B/C/D (coordinator or planner override)",
    )
    missing_conditions: List[str] = Field(
        default_factory=list,
        description="User-stated missing dimensions for workflow D; empty for other types",
    )
    steps: List[Step] = Field(
        default_factory=list,
        description="Research step to get more context",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "has_enough_context": False,
                    "thought": (
                        "To understand the current market trends in AI, we need to gather comprehensive information."
                    ),
                    "title": "AI Market Research Plan",
                    "steps": [
                        {
                            "need_search": True,
                            "title": "Current AI Market Analysis",
                            "description": (
                                "Collect data on market size, growth rates, major players, and investment trends in AI sector."
                            ),
                            "step_type": "research",
                        }
                    ],
                }
            ]
        }



# ══════════════════════════════════════════════════════════════════════
# 枚举类型
# ══════════════════════════════════════════════════════════════════════

class ReviewMode(str, Enum):
    """审查模式"""
    STRICT = "strict"
    BALANCED = "balanced"
    PRECISE = "precise"


class RiskStatus(str, Enum):
    """合规状态"""
    RISK = "风险"
    COMPLIANT = "合规"
    NEEDS_REVIEW = "需人工复核"


class RiskLevel(str, Enum):
    """风险等级"""
    HIGH = "高"
    MEDIUM = "中"
    NONE = "无"


class FindingConfidence(str, Enum):
    """发现置信度"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ══════════════════════════════════════════════════════════════════════
# CP Review Planner 模型
# ══════════════════════════════════════════════════════════════════════

class ReviewPoint(BaseModel):
    """被触发的审查点"""
    id: int = Field(..., ge=1, le=27, description="审查点编号 1-27")
    checklist_item_title: str = Field(..., description="审查点小标题")
    is_baseline: bool = Field(default=False, description="是否为基线审查点")
    triggered_reason: str = Field(..., description="触发原因")
    triggered_excerpt: str = Field(..., description="触发的材料原文或缺失描述")

    @field_validator("is_baseline", mode="before")
    @classmethod
    def coerce_baseline(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)


class BaselineSkip(BaseModel):
    """被跳过的基线审查点"""
    id: int = Field(..., ge=1, le=27)
    checklist_item_title: str
    skip_reason: str = Field(..., min_length=1, description="跳过理由（必须充分）")


class PlanStep(BaseModel):
    """调查计划中的步骤"""
    need_search: bool
    title: str = Field(..., min_length=1)
    background: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    step_type: StepType
    review_point_id: int = Field(..., ge=0, le=27, description="对应审查点（analysis=0）")

    @field_validator("step_type", mode="before")
    @classmethod
    def coerce_step_type(cls, v):
        if isinstance(v, str):
            return StepType(v.lower())
        return v

    @model_validator(mode="after")
    def check_consistency(self):
        """验证 need_search 与 step_type 一致性"""
        if self.step_type == StepType.RESEARCH and not self.need_search:
            self.need_search = True
        if self.step_type == StepType.ANALYSIS and self.need_search:
            self.need_search = False
        return self


class CPPlannerInput(BaseModel):
    """Planner 的输入"""
    material_text: str = Field(..., min_length=1)
    max_step_num: int = Field(default=12, ge=3, le=20)


class CPPlannerOutput(BaseModel):
    """Planner 的输出"""
    thought: str = Field(default="")
    title: str = Field(default="")
    review_points: list[ReviewPoint] = Field(default_factory=list)
    baseline_skips: list[BaselineSkip] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_plan_structure(self):
        """验证计划结构完整性"""
        if not self.steps:
            return self
        # 最后一步应为 analysis
        if self.steps[-1].step_type != StepType.ANALYSIS:
            pass  # 记录警告但不阻断
        # research 步骤数应与 review_points 数一致
        research_count = sum(1 for s in self.steps if s.step_type == StepType.RESEARCH)
        if research_count != len(self.review_points):
            pass  # 允许不一致，仅记录
        return self

    @property
    def research_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.step_type == StepType.RESEARCH]

    @property
    def analysis_step(self) -> Optional[PlanStep]:
        analysis_steps = [s for s in self.steps if s.step_type == StepType.ANALYSIS]
        return analysis_steps[0] if analysis_steps else None


# ══════════════════════════════════════════════════════════════════════
# CP Searcher 模型
# ══════════════════════════════════════════════════════════════════════

class ToolReturn(BaseModel):
    """单次工具调用的返回结果"""
    tool_name: str = Field(..., description="工具名称")
    call_index: int = Field(..., ge=1, description="调用序号")
    query: Optional[str] = Field(default=None, description="local_search_tool 的 query")
    url: Optional[str] = Field(default=None, description="crawl_tool 的 url")
    content: str = Field(default="", description="工具返回原始内容")


class CPSearcherInput(BaseModel):
    """Searcher 的输入"""
    search_title: str
    search_background: str
    search_description: str


class CPSearcherOutput(BaseModel):
    """Searcher 的输出"""
    annotation_markdown: str = Field(default="", description="检索摘要 Markdown")
    raw_tool_returns: list[ToolReturn] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# CP Evidence Evaluator 模型
# ══════════════════════════════════════════════════════════════════════

class CPEvaluatorInput(BaseModel):
    """Evaluator 的输入"""
    search_title: str
    search_background: str
    search_description: str
    searcher_annotation: str
    raw_tool_returns: list[ToolReturn] = Field(default_factory=list)


class CPEvaluatorOutput(BaseModel):
    """Evaluator 的输出"""
    evaluation_markdown: str


# ══════════════════════════════════════════════════════════════════════
# CP Point Analyst 模型
# ══════════════════════════════════════════════════════════════════════

class CPPointAnalystInput(BaseModel):
    """Point Analyst 的输入"""
    material_text: str
    review_point_id: int = Field(..., ge=1, le=27)
    review_topic: str
    is_baseline: bool = Field(default=False)
    triggered_reason: str
    triggered_excerpt: str
    evaluator_output: str


class CPPointAnalystOutput(BaseModel):
    """Point Analyst 的输出"""
    review_point_id: int = Field(..., ge=1, le=27)
    review_topic: str
    status: RiskStatus
    risk_level: RiskLevel
    finding_confidence: FindingConfidence
    material_excerpt: str
    issue_analysis: str
    citation_external: str = Field(default="")
    citation_internal: str = Field(default="")
    confidence_reasoning: str = Field(default="")

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, v):
        mapping = {
            "风险": RiskStatus.RISK,
            "合规": RiskStatus.COMPLIANT,
            "需人工复核": RiskStatus.NEEDS_REVIEW,
            "risk": RiskStatus.RISK,
            "compliant": RiskStatus.COMPLIANT,
            "needs_review": RiskStatus.NEEDS_REVIEW,
        }
        if isinstance(v, str) and v in mapping:
            return mapping[v]
        return v

    @field_validator("risk_level", mode="before")
    @classmethod
    def coerce_risk_level(cls, v):
        mapping = {
            "高": RiskLevel.HIGH,
            "中": RiskLevel.MEDIUM,
            "无": RiskLevel.NONE,
            "high": RiskLevel.HIGH,
            "medium": RiskLevel.MEDIUM,
            "none": RiskLevel.NONE,
            "低": RiskLevel.NONE,
        }
        if isinstance(v, str) and v.lower() in mapping:
            return mapping[v.lower() if v.lower() in mapping else v]
        if isinstance(v, str) and v in mapping:
            return mapping[v]
        return v

    @field_validator("finding_confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v):
        mapping = {
            "high": FindingConfidence.HIGH,
            "medium": FindingConfidence.MEDIUM,
            "low": FindingConfidence.LOW,
            "高": FindingConfidence.HIGH,
            "中": FindingConfidence.MEDIUM,
            "低": FindingConfidence.LOW,
        }
        if isinstance(v, str):
            return mapping.get(v.lower(), FindingConfidence.LOW)
        return v

    @model_validator(mode="after")
    def enforce_confidence_rules(self):
        """强制执行置信度特殊规则"""
        if self.status == RiskStatus.COMPLIANT:
            self.finding_confidence = FindingConfidence.HIGH
        elif self.status == RiskStatus.NEEDS_REVIEW:
            self.finding_confidence = FindingConfidence.LOW
        return self


# ══════════════════════════════════════════════════════════════════════
# CP Final Reporter 模型
# ══════════════════════════════════════════════════════════════════════

class Citation(BaseModel):
    """参考来源"""
    index: int = Field(..., ge=1)
    document_name: str
    url: str

    class Config:
        frozen = True  # 支持去重时的 hash

    def __hash__(self):
        return hash(self.url)

    def __eq__(self, other):
        if isinstance(other, Citation):
            return self.url == other.url
        return False


class CPReporterInput(BaseModel):
    """Final Reporter 的输入"""
    material_text: str
    review_mode: ReviewMode
    point_assessments: list[CPPointAnalystOutput]
    available_citations: list[Citation] = Field(default_factory=list)


class CPReporterOutput(BaseModel):
    """Final Reporter 的输出"""
    report_markdown: str


# ══════════════════════════════════════════════════════════════════════
# 流水线最终输出
# ══════════════════════════════════════════════════════════════════════

class CPReviewResult(BaseModel):
    """消保合规审查流水线的最终输出"""
    # 元信息
    material_text: str
    review_mode: ReviewMode
    review_point_count: int = Field(..., ge=0)

    # Planner 输出
    plan: CPPlannerOutput

    # 逐审查点评估
    point_assessments: list[CPPointAnalystOutput]

    # 最终报告
    report: str

    # 溯源信息
    citations: list[Citation] = Field(default_factory=list)

    @property
    def risk_items(self) -> list[CPPointAnalystOutput]:
        return [pa for pa in self.point_assessments if pa.status == RiskStatus.RISK]

    @property
    def compliant_items(self) -> list[CPPointAnalystOutput]:
        return [pa for pa in self.point_assessments if pa.status == RiskStatus.COMPLIANT]

    @property
    def high_confidence_risks(self) -> list[CPPointAnalystOutput]:
        return [
            pa for pa in self.point_assessments
            if pa.status == RiskStatus.RISK and pa.finding_confidence == FindingConfidence.HIGH
        ]
