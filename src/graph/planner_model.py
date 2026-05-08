from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════
#  Step & Plan（Planner 的结构化输出）
# ═══════════════════════════════════════════════════════

class StepType(str, Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"


class Step(BaseModel):
    """单个执行步骤 — BI / CP 共用"""

    need_search: bool = Field(..., description="该步骤是否需要检索")
    title: str = Field(..., description="步骤标题")
    background: str = Field("", description="上下文背景，说明为何需要此步骤")
    description: str = Field(..., description="具体要收集/分析的内容")
    step_type: StepType = Field(..., description="步骤性质: research / analysis")
    execution_res: Optional[str] = Field(
        default=None, description="步骤执行结果（由下游节点填充）"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "need_search": True,
                    "title": "费率与收费条款检索",
                    "background": "消保要求费率披露完整、醒目",
                    "description": "检索文档中所有涉及年费、利率、手续费、罚息的条款原文",
                    "step_type": "research",
                }
            ]
        }
    }


class Plan(BaseModel):
    """执行计划 — BI / CP 共用（Planner 的结构化输出）"""

    locale: str = Field(default="zh_CN")
    has_enough_context: bool = Field(default=True)
    thought: str = Field(default="", description="规划思考过程")
    title: str = Field(default="")
    workflow_type: str = Field(
        default="A",
        description="工作流类型 A/B/C/D",
    )
    missing_conditions: List[str] = Field(
        default_factory=list,
        description="缺失维度（仅 workflow D）",
    )
    steps: List[Step] = Field(default_factory=list, description="执行步骤列表")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "has_enough_context": False,
                    "thought": "需要逐条检索文档中的费率披露、权益说明等条款",
                    "title": "信用卡营销材料消保合规审查计划",
                    "workflow_type": "A",
                    "steps": [
                        {
                            "need_search": True,
                            "title": "费率与收费条款检索",
                            "background": "消保要求费率披露完整、醒目",
                            "description": "检索文档中所有涉及年费、利率、手续费、罚息的条款原文",
                            "step_type": "research",
                        }
                    ],
                }
            ]
        }
    }


# ═══════════════════════════════════════════════════════
#  Analyst 结构化输出
# ═══════════════════════════════════════════════════════

class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskFinding(BaseModel):
    """一条风险发现"""

    finding_id: str = Field(..., description="发现编号，如 F-001")
    rule_ids: List[str] = Field(default_factory=list, description="关联的规则编号")
    risk_level: RiskLevel = Field(..., description="风险等级")
    description: str = Field(..., description="风险描述")
    evidence_summary: str = Field("", description="支撑证据摘要")
    recommendation: str = Field("", description="整改建议")


class AnalystOutput(BaseModel):
    """Analyst 节点的结构化输出（JSON）"""

    replanning_needed: bool = Field(False, description="是否需要补充调研")
    replanning_reason: str = Field("", description="补充调研理由")
    risk_findings: List[RiskFinding] = Field(
        default_factory=list, description="风险发现列表"
    )
    observations: List[str] = Field(
        default_factory=list, description="综合观察/结论"
    )
    confidence_score: float = Field(
        0.0, ge=0.0, le=1.0, description="整体判断置信度"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "replanning_needed": False,
                    "replanning_reason": "",
                    "risk_findings": [
                        {
                            "finding_id": "F-001",
                            "rule_ids": ["R-001", "R-003"],
                            "risk_level": "high",
                            "description": "年费标准未以醒目方式展示",
                            "evidence_summary": "合约第3条年费信息字号与正文一致，未加粗或使用对比色",
                            "recommendation": "建议将年费信息以加粗/加大字号/对比色方式展示",
                        }
                    ],
                    "observations": ["费率披露基本完整但醒目性不足"],
                    "confidence_score": 0.82,
                }
            ]
        }
    }


# ═══════════════════════════════════════════════════════
#  Resource（检索资源引用）
# ═══════════════════════════════════════════════════════

class Resource(BaseModel):
    """检索到的资源/文档片段"""

    source: str = Field(..., description="来源标识")
    content: str = Field(..., description="原文内容")
    metadata: dict = Field(default_factory=dict)