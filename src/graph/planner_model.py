from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


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


class Plan(BaseModel):
    """执行计划 — BI / CP 共用（Planner 的结构化输出）"""

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
