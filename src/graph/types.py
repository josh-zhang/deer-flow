"""
V2 Investigation Pipeline — State Definition

变更说明（相比 V1）：
- 新增: workflow_type, workflow_confidence    ← Coordinator 输出
- 新增: missing_conditions                   ← Planner 输出（工作流 D）
- 新增: plan_validator 相关标志位             ← Layer 1/2 防御机制
- 新增: curator_analysis_views               ← 替代原 curator_searcher_views
- 新增: searcher_summaries                   ← 跨步骤检索摘要
- 新增: analyst_output (dict)                ← Analyst 结构化 JSON 输出
- 新增: replanning_reason                    ← 替代原 analyst_feedback
- 移除: goto                                 ← 路由改由条件边函数显式处理
- 移除: observations                         ← 功能由 searcher_results 替代
- 重命名: curator_searcher_views → curator_analysis_views

累计型列表字段使用 Annotated[list[X], operator.add]，
节点只需返回增量 [new_item]，框架自动追加到现有列表。
"""

from __future__ import annotations

import operator
from dataclasses import field
from typing import Annotated, Any

from langgraph.graph import MessagesState

from src.graph.planner_model import Plan
from src.rag import Resource


class State(MessagesState):
    """V2 Investigation Pipeline State"""

    # ═══════════════════════════════════════════════════
    #  Coordinator Output
    # ═══════════════════════════════════════════════════
    research_topic: str = ""
    original_topic: str = "" # use to keep original query user provided
    clarified_research_topic: str = "" # use to keep clarified query for researcher
    workflow_type: str = "A"          # A / B / C / D
    workflow_confidence: str = "high"  # high / medium / low

    # ═══════════════════════════════════════════════════
    #  Clarification (disabled by default)
    # ═══════════════════════════════════════════════════
    enable_clarification: bool = False
    clarification_rounds: int = 0
    clarification_history: Annotated[list[str], operator.add] = field(
        default_factory=list
    )
    is_clarification_complete: bool = False
    max_clarification_rounds: int = 3

    # ═══════════════════════════════════════════════════
    #  Background Investigation
    # ═══════════════════════════════════════════════════
    enable_background_investigation: bool = True
    background_investigation_results: str | None = None
    resources: Annotated[list[Resource], operator.add] = field(
        default_factory=list
    )

    # ═══════════════════════════════════════════════════
    #  Planner
    # ═══════════════════════════════════════════════════
    current_plan: Plan | None = None
    current_plan_last_round: Plan | None = None # use to keep last round plan for replan
    missing_conditions: list[str] = field(default_factory=list)  # workflow D
    plan_iterations: int = 0
    max_plan_iterations: int = 3 # used for counting creating plan based on human feedback, not for replan
    replan_iterations: int = 0 # used for counting replan iterations
    last_plan_text: str = ""  # last planner raw JSON / text for replan context

    # ═══════════════════════════════════════════════════
    #  Plan Validation — Layer 1 (Override) + Layer 2 (Structure)
    #
    #  这两个标志位在 Analyst 触发 replan 时被重置为 False,
    #  确保新的规划周期拥有完整的校验能力。
    # ═══════════════════════════════════════════════════
    planner_override_occurred: bool = False
    structure_validation_retried: bool = False
    _plan_validator_needs_rerun: bool = False
    skip_next_plan_iteration_increment: bool = False

    # ═══════════════════════════════════════════════════
    #  Searcher Output — 跨步骤 & 跨迭代累积
    #
    #  searcher_results:   每个 research step 的原始工具返回，供后续 curator 提取相关信息
    #  searcher_summaries: 检索注释，供后续 Searcher 参考已有线索
    # ═══════════════════════════════════════════════════
    # 整表替换：planner / replan 时重置，节点返回完整列表
    searcher_results: list[str] = field(default_factory=list)
    searcher_summaries: list[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════
    #  Evidence Curator Output — 跨步骤 & 跨迭代累积
    #
    #  curator_rule_splitter_views: 完整视图 → Rule Splitter （注意execution_res存放分析视图 → Analyst）
    # ═══════════════════════════════════════════════════
    curator_rule_splitter_views: list[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════
    #  Rule Splitter Output — 跨步骤 & 跨迭代累积
    # ═══════════════════════════════════════════════════
    atomic_rules: list[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════
    #  Arbitrator Output
    #  每次 Arbitrator 运行时覆盖（处理全量 atomic_rules）
    # ═══════════════════════════════════════════════════
    arbitration_result: str = ""

    # ═══════════════════════════════════════════════════
    #  Analyst Output
    #
    #  analyst_output:     完整结构化 JSON（含 conclusions, analysis_text 等）
    #  replanning_needed:  由 Analyst 设置，驱动 route_from_analyst
    #  replanning_reason:  Analyst 说明缺失信息和建议方向
    #  observations:       Analyst 的观察结果，供后续 Reporter 参考
    # ═══════════════════════════════════════════════════
    analyst_output: dict[str, Any] = field(default_factory=dict)
    replanning_needed: bool = False
    replanning_reason: str = ""
    observations: list[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════
    #  Reporter Output — 最终交付物
    # ═══════════════════════════════════════════════════
    final_report: str = ""

    # ═══════════════════════════════════════════════════
    #  Citations — 全局累积
    # ═══════════════════════════════════════════════════
    citations: Annotated[list[dict[str, Any]], operator.add] = field(
        default_factory=list
    )

    # ═══════════════════════════════════════════════════
    #  UI / Compat
    # ═══════════════════════════════════════════════════
    auto_accepted_plan: bool = False

    # workflow control
    goto:str = "planner"