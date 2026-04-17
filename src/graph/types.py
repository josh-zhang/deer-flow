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

from src.prompts.planner_model import Plan
from src.rag import Resource


class State(MessagesState):
    """V2 Investigation Pipeline State"""

    # ═══════════════════════════════════════════════════
    #  Coordinator Output
    # ═══════════════════════════════════════════════════
    research_topic: str = ""
    clarified_research_topic: str = ""
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
    missing_conditions: list[str] = field(default_factory=list)  # workflow D
    plan_iterations: int = 0
    max_plan_iterations: int = 3

    # ═══════════════════════════════════════════════════
    #  Plan Validation — Layer 1 (Override) + Layer 2 (Structure)
    #
    #  这两个标志位在 Analyst 触发 replan 时被重置为 False,
    #  确保新的规划周期拥有完整的校验能力。
    # ═══════════════════════════════════════════════════
    planner_override_occurred: bool = False
    structure_validation_retried: bool = False

    # ═══════════════════════════════════════════════════
    #  Step Execution Tracking
    #
    #  current_step_index: 当前 plan 中已完成的 research step 数量。
    #  每完成一个 research step（rule_splitter 输出后），递增 1。
    #  新 plan 生成时（含 replan）重置为 0。
    # ═══════════════════════════════════════════════════
    current_step_index: int = 0

    # ═══════════════════════════════════════════════════
    #  Searcher Output — 跨步骤 & 跨迭代累积
    #
    #  searcher_results:   每个 research step 的原始工具返回 + 检索注释
    #  searcher_summaries: 精简摘要，供后续 Searcher 参考已有线索
    # ═══════════════════════════════════════════════════
    searcher_results: Annotated[list[str], operator.add] = field(
        default_factory=list
    )
    searcher_summaries: Annotated[list[str], operator.add] = field(
        default_factory=list
    )

    # ═══════════════════════════════════════════════════
    #  Evidence Curator Output — 跨步骤 & 跨迭代累积
    #
    #  curator_analysis_views:      分析视图 → Analyst + Reporter
    #  curator_rule_splitter_views: 完整视图 → Rule Splitter
    # ═══════════════════════════════════════════════════
    curator_analysis_views: Annotated[list[str], operator.add] = field(
        default_factory=list
    )
    curator_rule_splitter_views: Annotated[list[str], operator.add] = field(
        default_factory=list
    )

    # ═══════════════════════════════════════════════════
    #  Rule Splitter Output — 跨步骤 & 跨迭代累积
    # ═══════════════════════════════════════════════════
    atomic_rules: Annotated[list[str], operator.add] = field(
        default_factory=list
    )

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
    # ═══════════════════════════════════════════════════
    analyst_output: dict[str, Any] = field(default_factory=dict)
    replanning_needed: bool = False
    replanning_reason: str = ""

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