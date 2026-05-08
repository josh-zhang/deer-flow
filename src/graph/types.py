from __future__ import annotations

import operator
from dataclasses import field
from typing import Annotated, Any

from langgraph.graph import MessagesState

from src.models.base import Plan, Resource


class State(MessagesState):
    """BI / CP 通用 Investigation Pipeline State"""

    # ═══════════════════════════════════════════════════
    #  Pipeline 模式标识
    # ═══════════════════════════════════════════════════
    pipeline_mode: str = "bi"  # "bi" | "cp"

    # ═══════════════════════════════════════════════════
    #  Coordinator Output
    # ═══════════════════════════════════════════════════
    research_topic: str = ""
    original_topic: str = ""
    clarified_research_topic: str = ""
    workflow_type: str = "A"
    workflow_confidence: str = "high"

    # ═══════════════════════════════════════════════════
    #  Clarification
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
    #  Planner（结构化输出 — JSON）
    # ═══════════════════════════════════════════════════
    current_plan: Plan | None = None
    current_plan_last_round: Plan | None = None
    missing_conditions: list[str] = field(default_factory=list)
    plan_iterations: int = 0
    max_plan_iterations: int = 3
    replan_iterations: int = 0
    max_replan_iterations: int = 2  # CP 默认 2，BI 默认 3
    last_plan_text: str = ""

    # ═══════════════════════════════════════════════════
    #  Plan Validation
    # ═══════════════════════════════════════════════════
    planner_override_occurred: bool = False
    structure_validation_retried: bool = False
    _plan_validator_needs_rerun: bool = False
    skip_next_plan_iteration_increment: bool = False

    # ═══════════════════════════════════════════════════
    #  Searcher Output — Markdown 文本，跨步骤累积
    # ═══════════════════════════════════════════════════
    searcher_results: list[str] = field(default_factory=list)
    searcher_summaries: list[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════
    #  Curator Output — Markdown 文本
    # ═══════════════════════════════════════════════════
    curator_rule_splitter_views: list[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════
    #  Rule Splitter Output — Markdown 文本
    # ═══════════════════════════════════════════════════
    atomic_rules: list[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════
    #  Arbitrator Output — Markdown 文本
    # ═══════════════════════════════════════════════════
    arbitration_result: str = ""

    # ═══════════════════════════════════════════════════
    #  Analyst Output（结构化输出 — JSON）
    # ═══════════════════════════════════════════════════
    analyst_output: dict[str, Any] = field(default_factory=dict)
    replanning_needed: bool = False
    replanning_reason: str = ""
    observations: list[str] = field(default_factory=list)

    # ═══════════════════════════════════════════════════
    #  Reporter Output — Markdown 文本
    # ═══════════════════════════════════════════════════
    final_report: str = ""

    # ═══════════════════════════════════════════════════
    #  Citations — 全局累积
    # ═══════════════════════════════════════════════════
    citations: Annotated[list[dict[str, Any]], operator.add] = field(
        default_factory=list
    )

    # ═══════════════════════════════════════════════════
    #  UI / Flow Control
    # ═══════════════════════════════════════════════════
    auto_accepted_plan: bool = False
    goto: str = "planner"