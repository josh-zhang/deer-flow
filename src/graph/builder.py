from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.graph.types import State
from src.graph.nodes import (
    analyst_node,
    arbitrator_node,
    background_investigation_node,
    coordinator_node,
    curator_node,
    human_feedback_node,
    plan_validator_node,
    planner_node,
    reporter_node,
    researcher_node,
    rule_splitter_node,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
#  Routing Functions
# ═══════════════════════════════════════════════════════

def route_from_validator(
    state: State,
) -> Literal["planner", "researcher", "arbitrator", "analyst"]:
    """
    plan_validator 出口路由:

    1. _plan_validator_needs_rerun        → planner
    2. 无有效 plan                        → planner
    3. 有未完成 research step             → researcher
    4. 有未完成 analysis step:
         BI  → arbitrator (走 arbitrator → analyst 路径)
         CP  → analyst    (跳过 arbitrator)
    5. fallback: 同 4
    """
    if state.get("_plan_validator_needs_rerun", False):
        return "planner"

    plan = state.get("current_plan")
    if plan is None or not getattr(plan, "steps", None):
        return "planner"

    pipeline_mode = state.get("pipeline_mode", "bi")
    fallback = "arbitrator" if pipeline_mode == "bi" else "analyst"

    for step in plan.steps:
        if not step.execution_res:
            if getattr(step, "step_type", "") == "research":
                return "researcher"
            elif getattr(step, "step_type", "") == "analysis":
                return fallback

    return fallback


def route_from_curator(
    state: State,
) -> Literal["rule_splitter", "researcher", "analyst"]:
    """
    curator 出口路由:

    BI mode → rule_splitter（保持原有链路）
    CP mode → 跳过 rule_splitter，直接判断下一步:
        - 有未完成 research step → researcher
        - 否则                   → analyst
    """
    pipeline_mode = state.get("pipeline_mode", "bi")

    if pipeline_mode == "bi":
        return "rule_splitter"

    # ── CP mode: skip rule_splitter ──
    plan = state.get("current_plan")
    if plan is None or not getattr(plan, "steps", None):
        return "analyst"

    for step in plan.steps:
        if not step.execution_res and getattr(step, "step_type", "") == "research":
            return "researcher"

    return "analyst"


def route_from_splitter(
    state: State,
) -> Literal["researcher", "arbitrator"]:
    """
    rule_splitter 后路由（仅 BI mode 会到达此节点）:
    - 有未完成 research step → researcher
    - 否则                   → arbitrator
    """
    plan = state.get("current_plan")
    if plan is None or not getattr(plan, "steps", None):
        return "arbitrator"

    for step in plan.steps:
        if not step.execution_res:
            if step.step_type == "research":
                return "researcher"
            return "arbitrator"

    return "arbitrator"


def route_from_analyst(state: State) -> Literal["reporter", "planner"]:
    """
    analyst 后路由（BI / CP 通用）:
    - replanning_needed=True 且未超限 → planner
    - 否则                            → reporter
    """
    if state.get("replanning_needed", False):
        replan_it = state.get("replan_iterations", 0)
        max_replan = state.get("max_replan_iterations", 2)
        if replan_it < max_replan:
            logger.info(
                f"Analyst triggered replan ({replan_it}/{max_replan}). "
                f"Reason: {state.get('replanning_reason', 'N/A')}"
            )
            return "planner"
        else:
            logger.warning(
                "Analyst requested replan but max iterations reached. Forcing report."
            )

    return "reporter"


# ═══════════════════════════════════════════════════════
#  Graph Construction
# ═══════════════════════════════════════════════════════

def _build_graph() -> StateGraph:
    builder = StateGraph(State)

    # ── Register Nodes（全部注册，按 mode 选择性经过）──
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("planner", planner_node)
    builder.add_node("plan_validator", plan_validator_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("curator", curator_node)
    builder.add_node("rule_splitter", rule_splitter_node)   # BI only
    builder.add_node("arbitrator", arbitrator_node)         # BI only
    builder.add_node("analyst", analyst_node)
    builder.add_node("reporter", reporter_node)

    # ── Fixed Edges ──
    builder.add_edge(START, "coordinator")
    builder.add_edge("background_investigator", "planner")
    builder.add_edge("human_feedback", "plan_validator")
    builder.add_edge("researcher", "curator")
    # ❌ 移除原来的: builder.add_edge("curator", "rule_splitter")
    builder.add_edge("arbitrator", "analyst")              # BI 路径专用
    builder.add_edge("reporter", END)

    # ── Conditional Edges ──
    builder.add_conditional_edges(
        "plan_validator",
        route_from_validator,
        ["planner", "researcher", "arbitrator", "analyst"],  # 新增 "analyst"
    )
    builder.add_conditional_edges(
        "curator",
        route_from_curator,                                  # ★ 新增：替代固定边
        ["rule_splitter", "researcher", "analyst"],
    )
    builder.add_conditional_edges(
        "rule_splitter",
        route_from_splitter,
        ["researcher", "arbitrator"],
    )
    builder.add_conditional_edges(
        "analyst",
        route_from_analyst,
        ["reporter", "planner"],
    )

    return builder


def build_graph():
    """编译统一 pipeline（无 checkpointer）"""
    return _build_graph().compile()


def build_graph_with_memory():
    """带内存 checkpointer 的统一 pipeline"""
    return _build_graph().compile(checkpointer=MemorySaver())