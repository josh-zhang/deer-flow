"""
V2 Investigation Pipeline — Graph Definition & Routing

图结构变更说明（相比 V1）：
- 新增节点: plan_validator（Layer 1 Override + Layer 2 结构校验）
- 边变更: planner → plan_validator → [planner | searcher | arbitrator]
  （V1: planner 直连 searcher/arbitrator）
- 其余拓扑不变
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from .state import State
from .nodes import (
    coordinator_node,
    background_investigation_node,
    planner_node,
    plan_validator_node,
    searcher_node,
    evidence_curator_node,
    rule_splitter_node,
    arbitrator_node,
    analyst_node,
    reporter_node,
)
from .prompt_utils import get_research_steps

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Graph Construction
# ─────────────────────────────────────────────

def _build_base_graph() -> StateGraph:
    builder = StateGraph(State)

    # ── Register Nodes ──
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)
    builder.add_node("plan_validator", plan_validator_node)  # NEW
    builder.add_node("searcher", searcher_node)
    builder.add_node("evidence_curator", evidence_curator_node)
    builder.add_node("rule_splitter", rule_splitter_node)
    builder.add_node("arbitrator", arbitrator_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("reporter", reporter_node)

    # ── Fixed Edges ──
    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "background_investigator")
    builder.add_edge("background_investigator", "planner")
    builder.add_edge("planner", "plan_validator")  # CHANGED: was planner → conditional
    builder.add_edge("searcher", "evidence_curator")
    builder.add_edge("evidence_curator", "rule_splitter")
    builder.add_edge("arbitrator", "analyst")
    builder.add_edge("reporter", END)

    # ── Conditional Edges ──

    # plan_validator 三出口:
    #   planner    — Override 或结构校验失败，需重新规划
    #   searcher   — 校验通过，开始第一个 research step
    #   arbitrator — 校验通过但无 research step（理论上不应出现）
    builder.add_conditional_edges(
        "plan_validator",
        route_from_validator,
        ["planner", "searcher", "arbitrator"],
    )

    # rule_splitter 双出口:
    #   searcher   — 还有未完成的 research step
    #   arbitrator — 所有 research step 已完成
    builder.add_conditional_edges(
        "rule_splitter",
        route_from_splitter,
        ["searcher", "arbitrator"],
    )

    # analyst 双出口:
    #   reporter — 信息充分，输出最终结论
    #   planner  — 信息不足，触发 replan
    builder.add_conditional_edges(
        "analyst",
        route_from_analyst,
        ["reporter", "planner"],
    )

    return builder


# ─────────────────────────────────────────────
#  Routing Functions
# ─────────────────────────────────────────────

def route_from_validator(state: State) -> Literal["planner", "searcher", "arbitrator"]:
    """
    plan_validator 的出口路由。

    路由逻辑（按优先级）：
    1. Override 检测触发 → planner（用新 workflow_type 重新渲染）
    2. 结构校验失败   → planner（降级为 A 重新规划）
    3. 校验通过       → searcher（有 research step）或 arbitrator（无 research step）
    """
    plan = state.get("current_plan")

    # ── Layer 1: Override 需要重跑 ──
    if plan is not None and hasattr(plan, "workflow_type"):
        planner_type = getattr(plan, "workflow_type", state["workflow_type"])
        current_type = state["workflow_type"]
        # 如果 plan 声明的类型与 state 中的不同，且 validator 刚设置了 override 标志
        if planner_type != current_type and not state.get("planner_override_occurred", False):
            # 注意：此分支理论上不应命中，因为 plan_validator_node 已在 override 时
            # 更新了 state["workflow_type"]。作为防御性保留。
            return "planner"

    # ── Layer 2: 结构校验失败需要重跑 ──
    # plan_validator_node 通过设置 workflow_type="A" + structure_validation_retried=True 来信号
    # 这里通过一个简单的标志判断
    if state.get("_plan_validator_needs_rerun", False):
        return "planner"

    # ── 正常路由 ──
    research_steps = get_research_steps(plan)
    if research_steps:
        return "searcher"
    return "arbitrator"


def route_from_splitter(state: State) -> Literal["searcher", "arbitrator"]:
    """
    rule_splitter 完成后的路由。

    判断当前 plan 的 research step 是否全部完成：
    - current_step_index < len(research_steps) → 下一个 searcher
    - 否则 → arbitrator 汇总仲裁
    """
    plan = state.get("current_plan")
    research_steps = get_research_steps(plan)
    current_idx = state.get("current_step_index", 0)

    if current_idx < len(research_steps):
        return "searcher"
    return "arbitrator"


def route_from_analyst(state: State) -> Literal["reporter", "planner"]:
    """
    analyst 完成后的路由。

    - replanning_needed=True 且未超迭代上限 → planner（replan）
    - 否则 → reporter（最终报告）
    """
    if state.get("replanning_needed", False):
        plan_iterations = state.get("plan_iterations", 0)
        max_iterations = state.get("max_plan_iterations", 3)
        if plan_iterations < max_iterations:
            logger.info(
                f"Analyst triggered replan (iteration {plan_iterations}/{max_iterations}). "
                f"Reason: {state.get('replanning_reason', 'N/A')}"
            )
            return "planner"
        else:
            logger.warning(
                "Analyst requested replan but max iterations reached. "
                "Forcing final report with insufficient confidence."
            )
    return "reporter"