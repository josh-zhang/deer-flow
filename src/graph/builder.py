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

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from src.graph.types import State
from src.graph.nodes import (
    coordinator_node,
    background_investigation_node,
    human_feedback_node,
    planner_node,
    plan_validator_node,
    researcher_node,
    curator_node,
    rule_splitter_node,
    arbitrator_node,
    analyst_node,
    reporter_node,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Graph Construction
# ─────────────────────────────────────────────

def _build_base_graph() -> StateGraph:
    builder = StateGraph(State)

    # ── Register Nodes ──
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("planner", planner_node)
    builder.add_node("plan_validator", plan_validator_node)  # NEW
    builder.add_node("researcher", researcher_node)
    builder.add_node("curator", curator_node)
    builder.add_node("rule_splitter", rule_splitter_node)
    builder.add_node("arbitrator", arbitrator_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("reporter", reporter_node)

    # ── Fixed Edges ──
    builder.add_edge(START, "coordinator")
    # coordinator 到 background_investigator 通过node.py代码跳跃
    builder.add_edge("background_investigator", "planner")
    # planner 到 human_feedback 需要node.py的代码进行用户交付确认
    builder.add_edge("human_feedback", "plan_validator")
    builder.add_edge("researcher", "curator")
    builder.add_edge("curator", "rule_splitter")
    builder.add_edge("arbitrator", "analyst")
    builder.add_edge("reporter", END)

    # ── Conditional Edges ──

    # plan_validator 三出口:
    #   planner    — Override 或结构校验失败，需重新规划
    #   researcher   — 校验通过，开始第一个 research step
    #   arbitrator — 校验通过但无 research step（理论上不应出现）
    builder.add_conditional_edges(
        "plan_validator",
        route_from_validator,
        ["planner", "researcher", "arbitrator"],
    )

    # rule_splitter 双出口:
    #   researcher   — 还有未完成的 research step
    #   arbitrator — 所有 research step 已完成
    builder.add_conditional_edges(
        "rule_splitter",
        route_from_splitter,
        ["researcher", "arbitrator"],
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


def build_graph():
    """编译调查流水线图（无 checkpointer，适合一次性脚本）。"""
    return _build_base_graph().compile()


def build_graph_with_memory():
    """带内存 checkpointer，供 API / 多轮会话使用 thread_id 恢复状态。"""
    return _build_base_graph().compile(checkpointer=MemorySaver())


# ─────────────────────────────────────────────
#  Routing Functions
# ─────────────────────────────────────────────

def route_from_validator(state: State) -> Literal["planner", "researcher", "arbitrator"]:
    """
    plan_validator 的出口路由。

    路由逻辑（按优先级）：
    1. Override 检测触发 → planner（用新 workflow_type 重新渲染）
    2. 结构校验失败   → planner（降级为 A 重新规划）
    3. 校验通过       → searcher（有 research step）或 arbitrator（无 research step）
    """
    plan = state.get("current_plan")

    # ── Layer 1+2: validator 节点要求重跑 planner ──
    if state.get("_plan_validator_needs_rerun", False):
        return "planner"

    if plan is None or not getattr(plan, "steps", None):
        return "planner"

    goto = "reporter"

    # only when step.execution_res got value, the step is considered as completed
    for step in plan.steps:
        if not step.execution_res:
            if getattr(step, "step_type", "") == "research":
                goto = "researcher"
                break
            elif getattr(step, "step_type", "") == "analysis":
                goto = "arbitrator"
                break
    
    return goto

def route_from_splitter(state: State) -> Literal["researcher", "arbitrator"]:
    """
    rule_splitter 完成后的路由。

    判断当前 plan 的 research step 是否全部完成：
    - current_step_index < len(research_steps) → 下一个 searcher
    - 否则 → arbitrator 汇总仲裁
    """
    plan = state.get("current_plan")
    if plan is None or not getattr(plan, "steps", None):
        return "arbitrator"

    in_completed_step = None
    for step in plan.steps:
        if not step.execution_res:
            in_completed_step = step
            break
    if not in_completed_step:
        return "arbitrator"
    
    if in_completed_step.step_type == "research":
        return "researcher"

    return "arbitrator"

def route_from_analyst(state: State) -> Literal["reporter", "planner"]:
    """
    analyst 完成后的路由。

    - replanning_needed=True 且未超迭代上限 → planner（replan）
    - 否则 → reporter（最终报告）
    """
    if state.get("replanning_needed", False):
        replan_it = state.get("replan_iterations", 0)
        max_replan = 3
        if replan_it < max_replan:
            logger.info(
                f"Analyst triggered replan (replan_iterations={replan_it}/{max_replan}). "
                f"Reason: {state.get('replanning_reason', 'N/A')}"
            )
            return "planner"
        else:
            logger.warning(
                "Analyst requested replan but max iterations reached. "
                "Forcing final report with insufficient confidence."
            )
    return "reporter"
