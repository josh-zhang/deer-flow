from __future__ import annotations
import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.graph.planner_model import StepType, Plan
from src.graph.nodes import (
    analyst_node,
    background_investigation_node,
    coordinator_node,
    curator_node,
    human_feedback_node,
    plan_validator_node,
    planner_node,
    reporter_node,
    researcher_node,
    rule_splitter_node,
    arbitrator_node,
)
from src.graph.types import State

logger = logging.getLogger(__name__)



def route_from_planner(state: State) -> Literal["planner", "researcher", "arbitrator", "analyst"]:
    """
    从 planner 路由: 检查计划是否有效
    """
    current_plan = state.get("current_plan")
    logger.debug(f"route from planner {current_plan}")

    if not current_plan:
        logger.warning("current_plan is None")
        return "planner"
    # 防御: planner 存的是原始文本时（validator 解析失败未回写），回 planner 重新生成
    if isinstance(current_plan, str):
        logger.warning("route_from_planner: current_plan is still a string, goto planner")
        return "planner"
    if getattr(current_plan, "steps") is None or not current_plan.steps:
        return "planner"

    # — layer 1: Override 需要覆盖
    if hasattr(current_plan, "workflow_type"):
        planner_type = getattr(current_plan, "workflow_type", state["workflow_type"])

        current_type = state["workflow_type"]

        # 如果声明的类型与 state 中不同, 且 validator 端设置了 override 标志
        if planner_type != current_type and not state.get("planner_override_occurred", False):
            # 注意: 此分支理论上不会命中, 因为 plan_validator_node 已在 override 时
            # 更新了 state["workflow_type"]_. 作为防御性保留.
            return "planner"

    # — Layer 2: plan_validator 检测到致命数据类型错误需要重建 — 通过设置 workflow_type="A" +
    # structure_validation_retried=True 来续写
    # 这里通过一个简单的标志判断
    if state.get("_plan_validator_needs_rerun", False):
        return "planner"

    pipeline_mode = state.get("pipeline_mode", "bi")

    fallback = "arbitrator" if pipeline_mode == "bi" else "analyst"

    for step in current_plan.steps:
        if not step.execution_res:
            if getattr(step, "step_type", "") == "research":
                return "researcher"
            elif getattr(step, "step_type", "") == "analysis":
                return fallback
            return fallback

    return fallback


def route_from_splitter(state: State):
    """
    从 Splitter 路由: 控制研究循环, 如果还有下一步, 回 researcher, 否则去 arbitrator
    """
    current_plan = state.get("current_plan")

    logger.debug(f"route from splitter {current_plan}")

    if not current_plan or not getattr(current_plan, "steps", None):
        logger.error("route from splitter: current_plan is missing steps")
        return "reporter"

    # Find first incomplete step execution_res
    incomplete_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            incomplete_step = step
            break

    if incomplete_step is not None and incomplete_step.step_type == StepType.RESEARCH:
        return "researcher"

    pipeline_mode = state.get("pipeline_mode", "bi")

    if pipeline_mode == "bi":
        return "arbitrator"
    elif pipeline_mode == "cp":
        return "analyst"
    else:
        return "analyst"


def route_from_analyst(state: State):
    """
    从 Analyst 路由: 检查是否需要重规划
    """
    # 建议在 State 中增加 replan: bool 字段, 由 Analyst 逻辑决定
    # 如果没有该字段, 也可以通过判断最后一条 observation 是否包含"无法分析"等关键词判断
    replanning_needed = state.get("replan", False)

    replan_iterations = state.get("replan_iterations", 0)

    MAX_ITERATIONS = 2  # 设定最大重规划值

    # 这里已有的重规划次数
    if replanning_needed and replan_iterations < MAX_ITERATIONS:
        logger.info(f"[Router] Analyst 发现证据不足, 触发重规划 (replan_iterations + 1) 次重规划.")
        return "planner"

    if replanning_needed and replan_iterations >= MAX_ITERATIONS:
        logger.info("[Router] Analyst 达到最大重规划次数, 生成报告(含缺失说明).")
        return "reporter"

    logger.info("[Router] Analyst 分析完成, 准备生成最终报告.")
    return "reporter"


def build_base_graph() -> StateGraph:
    """Build and return the base state graph with all nodes and edges."""
    # — Register Nodes (全部注册, 按 mode 选择性经过) —
    builder = StateGraph(State)
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("planner", planner_node)
    builder.add_node("plan_validator", plan_validator_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("curator", curator_node)
    builder.add_node("rule_splitter", rule_splitter_node)
    builder.add_node("arbitrator", arbitrator_node)  # BI only
    builder.add_node("analyst", analyst_node)
    builder.add_node("reporter", reporter_node)

    # — Fixed Edges —
    builder.add_edge(START, "coordinator")
    builder.add_edge("background_investigator", "planner")
    builder.add_edge("researcher", "curator")
    builder.add_edge("curator", "rule_splitter")
    builder.add_edge("arbitrator", "analyst")  # BI 路径专属
    builder.add_edge("reporter", END)

    # — Conditional Edges —
    builder.add_conditional_edges("plan_validator", route_from_planner, ["planner", "researcher", "arbitrator", "analyst"])  # route_from_planner
    builder.add_conditional_edges("rule_splitter", route_from_splitter, ["researcher", "arbitrator", "analyst"])
    builder.add_conditional_edges("analyst", route_from_analyst, ["reporter", "planner"])

    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()
    # build state graph
    builder = build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without checkpointer."""
    # build state graph
    builder = build_base_graph()
    return builder.compile()
