# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.prompts.planner_model import StepType

from .nodes import (
    analyst_node,
    background_investigation_node,
    coordinator_node,
    planner_node,
    reporter_node,
    researcher_node,
)
from .types import State


def continue_to_running_research_team(state: State):
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        return "planner"

    if all(step.execution_res for step in current_plan.steps):
        return "planner"

    # Find first incomplete step
    incomplete_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            incomplete_step = step
            break

    if not incomplete_step:
        return "planner"

    if incomplete_step.step_type == StepType.RESEARCH:
        return "researcher"
    if incomplete_step.step_type == StepType.ANALYSIS:
        return "analyst"
    return "planner"


# def _build_base_graph():
#     builder = StateGraph(State)
#     builder.add_edge(START, "coordinator")  # 包含需求澄清
#     builder.add_node("coordinator", coordinator_node)
#     builder.add_node("background_investigator", background_investigation_node)
#     builder.add_node("planner", planner_node)
#
#     # 细化执行循环
#     builder.add_node("researcher", researcher_node)
#     builder.add_node("rule_splitter", rule_splitter_node)
#     builder.add_node("arbitrator", arbitrator_node)
#     builder.add_node("analyst", analyst_node)
#     builder.add_node("reporter", reporter_node)
#
#     builder.add_edge("coordinator", "background_investigator")
#     builder.add_edge("background_investigator", "planner")
#
#     # 规划器决定下一步是检索还是分析
#     builder.add_conditional_edges("planner", route_from_planner, ["researcher", "arbitrator"])
#
#     # 检索后直接接拆分，拆分完判断是否需要继续检索还是分析
#     builder.add_edge("researcher", "rule_splitter")
#     builder.add_conditional_edges("rule_splitter", route_from_splitter, ["researcher", "arbitrator"])
#
#     # 所有检索和拆分完成后，进行仲裁，然后分析，最后出报告
#     builder.add_edge("arbitrator", "analyst")
#
#     # 增加动态Replanning机制，如果analyst发现缺失信息，进行重新规划
#     builder.add_conditional_edges("analyst", route_from_analyst, ["reporter", "planner"])
#
#     builder.add_edge("reporter", END)
#     return builder


def _build_base_graph():
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)

    # 拆分后的检索+评估节点
    builder.add_node("searcher", searcher_node)              # 原 researcher，聚焦检索
    builder.add_node("evidence_curator", evidence_curator_node)  # 新增，聚焦信息评估
    builder.add_node("rule_splitter", rule_splitter_node)
    builder.add_node("arbitrator", arbitrator_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("reporter", reporter_node)

    builder.add_edge("coordinator", "background_investigator")
    builder.add_edge("background_investigator", "planner")

    builder.add_conditional_edges("planner", route_from_planner, ["searcher", "arbitrator"])

    # 检索 → 评估 → 拆分，拆分完判断继续检索还是仲裁
    builder.add_edge("searcher", "evidence_curator")
    builder.add_edge("evidence_curator", "rule_splitter")
    builder.add_conditional_edges("rule_splitter", route_from_splitter, ["searcher", "arbitrator"])

    builder.add_edge("arbitrator", "analyst")
    builder.add_conditional_edges("analyst", route_from_analyst, ["reporter", "planner"])
    builder.add_edge("reporter", END)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()


graph = build_graph()
