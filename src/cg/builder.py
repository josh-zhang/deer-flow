from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.cg.nodes import (
    adapt_audit_assemble_node,
    copy_strategist_node,
    copy_writer_node,
    fact_miner_node,
    rule_miner_node,
)
from src.cg.types import CGState

logger = logging.getLogger(__name__)


def _build_cg_graph() -> StateGraph:
    """构建 CG 流水线 StateGraph（未编译）。

    拓扑：
      START → fact_miner  ─┐
      START → rule_miner  ─┤→ copy_strategist → copy_writer → adapt_audit_assemble → END

    fact_miner 和 rule_miner 并行执行（LangGraph 支持从 START 出发的多条边并行）。
    两者都完成后，LangGraph 会自动汇合进入 copy_strategist。
    """
    builder = StateGraph(CGState)

    # ── 注册节点 ──
    builder.add_node("fact_miner", fact_miner_node)
    builder.add_node("rule_miner", rule_miner_node)
    builder.add_node("copy_strategist", copy_strategist_node)
    builder.add_node("copy_writer", copy_writer_node)
    builder.add_node("adapt_audit_assemble", adapt_audit_assemble_node)

    # ── Phase 1：并行双库检索 ──
    builder.add_edge(START, "fact_miner")
    builder.add_edge(START, "rule_miner")

    # ── Phase 1→2：等待双库完成后进入策略规划 ──
    builder.add_edge("fact_miner", "copy_strategist")
    builder.add_edge("rule_miner", "copy_strategist")

    # ── Phase 2→3→4 ──
    builder.add_edge("copy_strategist", "copy_writer")
    builder.add_edge("copy_writer", "adapt_audit_assemble")
    builder.add_edge("adapt_audit_assemble", END)

    return builder


def build_cg_graph():
    """编译 CG 流水线（无 checkpointer）。"""
    return _build_cg_graph().compile()


def build_cg_graph_with_memory():
    """带内存 checkpointer 的 CG 流水线。"""
    return _build_cg_graph().compile(checkpointer=MemorySaver())
