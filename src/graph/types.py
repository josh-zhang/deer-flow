from __future__ import annotations

import operator
from dataclasses import field
from typing import Annotated, Any

from langgraph.graph import MessagesState

from src.graph.planner_model import Plan
from src.rag import Resource


class State(MessagesState):
    """BI / CP 通用 Investigation Pipeline State"""

    # ═══════════════════════════════════════════════════
    #  Locale / Pipeline 模式标识
    # ═══════════════════════════════════════════════════
    locale: str = "zh-CN"

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
    background_investigation_results: str | None = None  # 本体映射 XML 或降级时的原始检索 payload
    kb_panorama: str | None = None  # P1 修正：KB 轻量检索概要（独立于本体映射，供 Planner 长尾参考）
    resources: list[Resource] = []

    # ═══════════════════════════════════════════════════
    #  Planner（结构化输出 — JSON）
    # ═══════════════════════════════════════════════════
    current_plan: Plan | None = None
    current_plan_cached: Plan | None = None
    missing_conditions: list[str] = field(default_factory=list)
    plan_iterations: int = 0
    max_plan_iterations: int = 3
    replan_iterations: int = 0
    last_plan: str = ""

    # ═══════════════════════════════════════════════════
    #  Plan Validation
    # ═══════════════════════════════════════════════════
    planner_override_occurred: bool = False
    structure_validation_retried: bool = False
    _plan_validator_needs_rerun: bool = False

    # ═══════════════════════════════════════════════════
    #  Searcher Output — Markdown 文本，跨步骤累积
    # ═══════════════════════════════════════════════════
    observations: list[str] = []
    search_results: list[tuple[str, str]] = []

    # ═══════════════════════════════════════════════════
    #  Curator Output — Markdown 文本
    # ═══════════════════════════════════════════════════
    curator_rule_splitter_views: list[str] = []

    # ═══════════════════════════════════════════════════
    #  Rule Splitter Output — Markdown 文本
    # ═══════════════════════════════════════════════════
    atomic_rules: list[str] = []

    # ═══════════════════════════════════════════════════
    #  Arbitrator Output — Markdown 文本
    # ═══════════════════════════════════════════════════
    arbitration_result: str = ""

    # ═══════════════════════════════════════════════════
    #  Analyst Output（结构化输出 — JSON）
    # ═══════════════════════════════════════════════════
    analyst_output: dict[str, Any] = field(default_factory=dict)
    replan: bool = False
    replan_reason: str = ""

    # ═══════════════════════════════════════════════════
    #  Reporter Output — Markdown 文本
    # ═══════════════════════════════════════════════════
    final_report: str = ""

    # ═══════════════════════════════════════════════════
    #  Citations — 全局累积
    # ═══════════════════════════════════════════════════
    citations: list[dict[str, Any]] = field(
        default_factory=list
    )

    # ═══════════════════════════════════════════════════
    #  UI / Flow Control
    # ═══════════════════════════════════════════════════
    auto_accepted_plan: bool = False
    goto: str = "planner"

    # ═══════════════════════════════════════════════════
    #  Document Chunk Maps — 全局累积，跨步骤合并
    #  用途：Curator 引用段落编号后，框架通过此映射提取原文
    #        传给 Rule Splitter / Analyst
    # ═══════════════════════════════════════════════════
    document_chunk_maps: dict[str, dict[str, str]] = field(default_factory=dict)
    # {document_title: {chunk_index: chunk_content}}

    # ═══════════════════════════════════════════════════
    #  Document Metadata — 全局累积
    #  用途：跟踪每个文档的 URL、编号、是否经过截取等
    # ═══════════════════════════════════════════════════
    document_metadata: dict[str, dict] = field(default_factory=dict)
    # {document_title: {"document_url", "file_id", "description", "is_extracted", "source_tools"}}

    # ═══════════════════════════════════════════════════
    #  User-attached Files — 跨轮累积
    #  用途：前端 /chat 输入框上传的文件（文本 / 图片），节点可直接读取
    #  Entry shape:
    #    {id, message_id, name, mime, kind ("text"|"image"),
    #     size_bytes, text (when kind=="text"), b64 (when kind=="image")}
    # ═══════════════════════════════════════════════════
    attached_files: list[dict[str, Any]] = field(
        default_factory=list
    )
