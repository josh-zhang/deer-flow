# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
V2 调查流水线节点实现。

与 `types.State`、`builder.py` 拓扑对齐：
coordinator →(Command)→ background → planner →(Command)→ human_feedback → plan_validator
→ researcher → curator → rule_splitter → … → arbitrator → analyst →(replan|)→ reporter
"""

from __future__ import annotations

import json
import logging
import re
from functools import partial
from typing import Annotated, Any, Literal
from collections import defaultdict
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END
from langgraph.types import Command

from src.agents import create_agent
from src.citations import extract_citations_from_messages, merge_citations
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.graph.planner_model import Plan
from src.llms.llm import get_llm_by_type, get_llm_token_limit_by_type
from src.prompts.template import apply_prompt_template, get_system_prompt_template
from src.tools import crawl_tool, get_retriever_tool
from src.utils.context_manager import ContextManager
from src.utils.json_utils import repair_json_output, sanitize_tool_response
from src.extraction.chunk_extractor import ExtractionResult

from .ontology import (
    extract_ontology_mapping,
    render_arbitrator_bucketing_guide,
    render_planner_guardrail,
    render_skeleton_for_mapper,
)
from .curator_views import format_citations_for_reporter
from .types import State
from .utils import (
    build_clarified_topic_from_history,
    format_attached_files_for_prompt,
    get_latest_user_message,
    get_message_content,
    reconstruct_clarification_history,
)

logger = logging.getLogger(__name__)

_RESEARCHER_TAG = "[researcher]"
_CURATOR_TAG = "[curator]"
_RULE_DONE = "[规则拆分已完成]"


# ── Coordinator 工具（与 coordinator.zh_CN.md 一致）────────────────────────


@tool
def handoff_to_planner(
    research_topic: Annotated[str, "银行信用卡客户需求原文及澄清摘要。"],
    locale: Annotated[str, "用户语言区域，如 zh_CN。"],
    workflow_type: Annotated[str, "工作流类型：A / B / C / D。"],
    workflow_confidence: Annotated[str, "high / medium / low。"],
):
    """将已明确的客户需求交给规划专家。"""
    return "ok"


@tool
def handoff_after_clarification(
    locale: Annotated[str, "用户语言区域，如 zh_CN。"],
    research_topic: Annotated[str, "澄清后的完整客户需求表述。"],
    workflow_type: Annotated[str, "工作流类型：A / B / C / D。"],
    workflow_confidence: Annotated[str, "high / medium / low。"],
):
    """澄清结束后交给规划专家。"""
    return "ok"


def needs_clarification(state: dict) -> bool:
    if not state.get("enable_clarification", False):
        return False
    clarification_rounds = state.get("clarification_rounds", 0)
    is_clarification_complete = state.get("is_clarification_complete", False)
    max_clarification_rounds = state.get("max_clarification_rounds", 3)
    return (
        clarification_rounds > 0
        and not is_clarification_complete
        and clarification_rounds <= max_clarification_rounds
    )


def preserve_state_meta_fields(state: State) -> dict:
    """在 Command.update 中显式带回，避免被默认值覆盖。"""
    return {
        "locale": state.get("locale", "zh_CN"),
        "research_topic": state.get("research_topic", ""),
        "original_topic": state.get("original_topic", ""),
        "clarified_research_topic": state.get("clarified_research_topic", ""),
        "clarification_history": state.get("clarification_history", []),
        "enable_clarification": state.get("enable_clarification", False),
        "max_clarification_rounds": state.get("max_clarification_rounds", 3),
        "clarification_rounds": state.get("clarification_rounds", 0),
        "resources": state.get("resources", []),
        "enable_background_investigation": state.get("enable_background_investigation", True),
        "auto_accepted_plan": state.get("auto_accepted_plan", False),
        "max_plan_iterations": state.get("max_plan_iterations", 3),
    }


def _plan_steps(plan: Plan | None) -> list:
    if plan is None:
        return []
    return list(getattr(plan, "steps", None) or [])


def _first_pending_step_by_type(plan: Plan | None, step_type: str):
    for step in _plan_steps(plan):
        if getattr(step, "step_type", "") != step_type:
            continue
        if not (step.execution_res or "").strip():
            return step
    return None


def _research_steps(plan: Plan | None) -> list:
    return [s for s in _plan_steps(plan) if getattr(s, "step_type", "") == "research"]


def _generate_searcher_summary(
    step_title: str, step_description: str, parsed: dict[str, Any]
) -> str:
    covered = parsed.get("coverage") if isinstance(parsed, dict) else ""
    return (
        f"### {step_title}\n"
        f"- 检索内容：{step_description}\n"
        f"- 覆盖度：{covered or '（未提供）'}"
    )


def _first_research_step_pending_researcher(plan: Plan | None):
    return _first_pending_step_by_type(plan, "research")


def _first_analysis_step_pending_analyst(plan: Plan | None):
    return _first_pending_step_by_type(plan, "analysis")


def _first_research_step_pending_curator(plan: Plan | None):
    for step in _plan_steps(plan):
        if getattr(step, "step_type", "") != "research":
            continue
        er = (step.execution_res or "").lstrip()
        if not er:
            continue
        if _RULE_DONE in (step.execution_res or ""):
            continue
        if er.startswith(_CURATOR_TAG):
            continue
        if er.startswith(_RESEARCHER_TAG):
            return step
    return None


def _first_research_step_pending_rule_splitter(plan: Plan | None):
    for step in _plan_steps(plan):
        if getattr(step, "step_type", "") != "research":
            continue
        er = (step.execution_res or "").lstrip()
        if _RULE_DONE in (step.execution_res or ""):
            continue
        if er.startswith(_CURATOR_TAG):
            return step
    return None


def _extract_tool_payloads_from_messages(messages: list[Any]) -> str:
    parts: list[str] = []
    n = 0
    for m in messages or []:
        if isinstance(m, ToolMessage):
            n += 1
            name = m.name or "tool"
            parts.append(f"### {name}_{n}\n{sanitize_tool_response(str(m.content))}")
        elif isinstance(m, dict) and (m.get("role") or "").lower() == "tool":
            n += 1
            parts.append(
                f"### {m.get('name', 'tool')}_{n}\n"
                f"{sanitize_tool_response(str(m.get('content', '')))}"
            )
    return "\n\n".join(parts) if parts else "（本轮无工具返回）"


def _last_ai_message(messages: list[Any]) -> AIMessage | None:
    for m in reversed(messages or []):
        if isinstance(m, AIMessage):
            return m
    return None


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text or not text.strip():
        return None
    raw = text.strip()
    try:
        return json.loads(repair_json_output(raw))
    except json.JSONDecodeError:
        pass

    # 退化路径：不使用正则，从文本中按括号平衡提取第一个 JSON 对象
    start = raw.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : idx + 1]
                    try:
                        return json.loads(repair_json_output(candidate))
                    except json.JSONDecodeError:
                        break
    return None


def _extract_xml_block(text: str, tag: str) -> str:
    """提取 <tag>…</tag> 标签内的文本（非贪婪，取第一个匹配）。无匹配返回空串。"""
    if not text:
        return ""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_analyst_output(text: str) -> dict[str, Any] | None:
    """
    解析 Analyst 输出（XML 标签 + 小 JSON 混合契约）。

    新契约（analyst.md 2026-06-11 起）：
        <analyst_meta>{ …小 JSON… }</analyst_meta>
        <analysis_text>…Markdown…</analysis_text>
        <contradiction_text>…</contradiction_text>
        <risk_text>…</risk_text>

    兼容路径：未发现 <analyst_meta> 标签时回退到旧版"单一大 JSON"解析，
    保证旧提示词缓存 / 模型未遵循新契约时不致硬失败。

    返回与旧契约同构的 dict（文本块以 analysis_text 等 key 并入），便于
    下游（reporter / replanning 判断）无感切换。
    """
    if not text or not text.strip():
        return None

    meta_raw = _extract_xml_block(text, "analyst_meta")
    if not meta_raw:
        # 旧契约回退：整体当作一个 JSON 对象解析
        return _parse_json_object(text)

    parsed = _parse_json_object(meta_raw)
    if parsed is None:
        return None

    for tag in ("analysis_text", "contradiction_text", "risk_text"):
        block = _extract_xml_block(text, tag)
        if block:
            parsed[tag] = block
    return parsed


def _normalize_plan_dict(plan_dict: dict[str, Any]) -> dict[str, Any]:
    """
    对 planner 输出做轻量结构修复，保证与 planner.zh_CN.md 的输入/输出契约一致。
    """
    fixed = dict(plan_dict or {})
    fixed.setdefault("thought", "")
    fixed.setdefault("title", "")
    fixed.setdefault("workflow_type", "A")
    fixed.setdefault("missing_conditions", [])
    steps = fixed.get("steps")
    if not isinstance(steps, list):
        steps = []
    normalized_steps: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        s = dict(step)
        # step_type / need_search 双向兜底
        if not s.get("step_type"):
            s["step_type"] = "research" if s.get("need_search", False) else "analysis"
        if "need_search" not in s:
            s["need_search"] = s.get("step_type") == "research"
        # planner.zh_CN.md 要求每步必须有 background
        s.setdefault("background", s.get("title", ""))
        s.setdefault("description", "")
        s.setdefault("title", "")
        normalized_steps.append(s)
    fixed["steps"] = normalized_steps
    return fixed


def _dedupe_citations_preserve_order(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        url = (c.get("url") or "").strip()
        title = (c.get("title") or "").strip()
        key = url if url else f"title:{title}"
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _format_citation_list_for_reporter(citations: list[dict[str, Any]]) -> str:
    ordered = _dedupe_citations_preserve_order(citations)
    if not ordered:
        return (
            "### 可用参考来源（Citation list）\n\n"
            "当前无结构化引用条目。**禁止编造 URL。**\n"
            "若无链接，仅用《文档名》指代，勿使用 `[[n]](#ref-n)`。"
        )
    lines = [
        "### 可用参考来源（Citation list）",
        "",
        "正文可溯源结论须在句末使用 **`[[n]](#ref-n)`**，n 与下表序号一致；勿编造未列出来源。",
        "",
    ]
    for i, c in enumerate(ordered, 1):
        title = str(c.get("title") or "Untitled").strip() or "Untitled"
        url = str(c.get("url") or "").strip()
        extra = c.get("extra") if isinstance(c.get("extra"), dict) else {}
        lines.append(f"{i}. **{title}**")
        if url:
            lines.append(f"   - URL: `{url}`")
        lines.append("")
    return "\n".join(lines).rstrip()


def extract_plan_content(plan_data: str | dict | Any) -> str:
    if isinstance(plan_data, str):
        return plan_data
    if hasattr(plan_data, "content") and isinstance(plan_data.content, str):
        return plan_data.content
    if isinstance(plan_data, dict):
        if "content" in plan_data:
            c = plan_data["content"]
            if isinstance(c, str):
                return c
            if isinstance(c, dict):
                return json.dumps(c, ensure_ascii=False)
            return str(c)
        return json.dumps(plan_data, ensure_ascii=False)
    return str(plan_data)


async def _handle_recursion_limit_fallback(
    messages: list,
    agent_name: str,
    state: State,
) -> list:
    """Handle GraphRecursionError with graceful fallback using LLM summary.

    When the agent hits the recursion limit, this function generates a final output
    using only the observations already gathered, without calling any tools.

    Args:
        messages: Messages accumulated during agent execution before hitting limit
        agent_name: Name of the agent that hit the limit
        current_step: The current step being executed
        state: Current workflow state

    Returns:
        list: Messages including the accumulated messages plus the fallback summary

    Raises:
        Exception: If the fallback LLM call fails
    """
    logger.warning(
        f"Recursion limit reached for {agent_name} agent. "
        f"Attempting graceful fallback with {len(messages)} accumulated messages."
    )

    if len(messages) == 0:
        return messages

    cleared_messages = messages.copy()
    while len(cleared_messages) > 0 and cleared_messages[-1].type == "system":
        cleared_messages = cleared_messages[:-1]

    # Prepare state for prompt template
    fallback_state = {
        "locale": state.get("locale", "en-US"),
    }

    # Apply the recursion_fallback prompt template
    limit_prompt = get_system_prompt_template("recursion_fallback", fallback_state, None)
    fallback_messages = cleared_messages + [
        SystemMessage(content=limit_prompt)
    ]

    # Get the LLM without tools (strip all tools from binding)
    fallback_llm = get_llm_by_type(AGENT_LLM_MAP[agent_name])

    # Call the LLM with the updated messages
    fallback_response = fallback_llm.invoke(fallback_messages)
    fallback_content = fallback_response.content

    logger.info(
        f"Graceful fallback succeeded for {agent_name} agent. "
        f"Generated summary of {len(fallback_content)} characters."
    )

    # Sanitize response
    fallback_content = sanitize_tool_response(str(fallback_content))

    # Return the accumulated messages plus the fallback response
    result_messages = list(cleared_messages)
    result_messages.append(AIMessage(content=fallback_content, name=agent_name))

    return result_messages


async def background_investigation_node(state: State, config: RunnableConfig) -> dict:
    """BGI / Analyzer 双模式节点（BI + CP 共享，按 pipeline_mode 路由提示词）。

    BI 模式（pipeline_mode="bi"）：
        阶段 1 — 调用 BI/background_investigator.md（ReAct Agent + KB 探索）→ kb_panorama
        阶段 2 — 调用 BI/ontology_mapper.md（纯 LLM 本体映射）→ background_investigation_results

    CP 模式（pipeline_mode="cp"）：
        调用 CP/analyzer.md（纯 LLM 审查范围分析）→ analyzer_output
        不做 KB 探索和本体映射（CP 无需背景调研，Analyzer 直接分析宣传文本）
    """
    logger.info("background_investigation_node running (mode=%s)", state.get("pipeline_mode", "bi"))
    configurable = Configuration.from_runnable_config(config)
    mode = state.get("pipeline_mode", "bi")

    if mode == "cp":
        return await _run_cp_analyzer(state, configurable)
    else:
        return await _run_bi_bgi_and_mapper(state, config, configurable)


async def _run_cp_analyzer(state: State, configurable: Configuration) -> dict:
    """CP 模式：调用 CP/analyzer.md 分析宣传文本。"""
    promo = state.get("promotional_text", "")
    if not promo:
        logger.warning("CP analyzer: no promotional_text, skipping")
        return {
            "analyzer_output": "",
            "background_investigation_results": json.dumps([], ensure_ascii=False),
            "kb_panorama": "",
            **preserve_state_meta_fields(state),
        }

    from .ontology import get_cp_analyzer_template_vars

    sub: dict = {
        **state,
        **get_cp_analyzer_template_vars(),
        "messages": [HumanMessage(content=f"## 信用卡业务宣传文本\n\n{promo}")],
    }
    llm = get_llm_by_type(AGENT_LLM_MAP.get("cp_analyzer", "basic"))
    try:
        resp = await llm.ainvoke(
            apply_prompt_template("CP/analyzer", sub, configurable)
        )
        analyzer_output = str(resp.content or "")
    except Exception as e:
        logger.exception("CP analyzer failed: %s", e)
        analyzer_output = f"[Analyzer 执行失败: {e}]"

    return {
        "analyzer_output": analyzer_output,
        "background_investigation_results": json.dumps([], ensure_ascii=False),
        "kb_panorama": "",
        **preserve_state_meta_fields(state),
    }


async def _run_bi_bgi_and_mapper(
    state: State, config: RunnableConfig, configurable: Configuration
) -> dict:
    """BI 模式：BGI 多角度 KB 探索 + Ontology Mapper 双阶段。"""
    q = state.get("clarified_research_topic") or state.get("research_topic", "")
    if not q:
        return {
            "background_investigation_results": json.dumps([], ensure_ascii=False),
            "kb_panorama": "",
            **preserve_state_meta_fields(state),
        }

    locale = state.get("locale", "zh_CN")
    wf = state.get("workflow_type", "A")

    # ════════════════════════════════════════════════════
    #  阶段 1：BGI 多角度 KB 探索 → kb_panorama
    # ════════════════════════════════════════════════════
    kb_panorama = ""
    tools = [t for t in [get_retriever_tool(state.get("resources", [])), crawl_tool] if t]
    if tools:
        # BGI 输入：用户问题 + 工作流类型（Coordinator 已输出）
        # 注意：missing_conditions 由 Planner 输出，BGI 在 Planner 之前执行，不可引用
        bgi_user = f"## 用户问题\n{q}\n\n## 工作流类型\n{wf}\n"

        llm_limit = get_llm_token_limit_by_type(
            AGENT_LLM_MAP.get("background_investigator", "basic")
        )
        hook = partial(ContextManager(llm_limit, 3).compress_messages)
        bgi_agent = create_agent(
            "background_investigator",
            "background_investigator",
            tools,
            "background_investigator",
            hook,
            interrupt_before_tools=configurable.interrupt_before_tools,
            locale=locale,
        )
        try:
            bgi_out = await bgi_agent.ainvoke(
                {"messages": [HumanMessage(content=bgi_user)]},
                config={"recursion_limit": 15},
            )
            bgi_msgs = bgi_out.get("messages", [])
            bgi_last = _last_ai_message(bgi_msgs)
            kb_panorama = sanitize_tool_response(
                str(get_message_content(bgi_last) or "")
            )
        except Exception as e:
            logger.warning("BGI agent failed, kb_panorama will be empty: %s", e)

    # ════════════════════════════════════════════════════
    #  阶段 2：Ontology Mapper → background_investigation_results
    # ════════════════════════════════════════════════════
    mapper_human = f"### 用户问题\n{q}\n"
    if kb_panorama:
        mapper_human += (
            f"\n### 知识库全景概要（仅用于确认 KB 准确名称）\n{kb_panorama[:4000]}\n"
        )
    mapper_sub = {
        **state,
        "ontology_skeleton": render_skeleton_for_mapper(),
        "messages": [HumanMessage(content=mapper_human)],
    }
    mapping = ""
    try:
        mapper_llm = get_llm_by_type(AGENT_LLM_MAP.get("ontology_mapper", "basic"))
        mapper_content = str(
            (await mapper_llm.ainvoke(
                apply_prompt_template("ontology_mapper", mapper_sub, configurable)
            )).content or ""
        )
        mapping = extract_ontology_mapping(mapper_content)
    except Exception as e:
        logger.warning("ontology_mapper LLM failed: %s", e)

    if mapping:
        payload_str = mapping
    else:
        logger.warning(
            "ontology_mapper: no valid <ontology_mapping>, fallback to raw KB panorama"
        )
        payload_str = json.dumps(
            [{"query": q, "summary": kb_panorama}] if kb_panorama else [],
            ensure_ascii=False,
        )

    return {
        "background_investigation_results": payload_str,
        "kb_panorama": kb_panorama,
        **preserve_state_meta_fields(state),
    }


def planner_node(state: State, config: RunnableConfig) -> Command:
    """生成计划后跳转 human_feedback（图中无 planner→human_feedback 静态边）。"""
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state.get("plan_iterations", 0)
    iter_delta = 0 if state.get("skip_next_plan_iteration_increment") else 1
    max_step_num = (
        (config.get("configurable") or {}).get("max_step_num")
        or configurable.max_step_num
        or 5
    )
    wf = state.get("workflow_type", "A")
    prev_plan = state.get("current_plan")
    was_replan = bool(state.get("replanning_needed"))
    last_plan_text = state.get("last_plan_text") or ""

    planner_state: dict = {**state, "workflow_type": wf, "max_step_num": max_step_num}
    if state.get("enable_clarification", False) and state.get("clarified_research_topic"):
        modified = {**planner_state, "research_topic": state["clarified_research_topic"]}
        modified["messages"] = [{"role": "user", "content": state["clarified_research_topic"]}]
        messages = apply_prompt_template("planner", modified, configurable)
    else:
        messages = apply_prompt_template("planner", planner_state, configurable)

    if state.get("enable_background_investigation") and state.get("background_investigation_results"):
        bg_results = str(state["background_investigation_results"])
        # P1 修正（2026-06-10）：本体映射 + KB 概要双通道注入。
        # 本体边=必查维度硬性下限；KB 概要=长尾覆盖 + KB 准确名称权威来源。
        guardrail = render_planner_guardrail(bg_results)
        if guardrail:
            messages += [
                {
                    "role": "user",
                    "content": (
                        "## 本体映射结果（Ontology Mapping）\n"
                        + extract_ontology_mapping(bg_results)
                        + "\n\n"
                        + guardrail
                    ),
                }
            ]
        else:
            messages += [
                {
                    "role": "user",
                    "content": "背景调查参考：\n" + bg_results,
                }
            ]
        # KB 概要独立注入（无论本体映射是否成功，只要有 KB 概要就下发）
        kb_panorama = state.get("kb_panorama") or ""
        if kb_panorama.strip():
            messages += [
                {
                    "role": "user",
                    "content": (
                        "## 知识库探索概要（KB 准确名称以此为准）\n"
                        "以下为知识库轻量检索结果，用于：\n"
                        "1. **KB 准确名称校准**——知识库中的产品/业务准确名称以此为准"
                        "（优先于本体映射 <kb_terms> 中的名称）\n"
                        "2. **长尾信息参考**——本体映射覆盖核心骨架（16 类 + 20 边），"
                        "本体 <unmapped> 之外的长尾业务信息以此为线索安排探索性检索\n\n"
                        + kb_panorama[:6000]
                    ),
                }
            ]
    if state.get("replanning_reason"):
        messages += [
            {
                "role": "user",
                "content": "## 重规划反馈\n"
                + str(state["replanning_reason"])
                + "\n请调整检索步骤，避免重复无效路径。",
            }
        ]
    if was_replan and last_plan_text.strip():
        messages += [
            {
                "role": "user",
                "content": "## 上一版计划（原始文本）\n```\n" + last_plan_text.strip() + "\n```",
            }
        ]

        replan_section = ""
        if is_replan:
            findings_text = "\n".join(f"- {f}" for f in existing_findings) if existing_findings else "无"
            replan_section = f"""
    ## ⚠️ 重新规划（Replan）

    **原因**: {replanning_reason}

    **上一轮计划**:
    ```
    {last_plan_text}
    ```

    **已有发现**:
    {findings_text}

    **要求**: 针对缺失信息设计补充调研步骤，避免重复已完成的检索。
    """

    llm = get_llm_by_type(AGENT_LLM_MAP.get("planner", "basic"))
    full_response = ""
    for chunk in llm.stream(messages):
        full_response += chunk.content

    # 注意：不在 planner 阶段做 Plan.model_validate。
    # 原始计划文本先透传给 human_feedback，便于人类编辑后再做最终校验。
    candidate_plan_text = full_response

    reset = {
        "searcher_results": [],
        "searcher_summaries": [],
        "curator_rule_splitter_views": [],
        "atomic_rules": [],
        "arbitration_result": "",
        "analyst_output": {},
        "observations": [],
        "replanning_needed": False,
        "replanning_reason": "",
        "_plan_validator_needs_rerun": False,
        "planner_override_occurred": False,
        "structure_validation_retried": False,
    }
    update: dict[str, Any] = {
        **preserve_state_meta_fields(state),
        "messages": [AIMessage(content=full_response, name="planner")],
        "current_plan": candidate_plan_text,
        "current_plan_last_round": prev_plan if was_replan and prev_plan is not None else None,
        "last_plan_text": full_response,
        # workflow_type / missing_conditions 在 human_feedback 完成最终校验后再确定
        "workflow_type": wf,
        "workflow_confidence": state.get("workflow_confidence", "high"),
        "plan_iterations": plan_iterations + iter_delta,
        "skip_next_plan_iteration_increment": False,
        **reset,
    }
    return Command(update=update, goto="human_feedback")


def human_feedback_node(
    state: State, config: RunnableConfig
) -> Command[Literal["plan_validator", "reporter", "__end__"]]:
    current_plan = state.get("current_plan", "")

    # if the plan is accepted, run the following node
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0

    original_plan = current_plan

    try:
        # Repair the JSON output
        current_plan = repair_json_output(current_plan)
        # parse the plan to dict
        current_plan = json.loads(current_plan)
        current_plan_content = extract_plan_content(current_plan)
        
        # increment the plan iterations
        plan_iterations += 1
        # parse the plan
        new_plan = json.loads(repair_json_output(current_plan_content))
        new_plan = _normalize_plan_dict(new_plan)
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to parse plan: {str(e)}. Plan data type: {type(current_plan).__name__}")
        if isinstance(current_plan, dict) and "content" in original_plan:
            logger.warning(f"Plan appears to be an AIMessage object with content field")
        if plan_iterations > 1:  # the plan_iterations is increased before this check
            return Command(
                update=preserve_state_meta_fields(state),
                goto="reporter"
            )
        else:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__"
            )

    if not new_plan:
        return Command(
            update=preserve_state_meta_fields(state),
            goto="__end__"
        )

    current_plan = Plan.model_validate(new_plan)
    for s in _plan_steps(current_plan):
        s.execution_res = None

    # Build update dict with safe locale handling
    update_dict = {
        "current_plan": current_plan,
        "plan_iterations": plan_iterations,
        "workflow_type": getattr(current_plan, "workflow_type", None) or state.get("workflow_type", "A"),
        "missing_conditions": list(getattr(current_plan, "missing_conditions", []) or []),
        **preserve_state_meta_fields(state),
    }
    
    # Only override locale if new_plan provides a valid value, otherwise use preserved locale
    if new_plan.get("locale"):
        update_dict["locale"] = new_plan["locale"]
    
    return Command(
        update=update_dict,
        goto="plan_validator",
    )


def plan_validator_node(state: State, config: RunnableConfig) -> dict:
    plan = state.get("current_plan")
    coord_wf = state.get("workflow_type", "A")
    base = {**preserve_state_meta_fields(state), "_plan_validator_needs_rerun": False}

    if plan is None:
        return {
            **base,
            "workflow_type": state.get("workflow_type", "A"),
            "workflow_confidence": state.get("workflow_confidence", "high"),
            "_plan_validator_needs_rerun": True,
            "skip_next_plan_iteration_increment": True,
        }

    plan_wf = getattr(plan, "workflow_type", None) or coord_wf
    if plan_wf != coord_wf and not state.get("planner_override_occurred", False):
        return {
            **base,
            "workflow_type": plan_wf,
            "planner_override_occurred": True,
            "_plan_validator_needs_rerun": True,
            "skip_next_plan_iteration_increment": True,
            "current_plan": None,
        }

    ok, err = validate_plan_structure(plan, plan_wf)
    if not ok:
        logger.warning("Plan structure invalid: %s", err)
        if not state.get("structure_validation_retried", False):
            return {
                **base,
                "workflow_type": "A",
                "structure_validation_retried": True,
                "_plan_validator_needs_rerun": True,
                "skip_next_plan_iteration_increment": True,
                "current_plan": None,
            }

    plan_text = json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)

    return {
        **base,
        "last_plan_text": plan_text,
        "workflow_type": state.get("workflow_type", "A"),
        "workflow_confidence": state.get("workflow_confidence", "high"),
    }


async def coordinator_node(state: State, config: RunnableConfig) -> Command:
    configurable = Configuration.from_runnable_config(config)
    locale = state.get("locale", "zh_CN")
    agent = create_agent(
        "coordinator",
        "coordinator",
        [handoff_to_planner, handoff_after_clarification],
        "coordinator",
        interrupt_before_tools=configurable.interrupt_before_tools,
        locale=locale,
    )
    extra: list = []
    if not state.get("enable_clarification", False):
        extra.append(
            HumanMessage(
                content="[系统] 不进行多轮澄清时请直接 handoff_to_planner，并填写 workflow_type / workflow_confidence。"
            )
        )
    
    result = await agent.ainvoke(
        {"messages": (state.get("messages") or []) + extra},
        config={"recursion_limit": 25},
    )
    msgs = result.get("messages", [])
    last = _last_ai_message(msgs)
    if not last:
        # 无有效回复时保持在澄清回合，避免错误直跳下游
        return Command(
            update={**preserve_state_meta_fields(state), "messages": msgs},
            goto=END,
        )

    calls = getattr(last, "tool_calls", None) or []
    if calls:
        args = calls[0].get("args") or {}
        wf = (args.get("workflow_type") or "A").strip().upper()[:1] or "A"
        if wf not in {"A", "B", "C", "D"}:
            wf = "A"
        wconf = (args.get("workflow_confidence") or "medium").lower()
        if wconf not in ("high", "medium", "low"):
            wconf = "medium"
        topic = (args.get("research_topic") or state.get("research_topic", "")).strip()
        clar = reconstruct_clarification_history(
            msgs,
            fallback_history=state.get("clarification_history", []),
            base_topic=state.get("research_topic", ""),
        )
        _, latest_u = get_latest_user_message(msgs)
        if latest_u and (not clar or clar[-1] != latest_u):
            clar = clar + [latest_u] if clar else [latest_u]
        clarified = topic or state.get("research_topic", "")
        orig = (state.get("original_topic") or "").strip() or state.get("research_topic", "")
        return Command(
            update={
                **preserve_state_meta_fields(state),
                "messages": msgs,
                "research_topic": state.get("research_topic", topic) or topic,
                "original_topic": orig,
                "clarified_research_topic": clarified,
                "clarification_history": clar,
                "workflow_type": wf,
                "workflow_confidence": wconf,
                "is_clarification_complete": True,
            },
            goto="background_investigator",
        )

    text = (get_message_content(last) or "").strip()
    if not text:
        return Command(
            update={**preserve_state_meta_fields(state), "messages": msgs},
            goto=END,
        )

    # 未触发 handoff 说明 coordinator 仍在澄清。
    # 达到最大澄清轮次后自动收敛，沿用已知信息继续下游。
    enable_clarification = state.get("enable_clarification", False)
    current_rounds = state.get("clarification_rounds", 0) + 1
    max_rounds = state.get("max_clarification_rounds", 3)
    if enable_clarification and current_rounds > max_rounds:
        clar = reconstruct_clarification_history(
            msgs,
            fallback_history=state.get("clarification_history", []),
            base_topic=state.get("research_topic", ""),
        )
        clarified_topic, _ = build_clarified_topic_from_history(clar)
        topic = clarified_topic or state.get("clarified_research_topic") or state.get("research_topic", "")
        return Command(
            update={
                **preserve_state_meta_fields(state),
                "messages": msgs,
                "clarification_history": clar,
                "clarification_rounds": current_rounds,
                "is_clarification_complete": True,
                "clarified_research_topic": topic,
                "workflow_type": state.get("workflow_type", "A"),
                "workflow_confidence": state.get("workflow_confidence", "low"),
            },
            goto="background_investigator",
        )

    return Command(
        update={
            **preserve_state_meta_fields(state),
            "messages": msgs,
            "clarification_rounds": current_rounds,
            "is_clarification_complete": False,
        },
        goto=END,
    )


async def researcher_node(state: State, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    plan = state.get("current_plan")
    step = _first_research_step_pending_researcher(plan)
    if step is None:
        return {**preserve_state_meta_fields(state)}

    tools = [t for t in [get_retriever_tool(state.get("resources", [])), crawl_tool] if t]
    locale = state.get("locale", "zh_CN")
    wf = state.get("workflow_type", "A")
    wf_label = {"A": "A 定点调查", "B": "B 并行对比", "C": "C 扫描穷举", "D": "D 条件推理"}.get(
        wf, "A 定点调查"
    )
    summaries = "\n\n".join(state.get("searcher_summaries", [])) or "（无）"
    q = state.get("clarified_research_topic") or state.get("research_topic", "")
    user = (
        f"## 调查原始问题\n{q}\n\n## 工作流类型\n{wf_label}\n\n"
        f"## 已完成步骤检索摘要\n{summaries}\n\n"
        f"## 当前步骤\n- 标题：{step.title}\n- 背景：{step.background}\n- 检索内容：{step.description}\n"
    )
    attachments_block = format_attached_files_for_prompt(state)
    if attachments_block:
        user += f"\n## 用户附件\n\n{attachments_block}\n"
    rs = _research_steps(plan)
    step_no = next((i + 1 for i, s in enumerate(rs) if s is step), 0)

    llm_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP.get("researcher", "basic"))
    hook = partial(ContextManager(llm_limit, 3).compress_messages)
    agent = create_agent(
        "researcher",
        "researcher",
        tools,
        "researcher",
        hook,
        interrupt_before_tools=configurable.interrupt_before_tools,
        locale=locale,
    )
    try:
        out = await agent.ainvoke({"messages": [HumanMessage(content=user)]}, config={"recursion_limit": 25})
        ams = out.get("messages", [])
        light = sanitize_tool_response(str(get_message_content(_last_ai_message(ams)) or ""))
        raw = _extract_tool_payloads_from_messages(ams)
        bundle = f"## Step {step_no} — {step.title}\n### 检索注释\n{light}\n\n### 原始工具返回\n{raw}"
        step.execution_res = f"{_RESEARCHER_TAG}\n{light}"
    except GraphRecursionError as e:
        logger.warning("researcher recursion_limit fallback: %s", e)
        ams = await _handle_recursion_limit_fallback(
            [HumanMessage(content=user)],
            "researcher",
            state,
        )
        light = sanitize_tool_response(str(get_message_content(_last_ai_message(ams)) or ""))
        raw = _extract_tool_payloads_from_messages(ams)
        bundle = (
            f"## Step {step_no} — {step.title}\n"
            f"### 检索注释\n{light}\n\n"
            f"### 原始工具返回\n{raw}"
        )
        step.execution_res = f"{_RESEARCHER_TAG}\n{light}"
    except Exception as e:
        logger.exception("researcher: %s", e)
        step.execution_res = f"{_RESEARCHER_TAG}\n错误: {e}"
        return {
            **preserve_state_meta_fields(state),
            "current_plan": plan,
            "searcher_results": state.get("searcher_results", [])
            + [f"## Step {step_no}\n### 错误\n{step.execution_res}"],
        }
    
    return {
        **preserve_state_meta_fields(state),
        "current_plan": plan,
        "searcher_results": state.get("searcher_results", []) + [bundle],
            "citations": merge_citations(
                state.get("citations", []),
                extract_citations_from_messages(ams),
            ),
    }

async def curator_node(state: State, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    plan = state.get("current_plan")
    step = _first_research_step_pending_curator(plan)
    if step is None:
        return {**preserve_state_meta_fields(state)}

    rs = _research_steps(plan)
    idx = next((i + 1 for i, s in enumerate(rs) if s is step), 0)
    latest_raw = (state.get("searcher_results", []) or [])[-1] if state.get("searcher_results") else ""
    q = state.get("clarified_research_topic") or state.get("research_topic", "")
    human = (
        f"### 调查原始问题\n{q}\n\n### 当前步骤\n"
        f"- 标题：{step.title}\n- 背景：{step.background}\n- 评估内容：{step.description}\n\n"
        f"### 检索摘要\n{step.execution_res}\n\n### 原始工具返回\n{latest_raw}\n"
    )
    sub = {**state, "messages": [HumanMessage(content=human)]}
    llm = get_llm_by_type(AGENT_LLM_MAP.get("curator", "basic"))
    content = str((await llm.ainvoke(apply_prompt_template("curator", sub, configurable))).content or "")
    parsed = _parse_json_object(content)
    full = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else content
    summary = _generate_searcher_summary(
        step.title, step.description, parsed if isinstance(parsed, dict) else {}
    )
    step.execution_res = f"{_CURATOR_TAG}\n{full}"
    views = list(state.get("curator_rule_splitter_views", [])) + [
        f"### 步骤 {idx} 信息质量评估\n{full}"
    ]
    sums = list(state.get("searcher_summaries", [])) + [summary]
    return {
        **preserve_state_meta_fields(state),
        "current_plan": plan,
        "curator_rule_splitter_views": views,
        "searcher_summaries": sums,
    }


async def rule_splitter_node(state: State, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    plan = state.get("current_plan")
    step = _first_research_step_pending_rule_splitter(plan)
    if step is None or _RULE_DONE in (step.execution_res or ""):
        return {**preserve_state_meta_fields(state), "current_plan": plan}

    views = state.get("curator_rule_splitter_views", [])
    if not views:
        return {**preserve_state_meta_fields(state), "current_plan": plan}

    q = state.get("clarified_research_topic") or state.get("research_topic", "")
    human = f"### 调查原始问题\n{q}\n\n### 信息质量评估\n{views[-1]}\n"
    sub = {**state, "messages": [HumanMessage(content=human)]}
    llm = get_llm_by_type(AGENT_LLM_MAP.get("rule_splitter", "basic"))
    rules = str((await llm.ainvoke(apply_prompt_template("rule_splitter", sub, configurable))).content or "")
    er = step.execution_res or ""
    if _RULE_DONE not in er:
        step.execution_res = f"{er.rstrip()}\n\n{_RULE_DONE}"
    atoms = list(state.get("atomic_rules", [])) + [rules]
    return {
        **preserve_state_meta_fields(state),
        "current_plan": plan,
        "atomic_rules": atoms,
    }


async def arbitrator_node(state: State, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    atoms = state.get("atomic_rules", [])
    if not atoms:
        return {
            **preserve_state_meta_fields(state),
            "arbitration_result": "无原子规则可供仲裁。",
        }
    wf = state.get("workflow_type", "A")
    labels = {"A": "A 定点调查", "B": "B 并行对比", "C": "C 扫描穷举", "D": "D 条件推理"}
    q = state.get("clarified_research_topic") or state.get("research_topic", "")
    blob = "\n\n---\n\n".join(f"#### {i + 1}\n{t}" for i, t in enumerate(atoms))
    human = (
        f"### 调查原始问题\n{q}\n\n"
        f"### 当前工作流类型\n{labels.get(wf, wf)}\n\n"
        f"### 关联业务原子规则清单\n{blob}\n"
    )
    sub = {
        **state,
        "arbitrator_bucketing_guide": render_arbitrator_bucketing_guide(),
        "messages": [HumanMessage(content=human)],
    }
    llm = get_llm_by_type(AGENT_LLM_MAP.get("arbitrator", "basic"))
    out = str((await llm.ainvoke(apply_prompt_template("arbitrator", sub, configurable))).content or "")
    return {**preserve_state_meta_fields(state), "arbitration_result": out.strip()}


async def analyst_node(state: State, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    plan = state.get("current_plan")
    analysis_step = _first_analysis_step_pending_analyst(plan)
    views = "\n\n".join(state.get("curator_rule_splitter_views", []))
    q = state.get("clarified_research_topic") or state.get("research_topic", "")
    wf = state.get("workflow_type", "A")
    replan_it = state.get("replan_iterations", 0)
    missing_block = ""
    if wf == "D":
        miss = state.get("missing_conditions", [])
        miss_txt = "\n".join(f"- {m}" for m in miss) if miss else "（无）"
        missing_block = f"### 未提供关键条件（D）\n{miss_txt}\n\n"
    replan_limit_block = ""
    if replan_it > 1:
        replan_limit_block = (
            "### 重规划次数提示\n"
            f"当前 replan_iterations={replan_it}，已达到重规划上限。\n"
            "请基于现有证据直接输出最佳可得结论与局限性说明，"
            "不要再输出 `replanning_needed=true`。\n\n"
        )
    human = (
        f"### 调查原始问题\n{q}\n\n"
        f"{missing_block}"
        f"{replan_limit_block}"
        f"### 信息质量评估（各步）\n{views or '（无）'}\n\n"
        f"### 仲裁报告\n{state.get('arbitration_result', '（无）')}\n\n"
        f"### 分析步骤\n- 标题：{getattr(analysis_step, 'title', '')}\n"
        f"- 背景：{getattr(analysis_step, 'background', '')}\n"
        f"- 要求：{getattr(analysis_step, 'description', '')}\n"
    )
    sub = {**state, "messages": [HumanMessage(content=human)], "workflow_type": wf}
    llm = get_llm_by_type(AGENT_LLM_MAP.get("analyst", "basic"))
    text = str((await llm.ainvoke(apply_prompt_template("analyst", sub, configurable))).content or "").strip()
    parsed = _parse_analyst_output(text)
    replan = bool(parsed and (parsed.get("replanning_needed") or parsed.get("replanningNeeded")))
    low = text.lower()
    if not replan and '"replanning_needed": true' in low:
        replan = True
    reason = ""
    if isinstance(parsed, dict):
        reason = str(parsed.get("replanning_reason") or parsed.get("gap_analysis") or "")

    if analysis_step is not None:
        analysis_step.execution_res = "Analyst 已完成。"

    obs = list(state.get("observations", [])) + ([text] if text else [])
    replan_it = replan_it + (1 if replan else 0)
    return {
        **preserve_state_meta_fields(state),
        "current_plan": plan,
        "observations": obs,
        "analyst_output": parsed if isinstance(parsed, dict) else {},
        "replanning_needed": replan,
        "replanning_reason": reason or state.get("replanning_reason", ""),
        "replan_iterations": replan_it,
    }


def _render_comparison_table_md(table: dict[str, Any]) -> str:
    """将 analyst 的 comparison_table JSON 结构渲染为 Markdown 表格。

    期望结构：{"dimensions": [...], "objects": {"对象A": {维度: 值}}, "key_differences": [...]}
    objects 的值兼容 dict（按维度取值）和 list（按维度顺序对位）两种形态。
    """
    dims = [str(d) for d in (table.get("dimensions") or [])]
    objects = table.get("objects") or {}
    if not dims or not isinstance(objects, dict) or not objects:
        return ""

    obj_names = [str(k) for k in objects.keys()]
    lines = [
        "| 对比维度 | " + " | ".join(obj_names) + " |",
        "|:---|" + "|".join([":---"] * len(obj_names)) + "|",
    ]
    for i, dim in enumerate(dims):
        row = [dim]
        for name in obj_names:
            vals = objects.get(name)
            cell = ""
            if isinstance(vals, dict):
                cell = str(vals.get(dim, "知识库中未查到"))
            elif isinstance(vals, list):
                cell = str(vals[i]) if i < len(vals) else "知识库中未查到"
            elif vals is not None:
                cell = str(vals)
            row.append(cell.replace("\n", " ").replace("|", "／") or "知识库中未查到")
        lines.append("| " + " | ".join(row) + " |")

    diffs = table.get("key_differences") or []
    if diffs:
        lines.append("")
        lines.append("**关键差异**：")
        for d in diffs:
            lines.append(f"- {d}")
    return "\n".join(lines)


def _render_analyst_observation_for_reporter(
    parsed: dict[str, Any], wf: str
) -> str:
    """
    将 analyst_output（已解析 dict）程序化渲染为 reporter 可直接阅读的
    分节 Markdown，替代向 reporter 注入原始 JSON 文本。

    渲染失败/字段缺失均逐节降级，不抛异常。
    """
    parts: list[str] = []

    sc = parsed.get("scope_coverage") or {}
    if isinstance(sc, dict) and sc:
        parts.append(
            "### 调查覆盖范围\n"
            f"- 完整范围（full_scope）：{'、'.join(map(str, sc.get('full_scope') or [])) or '（不适用）'}\n"
            f"- 已覆盖（covered）：{'、'.join(map(str, sc.get('covered') or [])) or '（不适用）'}\n"
            f"- 未覆盖（not_covered）：{'、'.join(map(str, sc.get('not_covered') or [])) or '无'}"
        )

    conf = parsed.get("overall_confidence")
    if conf:
        parts.append(f"### 整体置信度\n{conf}")

    unc = parsed.get("uncovered_aspects") or []
    if unc:
        parts.append("### 未覆盖方面\n" + "\n".join(f"- {u}" for u in unc))

    conclusions = parsed.get("conclusions") or []
    if conclusions:
        c_lines = ["### 业务结论"]
        for i, c in enumerate(conclusions, 1):
            if not isinstance(c, dict):
                c_lines.append(f"{i}. {c}")
                continue
            line = f"{i}. {c.get('statement', '')}（置信度：{c.get('confidence', '')}）"
            srcs = c.get("supporting_sources") or []
            if srcs:
                line += f"\n   - 支撑来源：{'、'.join(f'《{s}》' for s in srcs)}"
            conds = c.get("conditions")
            if conds:
                line += f"\n   - 条件前提：{'；'.join(map(str, conds))}"
            if c.get("scope_note"):
                line += f"\n   - 范围说明：{c['scope_note']}"
            if c.get("expired_only"):
                line += "\n   - ⚠️ 该结论仅基于已过期依据，仅供历史参考"
            c_lines.append(line)
        parts.append("\n".join(c_lines))

    # ── 工作流特定结构 ──
    if wf == "B":
        table = parsed.get("comparison_table")
        if isinstance(table, dict):
            md = _render_comparison_table_md(table)
            if md:
                parts.append("### 对比表格（已渲染）\n" + md)
    elif wf == "C":
        enum = parsed.get("enumeration_list")
        if isinstance(enum, dict) and enum:
            e_lines = ["### 穷举列表"]
            if enum.get("condition"):
                e_lines.append(f"- 穷举条件：{enum['condition']}")
            for label, key in (
                ("已确认项目", "confirmed_items"),
                ("存疑项目", "uncertain_items"),
                ("仅过期来源项目", "expired_source_items"),
            ):
                items = enum.get(key) or []
                if items:
                    e_lines.append(f"- {label}：")
                    e_lines.extend(f"  - {it}" for it in items)
            if enum.get("completeness_note"):
                e_lines.append(f"- 完备性说明：{enum['completeness_note']}")
            parts.append("\n".join(e_lines))
    elif wf == "D":
        chain = parsed.get("condition_chain")
        if isinstance(chain, dict) and chain:
            d_lines = ["### 条件推理链"]
            if chain.get("target"):
                d_lines.append(f"- 判定目标：{chain['target']}")
            conds = chain.get("conditions") or []
            if conds:
                d_lines.append("- 条件清单：")
                d_lines.extend(f"  - {c}" for c in conds)
            if chain.get("reasoning_summary"):
                d_lines.append(f"- 推理概述：{chain['reasoning_summary']}")
            negs = chain.get("negative_rules") or []
            if negs:
                d_lines.append("- 否定性规则：")
                d_lines.extend(f"  - {n}" for n in negs)
            parts.append("\n".join(d_lines))

    for title, key, empty_hint in (
        ("分析过程", "analysis_text", "（无）"),
        ("矛盾信息处理", "contradiction_text", "未发现矛盾信息。"),
        ("风险提示", "risk_text", "未发现需要提示的风险。"),
    ):
        parts.append(f"### {title}\n{parsed.get(key) or empty_hint}")

    return "\n\n".join(parts)


async def reporter_node(state: State, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    plan = state.get("current_plan")
    wf = state.get("workflow_type", "A")
    thought = getattr(plan, "thought", "") if plan else ""

    # ── 分析结论：优先消费已解析的 analyst_output（程序化渲染分节 Markdown），
    #    解析失败时回退到 observations 原文（旧行为） ──
    analyst_parsed = state.get("analyst_output") or {}
    if isinstance(analyst_parsed, dict) and analyst_parsed.get("conclusions"):
        obs_text = _render_analyst_observation_for_reporter(analyst_parsed, wf)
    else:
        observations = list(state.get("observations", []))
        obs_text = "\n\n---\n\n".join(observations)

    citations = [c for c in (state.get("citations") or []) if isinstance(c, dict)]
    citations_section = format_citations_for_reporter(citations)

    q = state.get("clarified_research_topic") or state.get("research_topic", "")
    human = (
        f"## 1. 调查原始问题\n{q}\n\n"
        f"## 2. 调查计划制定思路\n{thought or '（无）'}\n\n"
        f"## 3. 分析结论\n{obs_text or '（无）'}\n\n"
        f"## 4. 可用参考来源\n{citations_section}\n"
    )
    sub = {**state, "workflow_type": wf, "messages": [HumanMessage(content=human)]}
    msgs = apply_prompt_template("reporter", sub, configurable)
    lim = get_llm_token_limit_by_type(AGENT_LLM_MAP.get("reporter", "basic"))
    comp = ContextManager(lim).compress_messages({"messages": msgs})
    llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "basic"))
    report = str(llm.invoke(comp.get("messages", msgs)).content or "").strip()
    return {
        **preserve_state_meta_fields(state),
        "final_report": report,
        "citations": citations,
        "replan_iterations": 0,
    }


def validate_plan_structure(plan: Any, workflow_type: str) -> tuple[bool, str]:
    """
    纯代码校验 Plan 结构是否符合声称的 workflow_type。
    不做语义判断，只检查结构特征。

    Returns:
        (is_valid, error_message)
    """
    if plan is None:
        return False, "Plan is None"

    steps = getattr(plan, "steps", [])
    research_steps = [s for s in steps if getattr(s, "step_type", "") == "research"]
    analysis_steps = [s for s in steps if getattr(s, "step_type", "") == "analysis"]

    # ── 通用校验 ──
    if not steps:
        return False, "Plan has no steps"

    if not analysis_steps:
        return False, "Plan missing analysis step"

    if getattr(analysis_steps[-1], "step_type", "") != "analysis":
        return False, "Last step is not analysis"

    if not research_steps:
        return False, "Plan has no research steps"

    # ── 工作流 B 校验：≥2 research step + 维度对齐 ──
    if workflow_type == "B":
        if len(research_steps) < 2:
            return False, (
                f"Workflow B requires ≥2 research steps, got {len(research_steps)}"
            )

    # ── 工作流 C 校验：≥2 不同角度的 research step ──
    elif workflow_type == "C":
        if len(research_steps) < 2:
            return False, (
                f"Workflow C requires ≥2 scan angles, got {len(research_steps)}"
            )

    # ── 工作流 D 校验：有否定条款检索 step ──
    elif workflow_type == "D":
        negative_keywords = ["例外", "禁止", "不适用", "不予", "除外", "限制"]
        has_negative_step = any(
            any(kw in getattr(s, "description", "") for kw in negative_keywords)
            for s in research_steps
        )
        if not has_negative_step:
            logger.warning("Workflow D: missing negative/exception clause search step")

        # missing_conditions 空值为 soft warning（用户可能提供了全部条件）
        missing = getattr(plan, "missing_conditions", [])
        if not missing:
            logger.warning(
                "Workflow D: missing_conditions is empty. "
                "This may be correct if user provided all conditions."
            )

    return True, ""


"""
Tool return pipeline: data models → tool formatters → hook → post-processing.

Unified data flow:
  Tool execution → ToolMessage(content=Markdown, artifact=structured) 
    → Hook captures artifact into cache
    → Searcher node end: format cache → Curator input + chunk_maps for State
"""



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 统一数据模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ToolChunk(BaseModel):
    """工具返回中的单个文档片段/段落。"""
    chunk_index: str
    chunk_content: str


class ToolDocumentReturn(BaseModel):
    """单个文档维度的工具返回数据。三种工具共用此结构。"""
    document_title: str
    document_url: str | None = None
    file_id: str | None = None
    description: str | None = None       # local_search 的 description 元数据

    chunks: list[ToolChunk] = Field(default_factory=list)

    # ── 截取元数据（仅 crawl/fetch 有值）──
    is_extracted: bool = False            # True = 执行了截取
    chunk_map: dict[str, str] | None = None  # 全量 chunk_index → content 映射


class ToolCallArtifact(BaseModel):
    """单次工具调用的结构化 artifact，存入 ToolMessage.artifact。"""
    tool_type: str                        # "local_search" | "crawl" | "fetch"
    documents: list[ToolDocumentReturn] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """Hook 缓存中的单条记录。"""
    call_index: int
    tool_name: str
    content_md: str                       # Searcher 可见的 Markdown
    artifact: ToolCallArtifact | None = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. 工具输出格式化器
#     每个工具调用这些函数来构造 (content, artifact) 元组
#     配合 @tool(response_format="content_and_artifact") 使用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 2a. local_search_tool ─────────────────────────

def format_local_search_return(
    raw_results: list[dict],
) -> tuple[str, dict]:
    """
    将 local_search 的原始结果格式化为 (content_md, artifact_dict)。

    raw_results 的期望结构（来自向量数据库）：
    [
        {
            "document_title": "信用卡分期业务管理办法",
            "document_url": "http://...",
            "file_id": "000001-000001",
            "description": "总行指引",
            "chunk_index": "3",
            "chunk_content": "白金卡账单分期手续费率...",
            "score": 0.85,
        },
        ...
    ]
    """
    # ── 按文档分组 ──
    doc_groups: dict[str, dict] = {}
    for r in raw_results:
        title = r.get("document_title", "未知文档")
        if title not in doc_groups:
            doc_groups[title] = {
                "document_title": title,
                "document_url": r.get("document_url"),
                "file_id": r.get("file_id"),
                "description": r.get("description"),
                "chunks": [],
            }
        doc_groups[title]["chunks"].append({
            "chunk_index": str(r.get("chunk_index", "")),
            "chunk_content": r.get("chunk_content", ""),
        })

    # ── 构建 content_md（Searcher 可见）──
    md_parts: list[str] = []
    artifact_docs: list[ToolDocumentReturn] = []

    for doc_idx, (title, doc_data) in enumerate(doc_groups.items(), 1):
        # Markdown header
        url_part = f" | url: {doc_data['document_url']}" if doc_data["document_url"] else ""
        fid_part = f" | 编号: {doc_data['file_id']}" if doc_data["file_id"] else ""
        desc_part = f" | {doc_data['description']}" if doc_data["description"] else ""
        md_parts.append(
            f"**文档 {doc_idx}** — 《{title}》{desc_part}{url_part}{fid_part}"
        )
        md_parts.append("")

        chunks = doc_data["chunks"]
        tool_chunks: list[ToolChunk] = []

        for chunk in chunks:
            idx = chunk["chunk_index"]
            content = chunk["chunk_content"].strip()
            md_parts.append(f"**[{idx}]**\n{content}")
            md_parts.append("")
            tool_chunks.append(ToolChunk(chunk_index=idx, chunk_content=content))

        md_parts.append("---")
        md_parts.append("")

        artifact_docs.append(ToolDocumentReturn(
            document_title=title,
            document_url=doc_data["document_url"],
            file_id=doc_data["file_id"],
            description=doc_data["description"],
            chunks=tool_chunks,
            is_extracted=False,
            chunk_map=None,
        ))

    content_md = "\n".join(md_parts).strip()
    artifact = ToolCallArtifact(tool_type="local_search", documents=artifact_docs)

    return content_md, artifact.model_dump()


# ── 2b. crawl_tool / fetch_tool ──────────────────

def format_crawl_fetch_return(
    extraction_result: ExtractionResult,
    tool_type: str = "crawl",
    document_url: str | None = None,
    file_id: str | None = None,
    description: str | None = None,
) -> tuple[str, dict]:
    """
    将 ExtractionResult 格式化为 (content_md, artifact_dict)。

    Args:
        extraction_result: 截取模块的输出。
        tool_type: "crawl" 或 "fetch"。
        document_url: 文档 URL。
        file_id: 知识库文档编号。
        description: 文档分类描述。
    """
    # ── content_md: Searcher 可见 ──
    if extraction_result.is_extracted:
        content_md = extraction_result.extracted_text_md
    else:
        content_md = extraction_result.full_text_md

    # ── artifact: 结构化数据 ──
    tool_chunks = [
        ToolChunk(chunk_index=ec.chunk_index, chunk_content=ec.chunk_content)
        for ec in extraction_result.extracted_chunks
    ]

    doc = ToolDocumentReturn(
        document_title=extraction_result.document_title,
        document_url=document_url,
        file_id=file_id,
        description=description,
        chunks=tool_chunks,
        is_extracted=extraction_result.is_extracted,
        chunk_map=extraction_result.chunk_map if extraction_result.chunk_map else None,
    )

    artifact = ToolCallArtifact(tool_type=tool_type, documents=[doc])

    return content_md, artifact.model_dump()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. Hook：拦截 ToolMessage，解析 artifact 存入缓存
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_tool_saver_hook(cache: list[ToolCallRecord]):
    """
    PreModelHook：每次模型调用前扫描 messages，
    将新出现的 ToolMessage 的 artifact 存入旁路缓存。

    执行顺序：先于 ContextManager（确保消息被压缩前已缓存）。
    """
    seen_ids: set[str] = set()

    def hook(messages: list) -> list:
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue

            msg_id = msg.id or str(id(msg))
            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)

            # 解析 artifact
            raw_artifact = getattr(msg, "artifact", None)
            parsed_artifact = None

            if raw_artifact is not None:
                try:
                    if isinstance(raw_artifact, dict):
                        parsed_artifact = ToolCallArtifact.model_validate(raw_artifact)
                    elif isinstance(raw_artifact, ToolCallArtifact):
                        parsed_artifact = raw_artifact
                except Exception as e:
                    logger.warning("Failed to parse tool artifact: %s", e)

            content_md = (
                msg.content if isinstance(msg.content, str)
                else json.dumps(msg.content, ensure_ascii=False)
            )

            cache.append(ToolCallRecord(
                call_index=len(cache) + 1,
                tool_name=getattr(msg, "name", None) or "unknown",
                content_md=content_md,
                artifact=parsed_artifact,
            ))

        return messages  # 原样返回，不改动消息

    return hook


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. Searcher 节点尾部处理
#     将缓存转化为 Curator 输入 + 提取 chunk_maps
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL_NAME_DISPLAY = {
    "local_search_tool": "语义检索",
    "crawl_tool": "全文获取(url)",
    "fetch_tool": "全文获取(文档名)",
}


def format_tool_cache_for_curator(
    cache: list[ToolCallRecord],
    step_title: str = "",
) -> str:
    """
    将缓存的全部工具调用格式化为 Curator 的"原始工具返回"输入。

    输出格式（Markdown）：
    === 第1次调用（local_search_tool / 语义检索）===
    [content_md from tool]

    === 第2次调用（crawl_tool / 全文获取）===
    [content_md from tool]
    """
    if not cache:
        return "（本步骤未执行任何工具调用）"

    parts: list[str] = []
    if step_title:
        parts.append(f"## {step_title} 的原始工具返回\n")

    for record in cache:
        display_name = TOOL_NAME_DISPLAY.get(record.tool_name, record.tool_name)
        parts.append(
            f"=== 第{record.call_index}次调用（{record.tool_name} / {display_name}）==="
        )
        parts.append("")
        parts.append(record.content_md)
        parts.append("")

    return "\n".join(parts).strip()


def extract_chunk_maps_from_cache(
    cache: list[ToolCallRecord],
) -> dict[str, dict[str, str]]:
    """
    从缓存中提取所有文档的 chunk_map。

    返回: {document_title: {chunk_index: chunk_content}}

    合并策略：
    - crawl/fetch 的 chunk_map (全量) 优先
    - local_search 的 chunks 作为补充
    - 同一文档多次出现时取并集
    """
    result: dict[str, dict[str, str]] = defaultdict(dict)

    # Pass 1: 收集 crawl/fetch 的完整 chunk_map（优先级高）
    for record in cache:
        if record.artifact is None:
            continue
        for doc in record.artifact.documents:
            if doc.chunk_map:
                # 完整 chunk_map 直接写入（覆盖 local_search 的部分数据）
                result[doc.document_title].update(doc.chunk_map)

    # Pass 2: 补充 local_search 的 chunks（不覆盖已有的）
    for record in cache:
        if record.artifact is None:
            continue
        for doc in record.artifact.documents:
            if doc.chunk_map:
                continue  # 已在 Pass 1 处理
            for chunk in doc.chunks:
                if chunk.chunk_index not in result[doc.document_title]:
                    result[doc.document_title][chunk.chunk_index] = chunk.chunk_content

    return dict(result)


def extract_document_metadata_from_cache(
    cache: list[ToolCallRecord],
) -> dict[str, dict]:
    """
    从缓存中提取所有文档的元数据（去重）。

    返回: {document_title: {"url": ..., "file_id": ..., "is_extracted": ...}}
    """
    metadata: dict[str, dict] = {}

    for record in cache:
        if record.artifact is None:
            continue
        for doc in record.artifact.documents:
            title = doc.document_title
            if title not in metadata:
                metadata[title] = {
                    "document_url": doc.document_url,
                    "file_id": doc.file_id,
                    "description": doc.description,
                    "is_extracted": doc.is_extracted,
                    "source_tools": [],
                }
            metadata[title]["source_tools"].append(record.tool_name)
            # crawl/fetch 的 is_extracted 优先（它比 local_search 更明确）
            if doc.is_extracted:
                metadata[title]["is_extracted"] = True

    return metadata


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. Searcher Node 集成示例
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def searcher_node(state: dict) -> dict:
    """
    Searcher 节点：执行 ReAct agent，缓存工具返回，格式化输出。

    返回更新 State 的字段：
    - searcher_results:         Curator 的原始工具返回 Markdown（追加当前步骤）
    - searcher_summaries:       Searcher 检索摘要文本（追加当前步骤）
    - document_chunk_maps:      全局文档 chunk_map 累积（合并更新）
    - document_metadata:        全局文档元数据累积（合并更新）
    """
    tool_returns_cache: list[ToolCallRecord] = []

    middleware = [
        # ① 先执行：缓存 ToolMessage artifact
        PreModelHookMiddleware(make_tool_saver_hook(tool_returns_cache)),
        # ② 后执行：消息压缩（可能删除旧 ToolMessage）
        PreModelHookMiddleware(partial(
            ContextManager(llm_token_limit, 3).compress_messages
        )),
    ]

    agent = create_agent(
        name="searcher",
        model=llm_model,
        tools=tools,
        middleware=middleware,
    )

    result = await agent.astream(state["messages"])

    # Curator 输入
    step_title = state.get("current_step_title", "")
    curator_tool_input = format_tool_cache_for_curator(tool_returns_cache, step_title)

    # Chunk maps（全局累积）
    new_chunk_maps = extract_chunk_maps_from_cache(tool_returns_cache)
    existing_maps = state.get("document_chunk_maps", {})
    for doc_title, cmap in new_chunk_maps.items():
        if doc_title in existing_maps:
            existing_maps[doc_title].update(cmap)
        else:
            existing_maps[doc_title] = cmap

    # Document metadata（全局累积）
    new_metadata = extract_document_metadata_from_cache(tool_returns_cache)
    existing_metadata = state.get("document_metadata", {})
    existing_metadata.update(new_metadata)

    # Citations（跨步骤 merge）
    new_citations = extract_citations_from_cache(tool_returns_cache)
    existing_citations = state.get("citations", [])
    merged_citations = merge_citations(existing_citations, new_citations)

    return {
        "messages": result["messages"],
        # Curator 输入：Markdown 格式的工具返回
        "searcher_results": [curator_tool_input],
        # Searcher 检索摘要（从 agent 最终文本输出提取）
        "searcher_summaries": [extract_final_text(result)],
        # 全局 chunk_map 累积
        "document_chunk_maps": existing_maps,
        # 全局文档元数据累积
        "document_metadata": existing_metadata,
        "citations": merged_citations,
    }