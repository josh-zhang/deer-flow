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
from functools import partial
from typing import Annotated, Any, Literal

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

from .types import State
from .utils import (
    build_clarified_topic_from_history,
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
    research_topic: Annotated[str, "中信银行信用卡客户需求原文及澄清摘要。"],
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
        file_id = str(c.get("file_id") or extra.get("file_id") or "").strip()
        lines.append(f"{i}. **{title}**")
        if url:
            lines.append(f"   - URL: `{url}`")
        if file_id:
            lines.append(f"   - 文档编号: `{file_id}`")
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


def background_investigation_node(state: State, config: RunnableConfig) -> dict:
    logger.info("background_investigator running")
    q = state.get("clarified_research_topic") or state.get("research_topic", "")
    retriever_tool = get_retriever_tool(state.get("resources", []))
    if not retriever_tool or not q:
        return {
            "background_investigation_results": json.dumps([], ensure_ascii=False),
            **preserve_state_meta_fields(state),
        }

    try:
        # local_search_tool 入参 schema: {"keywords": "..."}
        result = retriever_tool.invoke({"keywords": q})
        if isinstance(result, str):
            payload = [{"query": q, "summary": result}]
        else:
            payload = result
    except Exception as e:
        logger.warning("background retriever failed: %s", e)
        payload = [{"query": q, "error": str(e)}]

    return {
        "background_investigation_results": json.dumps(payload, ensure_ascii=False),
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
        messages += [
            {
                "role": "user",
                "content": "背景调查参考：\n" + str(state["background_investigation_results"]),
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
    sub = {**state, "messages": [HumanMessage(content=human)]}
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
    parsed = _parse_json_object(text)
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


async def reporter_node(state: State, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    plan = state.get("current_plan")
    wf = state.get("workflow_type", "A")
    thought = getattr(plan, "thought", "") if plan else ""
    observations = list(state.get("observations", []))
    obs_text = "\n\n---\n\n".join(observations)
    citations = [c for c in (state.get("citations") or []) if isinstance(c, dict)]
    cit = _format_citation_list_for_reporter(citations)
    q = state.get("clarified_research_topic") or state.get("research_topic", "")
    human = (
        f"## 1. 调查原始问题\n{q}\n\n"
        f"## 2. 调查计划制定思路\n{thought or '（无）'}\n\n"
        f"## 3. Analyst 的 observations\n{obs_text or '（无）'}\n\n"
        f"## 4. 可用参考来源\n{cit}\n"
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
