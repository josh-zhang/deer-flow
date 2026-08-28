import copy
import json
import logging
import os
import re
from typing import Annotated, Any, Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from src.agents import create_agent
from src.config.agents import get_agent_llm_type
from src.config.configuration import Configuration
from src.config.report_style import ReportStyle
from src.graph.curator_parser import (
    CuratorParseError,
    extract_citations_from_cache,
    merge_citations,
    parse_curator_output,
)
from src.graph.curator_views import (
    build_rule_splitter_view_with_resolved,
    extract_chunk_maps_from_cache,
    extract_document_metadata_from_cache,
    format_citations_for_reporter,
    format_tool_cache_for_curator,
    generate_analysis_view,
    resolve_all_evidence_chunks,
)
from src.graph.ontology import (
    extract_ontology_mapping,
    get_cp_analyzer_template_vars,
    render_arbitrator_bucketing_guide,
    render_cp_edge_constraints_for_assessor,
    render_cp_planner_edge_guide,
    render_element_class_mapping,
    render_planner_guardrail,
    render_skeleton_for_mapper,
)
from src.graph.planner_model import Plan, StepType
from src.graph.types import State
from src.graph.utils import (
    build_clarified_topic_from_history,
    is_user_message,
    prepare_reporter_input,
    reconstruct_clarification_history,
)
from src.llms.llm import get_llm_by_type, get_llm_token_limit_by_type
from src.prompts.template import apply_prompt_template, get_system_prompt_template
from src.rag.retriever import format_local_search_return
from src.tools import (
    crawl_tool,
    fetch_tool,
    get_retriever_tool,
)
from src.utils.context_manager import ContextManager, validate_message_content
from src.utils.json_utils import repair_json_output, sanitize_tool_response

logger = logging.getLogger(__name__)


@tool
def handoff_to_planner(
    research_topic: Annotated[str, "The topic of the research task to be handed off."],
    workflow_type: Annotated[str, "The workflow_type of the research task to be handed off."],
    workflow_confidence: Annotated[str, "The confidence of the workflow_type."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    logger.info(f"handoff_to_planner research_topic ({research_topic})")
    return


@tool
def handoff_after_clarification(
    research_topic: Annotated[
        str, "The clarified research topic based on all clarification rounds."
    ],
    workflow_type: Annotated[str, "The workflow_type of the research task to be handed off."],
    workflow_confidence: Annotated[str, "The confidence of the workflow_type."],
):
    """Handoff to planner after clarification rounds are complete. Pass all clarification history to planner for analysis."""
    logger.info(f"handoff_after_clarification research_topic ({research_topic})")
    return


def needs_clarification(state: dict) -> bool:
    """
    Check if clarification is needed based on current state.
    Centralized logic for determining when to continue clarification.
    """
    if not state.get("enable_clarification", False):
        return False
    clarification_rounds = state.get("clarification_rounds", 0)
    is_clarification_complete = state.get("is_clarification_complete", False)
    max_clarification_rounds = state.get("max_clarification_rounds", 2)
    # Need clarification if: enabled + has rounds + not complete + not exceeded max
    # Use <= because after asking the Nth question, we still need to wait for the Nth answer
    return (
        clarification_rounds > 0
        and not is_clarification_complete
        and clarification_rounds <= max_clarification_rounds
    )


def preserve_state_meta_fields(state: State) -> dict:
    """
    Extract meta/config fields that should be preserved across state transitions.
    These fields are critical for workflow continuity and should be explicitly
    included in all Command.update dicts to prevent them from reverting to defaults.

    Args:
        state: Current state object

    Returns:
        Dict of meta fields to preserve
    """
    return {
        "locale": state.get("locale", "zh-CN"),
        "research_topic": state.get("research_topic", ""),
        "original_topic": state.get("original_topic", ""),
        "clarified_research_topic": state.get("clarified_research_topic", ""),
        "clarification_history": state.get("clarification_history", []),
        "enable_clarification": state.get("enable_clarification", False),
        "max_clarification_rounds": state.get("max_clarification_rounds", 2),
        "clarification_rounds": state.get("clarification_rounds", 0),
        "resources": state.get("resources", []),
        "workflow_type": state.get("workflow_type", ""),
        "workflow_confidence": state.get("workflow_confidence", ""),
        "missing_conditions": state.get("missing_conditions", []),
        "pipeline_mode": state.get("pipeline_mode", ""),
    }


def validate_and_fix_plan(plan: dict) -> dict:
    """
    Validate and fix a plan to ensure it meets requirements.

    Args:
        plan: The plan dict to validate

    Returns:
        The validated/fixed plan dict
    """
    if not isinstance(plan, dict):
        return plan

    steps = plan.get("steps", [])

    # =========================================================================
    # SECTION 1: Repair missing step_type fields (Issue #650 fix)
    # =========================================================================
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue

        # Check if step_type is missing or empty
        if "step_type" not in step or not step.get("step_type"):
            # Infer step_type based on need_search value
            # Default to "analysis" for non-search steps (Issue #677: not all processing needs code)
            inferred_type = "research" if step.get("need_search", False) else "analysis"
            step["step_type"] = inferred_type
            logger.info(
                f"Repaired missing step_type for step {idx} ({step.get('title', 'Untitled')}): "
                f"inferred as '{inferred_type}' based on need_search={step.get('need_search', False)}"
            )

    check_last_step_is_analysis = False
    last_step = steps[-1] if steps else {}
    if isinstance(last_step, dict):
        step_type = last_step.get("step_type")
        if step_type == "analysis":
            check_last_step_is_analysis = True

    if not check_last_step_is_analysis:
        return {}
    return plan


async def _run_cp_analyzer(state: State, config: RunnableConfig, configurable: Configuration) -> dict:
    """CP 模式：通过调用 CP/analyzer.md 分析宣传文本。"""
    query = state.get("original_topic")
    logger.info(f"[background_investigation_node] search query: {query}")
    max_search_results = 10
    agent_type = "background_investigator"
    background_investigator = create_agent(
        agent_type,
        config,
        [],
    )
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"# 信用卡业务宣传文本\n\n{query}",
                name="user",
            )
        ]
    }
    sub: dict = {
        **state,
        **get_cp_analyzer_template_vars(),
    }
    response_content = await get_response(background_investigator, sub, agent_input, agent_type, configurable)
    if response_content is not None and response_content:
        logger.debug(f"background_investigator response_content {response_content}")
        retriever_tool = get_retriever_tool(
            max_search_results,
            configurable.report_style,
            state.get("resources", []),
        )
        documents = retriever_tool.retriever.query_relevant_documents(
            query, max_search_results, "", retriever_tool.resources
        )
        background_investigation_results, _ = format_local_search_return(documents)
        bi_result = {
            "background_investigation": background_investigation_results,
            "review_scope": response_content,
        }
        return {
            "background_investigation_results": json.dumps(
                bi_result,
                ensure_ascii=False,
            ),
            "kb_panorama": "",
            **preserve_state_meta_fields(state),
        }
    else:
        logger.warning(f"cp_background_investigator return empty result {response_content}")
        return Command(
            update=preserve_state_meta_fields(state),
            goto="__end__",
        )


async def _run_bi_bgi_and_mapper(
    state: State,
    config: RunnableConfig,
    configurable: Configuration,
) -> dict:
    """BI 模式: BGI 多角度 KB 探索 + Ontology Mapper 双阶段。"""
    query = state.get("original_topic")
    logger.info(f"[background_investigation_node] search query: {query}")
    max_search_results = 20
    logger.info(f"[background_investigation_node] Max search results: {max_search_results}")

    # Build tools list based on configuration
    # Add retriever tool if resources are available (always add, higher priority)
    retriever_tool = get_retriever_tool(
        max_search_results,
        configurable.report_style,
        state.get("resources", []),
    )
    if retriever_tool:
        logger.debug(f"[background_investigation_node] Adding retriever tool to tools list")
    else:
        return Command(
            update=preserve_state_meta_fields(state),
            goto="__end__",
        )
    tools = [retriever_tool]
    # logger.info(f"[background_investigation_node] Researcher tools count: {len(tools)}")
    logger.debug(
        f"[background_investigation_node] Researcher tools: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]}"
    )
    agent_type = "background_investigator"
    background_investigator = create_agent(
        agent_type,
        config,
        tools,
    )
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"# 用户问题\n\n{query}",
                name="user",
            )
        ]
    }
    kb_panorama = await get_response(background_investigator, state, agent_input, agent_type, configurable)
    if kb_panorama is not None and kb_panorama:
        logger.debug(f"background_investigator kb_panorama {kb_panorama}")
        # —— 本体映射 LLM 调用 ——
        mapper_human = f"## 用户问题\n{query}\n\n"
        if kb_panorama:
            mapper_human += f"{kb_panorama[:4000]}\n"
        mapper_sub = {
            **state,
            "ontology_skeleton": render_skeleton_for_mapper(),
            "messages": [HumanMessage(content=mapper_human, name="user")],
        }
        # if not configurable.enable_deep_thinking:
        #     llm_type = "lite_reasoning"
        # else:
        #     llm_type = "reasoning"
        llm_type = "reasoning"
        try:
            mapper_llm = get_llm_by_type(llm_type)
            mapper_content = str(
                (
                    await mapper_llm.ainvoke(
                        apply_prompt_template("ontology_mapper", mapper_sub, configurable)
                    )
                ).content
                or ""
            )
            mapping = extract_ontology_mapping(mapper_content)
        except Exception as e:
            logger.warning("ontology_mapper llm failed: %s", e)
            mapping = ""
        if mapping:
            payload_str = mapping
        else:
            logger.warning("ontology_mapper: no valid <ontology_mapping>, fallback to raw KB payload")
            payload_str = json.dumps(
                [{"query": query, "summary": kb_panorama}] if kb_panorama else [],
                ensure_ascii=False,
            )
        return {
            "background_investigation_results": payload_str,
            "kb_panorama": kb_panorama,
            **preserve_state_meta_fields(state),
        }
    else:
        logger.warning(f"background_investigator return empty result {kb_panorama}")
        return Command(
            update=preserve_state_meta_fields(state),
            goto="__end__",
        )


async def background_investigation_node(state: State, config: RunnableConfig):
    """Ontology Mapper + KB 概要双通道输出 (P1 改造 + 修正，2026-06-10)。
    产出两份独立结果:
    1. background_investigation_results: 本体映射 (Ontology Mapping) XML (结构护栏)
    2. kb_panorama: KB 轻量检索概要 (长尾信息参考 + KB 准确名称权威来源)
    Planner 同时消费两份输入: 本体边=必查维度硬性下限, KB 概要=长尾覆盖 + 名称校准。
    降级路径: LLM 映射失败时本体映射回退为空, KB 概要仍正常下发。
    """
    if not state.get("enable_background_investigation", True):
        return {}
    logger.info("background investigation node is running.")
    configurable = Configuration.from_runnable_config(config)
    report_style = configurable.report_style
    if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
        return await _run_bi_bgi_and_mapper(state, config, configurable)
    elif report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
        return await _run_cp_analyzer(state, config, configurable)
    else:
        return {}


def planner_node(state: State, config: RunnableConfig) -> Command[Literal["human_feedback", "reporter"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan with locale: %s", state.get("locale", "zh-CN"))
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state.get("plan_iterations", 0)
    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.max_plan_iterations:
        logger.warning(
            f"plan_iterations {plan_iterations} exceed max_plan_iterations {configurable.max_plan_iterations}"
        )
        return Command(
            update=preserve_state_meta_fields(state),
            goto="reporter",
        )
    logger.info(f"plan_iterations {plan_iterations} max_plan_iterations {configurable.max_plan_iterations}")
    workflow_type = state["workflow_type"]
    confidence = state.get("workflow_confidence", "high")
    is_override_retry = state.get("planner_override_occurred", False)
    is_validation_retry = state.get("structure_validation_retried", False)

    # —— Layer 0: 低置信度降级为静态全量 Prompt ——
    if confidence == "low" and not is_override_retry and not is_validation_retry:
        logger.info("Planner: Low confidence -> using static full prompt (all 4 workflows)")
        state["workflow_type"] = "A"
    prompt_key = "planner"
    report_style = configurable.report_style
    if report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
        state = {
            **state,
            "cp_planner_edge_guide": render_cp_planner_edge_guide(),
        }
    # For clarification feature: use the clarified research topic (complete history)
    if state.get("enable_clarification", False) and state.get("clarified_research_topic"):
        # Modify state to use clarified research topic instead of full conversation
        modified_state = copy.deepcopy(state)
        modified_state["messages"] = [
            {"role": "user", "content": state["clarified_research_topic"]}
        ]
        modified_state["research_topic"] = state["clarified_research_topic"]
        messages = apply_prompt_template(prompt_key, modified_state, configurable)
        logger.info(
            f"Clarification mode: Using clarified research topic: {state['clarified_research_topic']}"
        )
    else:
        # Normal mode: use full conversation history
        messages = apply_prompt_template(prompt_key, state, configurable)

    if state.get("enable_background_investigation") and state.get("background_investigation_results"):
        if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
            background_investigation_results_str = state["background_investigation_results"]
            if background_investigation_results_str:
                # P1 (2026-06-10) : 上游为 Ontology Mapper 时注入本体约束护栏 (Constraint Injection) :
                # 上游降级输出原始检索 payload 时回退旧版"背景调查参考"格式。
                guardrail = render_planner_guardrail(background_investigation_results_str)
                if guardrail:
                    messages += [
                        {
                            "role": "user",
                            "content": (
                                "# 本体映射结果 (Ontology Mapping) \n"
                                + extract_ontology_mapping(background_investigation_results_str)
                                + "\n\n"
                                + guardrail
                            ),
                        }
                    ]
            kb_panorama = state["kb_panorama"]
            if kb_panorama.strip():
                messages += [
                    {
                        "role": "user",
                        "content": kb_panorama,
                    }
                ]
        elif report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
            background_investigation_results = json.loads(state["background_investigation_results"])
            review_scope = background_investigation_results["review_scope"]
            messages += [
                {
                    "role": "user",
                    "content": f"# 审查范围分析结果\n\n{review_scope}",
                }
            ]
            background_investigation_results_str = background_investigation_results["background_investigation"]
            if background_investigation_results_str:
                messages += [
                    {
                        "role": "user",
                        "content": f"# 相关消保审查参考信息\n\n{background_investigation_results_str}",
                    }
                ]
        else:
            background_investigation_results_str = state["background_investigation_results"]
            if background_investigation_results_str:
                messages += [
                    {
                        "role": "user",
                        "content": f"# 相关业务参考信息\n\n{background_investigation_results_str}",
                    }
                ]

    if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
        replan_iterations = state.get("replan_iterations", 0)
        is_replan = state.get("replan", False)
        if replan_iterations > 0 and is_replan:
            replanning_reason = state.get("replan_reason")
            last_plan = state.get("last_plan")
            if replanning_reason and last_plan:
                messages += [
                    {
                        "role": "user",
                        "content": (
                            f"**注意：当前处于重规划阶段。**按照下方“重规划反馈”，对“上一轮计划”进行针对性调整，并重新制定调查计划。\n\n"
                            f"# 上一轮计划\n\n{last_plan}\n\n# 重规划反馈\n\n{replanning_reason}"
                        ),
                    }
                ]

    # Override 重试：附加说明
    if is_override_retry:
        messages += [
            {
                "role": "user",
                "content": (
                    f"你在上一次输出中将 workflow_type 修改为 {workflow_type}，"
                    "本次已使用该类型的策略模板。请基于此策略重新制定计划。"
                ),
            }
        ]
    # 结构校验重试：降级说明
    if is_validation_retry:
        messages += [
            {
                "role": "user",
                "content": (
                    "上一次生成的计划未通过结构校验，已降级为工作流 A（定点调查）。"
                    "请按照定点调查策略重新制定计划。"
                ),
            }
        ]

    llm_type = get_agent_llm_type("planner", configurable.enable_deep_thinking)
    llm = get_llm_by_type(llm_type)
    logger.debug(f"Planner inputs {messages}")
    response = llm.stream(messages)
    full_response = ""
    for chunk in response:
        full_response += chunk.content
    logger.debug(f"Planner full_response {full_response}")
    full_response = full_response.strip()
    logger.debug(f"Current state messages: {state['messages']}")

    # Clean the response first to handle markdown code blocks (```json, ```ts, etc.)
    cleaned_response = repair_json_output(full_response)
    # Validate explicitly that response content is valid JSON before proceeding to parse it
    if not cleaned_response.strip().startswith("{") and not cleaned_response.strip().startswith("["):
        logger.warning("Planner response does not appear to be valid JSON after cleanup")
        if plan_iterations > 0:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="reporter",
            )
        else:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__",
            )
    try:
        curr_plan = json.loads(cleaned_response)
        # Need to extract the plan from the full_response
        curr_plan_content = extract_plan_content(curr_plan)
        # Load the current_plan
        curr_plan = json.loads(repair_json_output(curr_plan_content))
        # full_response = f"```json\n{json.dumps(curr_plan, ensure_ascii=False)}\n```"
    except json.JSONDecodeError:
        logger.error("Planner response is not a valid JSON")
        if plan_iterations > 0:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="reporter",
            )
        else:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__",
            )
    logger.debug(f"Planner full_response final {full_response}")
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": full_response,
            "_plan_validator_needs_rerun": False,  # 重置路由信号
            **preserve_state_meta_fields(state),
        },
        goto="human_feedback",
    )


def _serialize_plan(plan) -> str:
    """将 Plan 对象序列化为 JSON 字符串，用于 Replan 时传递给 Planner。"""
    if plan is None:
        return "{}"
    if hasattr(plan, "model_dump"):
        return json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)
    if hasattr(plan, "__dict__"):
        return json.dumps(plan.__dict__, ensure_ascii=False, indent=2, default=str)
    return str(plan)


def extract_plan_content(plan_data: str | dict | Any) -> str:
    """
    Safely extract plan content from different types of plan data.

    Args:
        plan_data: The plan data which can be a string, AIMessage, or dict

    Returns:
        str: The plan content as a string (JSON string for dict inputs, or extracted/original string for other types)
    """
    if isinstance(plan_data, str):
        # If it's already a string, return as is
        return plan_data
    elif hasattr(plan_data, "content") and isinstance(plan_data.content, str):
        # If it's an AIMessage or similar object with a content attribute
        logger.debug(f"Extracting plan content from message object of type {type(plan_data).__name__}")
        return plan_data.content
    elif isinstance(plan_data, dict):
        # If it's already a dictionary, convert to JSON string
        # Need to check if it's dict with content field (AIMessage-like)
        if "content" in plan_data:
            if isinstance(plan_data["content"], str):
                logger.debug("Extracting plan content from dict with content field")
                return plan_data["content"]
            if isinstance(plan_data["content"], dict):
                logger.debug("Converting content field dict to JSON string")
                return json.dumps(plan_data["content"], ensure_ascii=False)
            if isinstance(plan_data["content"], list):
                # Handle multimodal message format where content is a list
                # Extract text content from the list structure
                logger.debug(
                    f"Extracting plan content from multimodal list format with {len(plan_data['content'])} elements"
                )
                for item in plan_data["content"]:
                    if isinstance(item, str) and item.strip():
                        # Return the first valid text content found
                        # We only take the first one because plan content should be a single JSON object
                        # Joining multiple text parts with newlines would produce invalid JSON
                        return item
                    elif isinstance(item, dict):
                        # Handle content block format like {"type": "text", "text": "..."}
                        if item.get("type") == "text" and "text" in item:
                            return item["text"]
                        elif "content" in item and isinstance(item["content"], str):
                            return item["content"]
                # No valid text content found - raise ValueError to trigger error handling
                # DO NOT use json.dumps() here as it would produce a JSON array that causes
                # Plan.model_validate() to fail with ValidationError (issue #845)
                raise ValueError(f"No valid text content found in multimodal list: {plan_data['content']}")
            else:
                logger.warning(
                    f"Unexpected type for 'content' field in plan_data dict: {type(plan_data['content']).__name__}, converting to string"
                )
                return str(plan_data["content"])
        else:
            logger.debug("Converting plan dictionary to JSON string")
            return json.dumps(plan_data, ensure_ascii=False)
    else:
        # For any other type, try to convert to string
        logger.warning(
            f"Unexpected plan data type {type(plan_data).__name__}, attempting to convert to string"
        )
        return str(plan_data)


def human_feedback_node(
    state: State, config: RunnableConfig
) -> Command[Literal["plan_validator"]]:
    current_plan = state.get("current_plan", "")
    # If the plan is accepted, run the following node
    plan_iterations = state.get("plan_iterations", 0)
    original_plan = current_plan
    try:
        # Safely extract plan content from different types (string, AIMessage, dict)
        # Repair the JSON output
        current_plan = repair_json_output(current_plan)
        # parse the plan to dict
        current_plan = json.loads(current_plan)
        current_plan_content = extract_plan_content(current_plan)
        # increment the plan iterations
        plan_iterations += 1
        # parse the plan
        new_plan = json.loads(repair_json_output(current_plan_content))
        new_plan = validate_and_fix_plan(new_plan)
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse plan: {str(e)}. Plan data type: {type(current_plan).__name__}")
        if isinstance(current_plan, dict) and "content" in original_plan:
            logger.warning(f"Plan appears to be an AIMessage object with content field")
        if plan_iterations > 1:
            # The plan_iterations is increased before this check
            return Command(
                update=preserve_state_meta_fields(state),
                goto="reporter",
            )
        else:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__",
            )
    if not new_plan:
        logger.error(f"new_plan last step is not analyst {new_plan}. Returning to planner for new plan.")
        return Command(
            update=preserve_state_meta_fields(state),
            goto="planner",
        )
    logger.debug(f"human_feedback plan {new_plan}")
    current_plan = Plan.model_validate(new_plan)
    # Build update dict with safe locale handling
    update_dict = {
        "current_plan": current_plan,
        # "last_plan": new_plan,
        "plan_iterations": plan_iterations,
        **preserve_state_meta_fields(state),
    }
    # Only override locale if new_plan provides a valid value, otherwise use preserved locale
    if new_plan.get("locale"):
        update_dict["locale"] = new_plan["locale"]
    return Command(
        update=update_dict,
        goto="plan_validator",
    )


def plan_validator_node(state: State) -> dict:
    """
    Layer 1 (Override 检测) + Layer 2 (结构校验)
    职责:
    1. 检查 Planner 输出的 workflow_type 是否与 state 一致 (Override 检测)
    2. 校验 Plan 结构是否符合声称的 workflow_type
    3. 通过 _plan_validator_needs_rerun 信号通知路由函数
    """
    plan = state.get("current_plan")
    if plan is None:
        logger.error("plan_validator: current_plan is None")
        return {"_plan_validator_needs_rerun": False}

    # planner 存入的是原始文本（full_response），此处统一解析为 Plan 对象供全下游使用
    if isinstance(plan, str):
        try:
            plan_data = json.loads(repair_json_output(plan))
            plan_content = extract_plan_content(plan_data)
            new_plan = json.loads(repair_json_output(plan_content))
            plan = Plan.model_validate(new_plan)
            for s in plan.steps:
                s.execution_res = None
            logger.info(f"plan_validator: parsed plan text -> Plan ({len(plan.steps)} steps, wf={plan.workflow_type})")
        except Exception as e:
            logger.error(f"plan_validator: failed to parse plan text to Plan: {e}")
            return {
                "_plan_validator_needs_rerun": False,
                "last_plan": plan,
                **preserve_state_meta_fields(state),
            }

    current_wf = state["workflow_type"]
    override_occurred = state.get("planner_override_occurred", False)
    validation_retried = state.get("structure_validation_retried", False)
    plan_str = _serialize_plan(plan)

    # -- Layer 1: Override Detection --
    planner_declared_type = getattr(plan, "workflow_type", "")
    logger.info(f"planner_declared_type: {planner_declared_type} current_wf {current_wf}")
    if planner_declared_type != current_wf and not override_occurred:
        logger.warning(
            f"Plan Validator [Layer 1]: Planner override detected '({current_wf}) -> {planner_declared_type}'. "
            f"Re-rendering with new type and rerunning Planner."
        )
        state["workflow_type"] = planner_declared_type
        state["workflow_confidence"] = "high"  # Planner 自行判断，视为高置信
        # Build update dict with safe locale handling
        return {
            "current_plan": plan,
            "planner_override_occurred": True,
            "_plan_validator_needs_rerun": True,
            "last_plan": plan_str,
            **preserve_state_meta_fields(state),
        }

    # -- Layer 2: Structure Validation --
    is_valid, error_msg = validate_plan_structure(plan, current_wf)
    if not is_valid and not validation_retried:
        logger.warning(
            f"Plan Validator [Layer 2]: Structure validation failed - {error_msg}. "
            f"Downgrading to workflow A and rerunning Planner."
        )
        state["workflow_type"] = "A"
        return {
            "current_plan": plan,
            "structure_validation_retried": True,
            "_plan_validator_needs_rerun": True,
            "last_plan": plan_str,
            **preserve_state_meta_fields(state),
        }

    if not is_valid and validation_retried:
        logger.warning(
            f"Plan Validator [Layer 2]: Structure still invalid after retry - {error_msg}. "
            f"Proceeding anyway."
        )

    # -- All Checks Passed --
    logger.info("Plan Validator: All checks passed.")
    return {
        "current_plan": plan,
        "_plan_validator_needs_rerun": False,
        "last_plan": plan_str,
        **preserve_state_meta_fields(state),
    }


def validate_plan_structure(plan: Any, workflow_type: str) -> tuple[bool, str]:
    """纯代码校验 Plan 结构是否符合声称的 workflow_type。不做语义判断，只检查结构指标。
    Returns: (is_valid, error_message)
    """

    def _extract_targets_from_description(description: str) -> list[str]:
        """
        从 Step description 中按分号拆分出标的列表。
        示例: "'中信银行白金卡'的年费标准;年费减免条件;年费收取时间" -> ['年费标准', '年费减免条件', '年费收取时间']
        """
        if not description:
            return []
        # 尝试用中文分号分割
        parts = re.split(r"[;；]", description)
        # 清理:去除首段 (通常是实体+业务名,不是标的)
        targets = []
        for part in parts:
            part = part.strip().rstrip(".")
            if part:
                targets.append(part)
        return targets

    if plan is None:
        return False, "Plan is None"

    steps = getattr(plan, "steps", [])
    research_steps = [s for s in steps if getattr(s, "step_type", "") == "research"]
    analysis_steps = [s for s in steps if getattr(s, "step_type", "") == "analysis"]

    # -- 通用校验 --
    if not steps:
        return False, "Plan has no steps"
    if not analysis_steps:
        return False, "Plan missing analysis step"
    if getattr(analysis_steps[-1], "step_type", "") != "analysis":
        return False, "Last step is not analysis"
    if not research_steps:
        return False, "Plan has no research steps"

    # -- 工作流 B 校验: ≥2 research step, 维度对齐
    if workflow_type == "B":
        if len(research_steps) < 2:
            return False, f"Workflow B requires ≥2 research steps, got {len(research_steps)}"
        # 检查维度清单是否结构标一致 (标的数量相同)
        target_counts = []
        for s in research_steps:
            desc = getattr(s, "description", "")
            targets = _extract_targets_from_description(desc)
            target_counts.append(len(targets))
        if len(set(target_counts)) > 1:
            logger.warning(
                f"Workflow B: dimension counts not aligned across research steps: {target_counts}"
            )

    # -- 工作流 C 校验: ≥2 不同角度的 research step --
    elif workflow_type == "C":
        if len(research_steps) < 2:
            return False, f"Workflow C requires ≥2 scan angles, got {len(research_steps)}"

    # -- 工作流 D 校验: 有否定条款检索 step --
    elif workflow_type == "D":
        negative_keywords = ["例外", "禁止", "不适用", "不予", "除外", "限制"]
        has_negative_step = any(
            any(kw in getattr(s, "description", "") for kw in negative_keywords)
            for s in research_steps
        )
        if not has_negative_step:
            logger.warning("Workflow D: missing negative/exception clause search step")
        # missing_conditions 空值为 soft warning (用户可能提供了全部条件)
        missing = getattr(plan, "missing_conditions", [])
        if not missing:
            logger.warning(
                "Workflow D: missing_conditions is empty. "
                "This may be correct if user provided all conditions."
            )

    return True, ""


def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "background_investigator", "coordinator", "__end__"]]:
    """Coordinator node that communicate with customers and handle clarification."""
    logger.info("Coordinator talking.")
    configurable = Configuration.from_runnable_config(config)
    report_style = configurable.report_style
    if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
        pipeline_mode = "bi"
    elif report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
        pipeline_mode = "cp"
    else:
        pipeline_mode = "unkonw"

    # Check if clarification is enabled
    enable_clarification = state.get("enable_clarification", False)
    initial_topic = state.get("research_topic", "")
    original_topic = state.get("original_topic", "")
    # clarified topic initial_topic
    workflow_type = state.get("workflow_type", "")
    workflow_confidence = state.get("workflow_confidence", "")
    logger.info(
        f"Coordinator talking. enable_clarification {enable_clarification}\n initial_topic {initial_topic}\noriginal_topic {original_topic}"
    )

    # =================================================================
    # BRANCH 1: Clarification DISABLED (Legacy Mode)
    # =================================================================
    if not enable_clarification:
        # Use normal prompt with explicit instruction to skip clarification
        sub = {
            **state,
            "enable_clarification": False,
        }
        messages = apply_prompt_template("coordinator", sub, configurable)
        # Bind both handoff_to_planner and direct_response tools
        tools = [handoff_to_planner]
        llm_type = get_agent_llm_type("coordinator", configurable.enable_deep_thinking)
        response = (
            get_llm_by_type(llm_type)
            .bind_tools(tools)
            .invoke(messages)
        )
        goto = "__end__"
        locale = state.get("locale", "zh-CN")
        # logger.info(f"Coordinator locale: {locale}")
        research_topic = state.get("research_topic", "")

        # Process tool calls for legacy mode
        if response.tool_calls:
            try:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    if tool_name == "handoff_to_planner":
                        logger.info("Handing off to planner")
                        goto = "planner"
                        # Extract research_topic if provided
                        if tool_args.get("research_topic"):
                            research_topic = tool_args.get("research_topic")
                        if tool_args.get("workflow_type"):
                            workflow_type = tool_args.get("workflow_type")
                        if tool_args.get("workflow_confidence"):
                            workflow_confidence = tool_args.get("workflow_confidence")
                        break
            except Exception as e:
                logger.error(f"Error processing tool calls: {e}")
                goto = "planner"
        # Do not return early - let code flow to unified return logic below
        # Set clarification variables for legacy mode
        clarification_rounds = 0
        clarification_history = []
        clarified_topic = research_topic

    # ------------------------------------------------
    # BRANCH 2: Clarification ENABLED (New Feature)
    # ------------------------------------------------
    else:
        # Load clarification state
        clarification_rounds = state.get("clarification_rounds", 0)
        clarification_history = list(state.get("clarification_history", [])) or []
        clarification_history = [item for item in clarification_history if item]
        max_clarification_rounds = state.get("max_clarification_rounds", 2)

        # Prepare the messages for the coordinator
        state_messages = list(state.get("messages", []))
        sub = {
            **state,
            "enable_clarification": True,
        }
        messages = apply_prompt_template("coordinator", sub, configurable)
        clarification_history = reconstruct_clarification_history(
            state_messages, clarification_history, initial_topic
        )
        clarified_topic, clarification_history = build_clarified_topic_from_history(
            clarification_history
        )
        logger.info("Clarification history rebuilt: %s", clarification_history)
        if clarification_history:
            initial_topic = clarification_history[0]
            latest_user_content = clarification_history[-1]
        else:
            latest_user_content = ""

        # Add clarification status for first round
        if clarification_rounds == 0:
            messages.append(
                {
                    "role": "user",
                    "content": "“澄清性提问”模式已激活，严格按照指令中的‘澄清性提问要求’进行提问。",
                }
            )

        current_response = latest_user_content or "No response"
        logger.info(
            "Clarification round %s/%s | topic: %s | current user response: %s",
            clarification_rounds,
            max_clarification_rounds,
            clarified_topic or initial_topic,
            current_response,
        )
        clarification_context = f"""继续澄清性提问 (回合 {clarification_rounds}/{max_clarification_rounds}): 用户上一轮回复: {current_response}
对缺失的维度继续澄清性提问，不要重复问题或开启新话题。"""
        messages.append({"role": "user", "content": clarification_context})

        # Bind both clarification tools - let LLM choose the appropriate one
        tools = [handoff_to_planner, handoff_after_clarification]

        # Check if we've already reached max rounds
        if clarification_rounds >= max_clarification_rounds:
            # Max rounds reached - force handoff by adding system instruction
            logger.warning(
                f"Max clarification rounds ({max_clarification_rounds}) reached. Forcing handoff to planner. Using prepared clarified topic: {clarified_topic}"
            )
            # Add system instruction to force handoff - let LLM choose the right tool
            messages.append(
                {
                    "role": "user",
                    "content": f"达到最大澄清回合数。你必须调用 handoff_after_clarification (not handoff_to_planner) and research_topic='{clarified_topic}'。不要继续提问了。",
                }
            )

        llm_type = get_agent_llm_type("coordinator", configurable.enable_deep_thinking)
        response = (
            get_llm_by_type(llm_type)
            .bind_tools(tools)
            .invoke(messages)
        )
        logger.debug(f"Current state messages: {state['messages']}")

        # Initialize response processing variables
        goto = "__end__"
        locale = state.get("locale", "zh-CN")
        research_topic = (
            clarification_history[0] if clarification_history else state.get("research_topic", "")
        )
        if not clarified_topic:
            clarified_topic = research_topic

        # --- Process LLM response ---
        # No tool calls - LLM is asking a clarifying question
        if not response.tool_calls and response.content:
            # Check if we've reached max rounds - if so, force handoff to planner
            if clarification_rounds >= max_clarification_rounds:
                logger.warning(
                    f"Max clarification rounds ({max_clarification_rounds}) reached. "
                    "LLM didn't call handoff tool, forcing handoff to planner."
                )
                goto = "planner"
                # Continue to final section instead of early return
            else:
                # Continue clarification process
                clarification_rounds += 1
                # Do NOT add LLM response to clarification_history - only user responses
                logger.info(
                    f"Clarification response {clarification_rounds}/{max_clarification_rounds}: {response.content}"
                )
                # Append coordinator's question to messages
                updated_messages = list(state_messages)
                if response.content:
                    updated_messages.append(
                        HumanMessage(content=response.content, name="coordinator")
                    )
                return Command(
                    update={
                        "messages": updated_messages,
                        "locale": locale,
                        "research_topic": research_topic,
                        "resources": configurable.resources,
                        "clarification_rounds": clarification_rounds,
                        "clarification_history": clarification_history,
                        "clarified_research_topic": clarified_topic,
                        "is_clarification_complete": False,
                        "goto": goto,
                        "citations": state.get("citations", []),
                        "pipeline_mode": pipeline_mode,
                        "__interrupt__": [("coordinator", response.content)],
                    },
                    goto=goto,
                )
        else:
            # LLM called a tool (handoff) or has no content - clarification complete
            if response.tool_calls:
                logger.info(
                    f"Clarification completed after {clarification_rounds} rounds. LLM called handoff tool."
                )
            else:
                logger.warning("LLM response has no content and no tool calls.")
            # goto will be set in the final section based on tool calls

    # ------------------------------------------------
    # Final: Build and return Command
    # ------------------------------------------------
    messages = list(state.get("messages", [])) or []
    if response.content:
        messages.append(HumanMessage(content=response.content, name="coordinator"))

    # Process tool calls for BOTH branches (legacy and clarification)
    if response.tool_calls:
        try:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                if tool_name in ["handoff_to_planner", "handoff_after_clarification"]:
                    logger.info("Handing off to planner")
                    goto = "planner"
                    if not enable_clarification and tool_args.get("research_topic"):
                        research_topic = tool_args["research_topic"]
                    if tool_args.get("workflow_type"):
                        workflow_type = tool_args.get("workflow_type")
                    if tool_args.get("workflow_confidence"):
                        workflow_confidence = tool_args.get("workflow_confidence")
                    if enable_clarification:
                        logger.info(
                            "Using prepared clarified topic: %s",
                            clarified_topic or research_topic,
                        )
                    else:
                        logger.info("Using research topic for handoff: %s", research_topic)
                    break
        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")
            goto = "planner"
    else:
        # No tool calls detected
        if enable_clarification:
            # BRANCH 2: Fallback to planner to ensure research proceeds
            logger.warning(
                "LLM didn't call any tools. This may indicate tool calling issues with the model. "
                "Falling back to planner to ensure research proceeds."
            )
            logger.debug(f"Coordinator response content: {response.content}")
            logger.debug(f"Coordinator response object: {response}")
            goto = "planner"
        else:
            # BRANCH 1: No tool calls means end workflow gracefully (e.g., greeting handled)
            logger.info("No tool calls in legacy mode - ending workflow gracefully")

    # Apply background_investigation routing if enabled (unified logic)
    if goto == "planner" and state.get("enable_background_investigation"):
        goto = "background_investigator"

    # Set default values for state variables (in case they're not defined in legacy mode)
    if not enable_clarification:
        clarification_rounds = 0
        clarification_history = []
    clarified_research_topic_value = clarified_topic or research_topic
    # clarified_research_topic: Complete clarified topic with all clarification rounds
    return Command(
        update={
            "messages": messages,
            "locale": locale,
            "research_topic": research_topic,
            "workflow_type": workflow_type,
            "workflow_confidence": workflow_confidence,
            "clarified_research_topic": clarified_research_topic_value,
            "resources": configurable.resources,
            "clarification_rounds": clarification_rounds,
            "clarification_history": clarification_history,
            "is_clarification_complete": goto != "coordinator",
            "goto": goto,
            "citations": state.get("citations", []),
            "pipeline_mode": pipeline_mode,
        },
        goto=goto,
    )


def reporter_node(state: State, config: RunnableConfig):
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    configurable = Configuration.from_runnable_config(config)
    current_plan = state.get("current_plan")
    original_topic = state.get("original_topic")
    report_style = configurable.report_style
    if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
        original_topic_title = "调查原始问题"
    elif report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
        original_topic_title = "信用卡业务宣传文本"
    else:
        original_topic_title = "调查原始问题"

    input_ = {
        "messages": [
            HumanMessage(
                f"## {original_topic_title}\n\n{original_topic}\n\n"
                f"## 调查计划制定思路\n\n{current_plan.thought}\n\n"
            )
        ]
    }
    modified_state = copy.deepcopy(state)
    modified_state["messages"] = input_["messages"]
    invoke_messages = apply_prompt_template("reporter", modified_state, configurable)
    observations = state.get("observations", [])
    observation_messages = []
    for observation in observations:
        # if observation.startswith("Below are some observations for the analyst task:"):
        if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
            # 业务分析专家(Analyst) 的分析结果 (observations)
            observation_content = (
                f"# Observations (分析结果) for the analyst (业务分析专家)\n\n{observation}"
            )
        elif report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
            observation_content = f"## 审查结果\n\n{observation}"
        else:
            observation_content = f"Below are some observations for the analyst task:\n\n{observation}"
        observation_messages.append(
            HumanMessage(
                content=observation_content,
                # content=observation,
                name="observation",
            )
        )

    # Context compression
    llm_type = get_agent_llm_type("reporter", configurable.enable_deep_thinking)
    llm_token_limit = get_llm_token_limit_by_type(llm_type)
    compressed_state = ContextManager(llm_token_limit).compress_messages(
        {"messages": observation_messages}
    )
    invoke_messages += compressed_state.get("messages", [])

    # Append citations AFTER observations so they are closest to the LLM's
    # generation point. This reduces the chance of the model "forgetting"
    # real URLs and fabricating plausible-looking ones instead.
    # If we have collected citations, provide them to the reporter
    citation_list = ""
    citations = state.get("citations", [])
    if citations:
        citation_list = format_citations_for_reporter(citations)
        logger.info(f"Providing {len(citations)} collected citations to reporter")
    if citation_list:
        invoke_messages.append(
            HumanMessage(
                content=citation_list,
            )
        )
    logger.debug(f"reporter Current invoke messages: {invoke_messages}")
    response = get_llm_by_type(llm_type).invoke(invoke_messages)
    response_content = response.content
    logger.debug(f"reporter response: {response_content}")
    return {
        "final_report": response_content,
        "citations": citations,  # Pass citations through to final state
    }


async def _handle_recursion_limit_fallback(
    messages: list,
    agent_name: str,
    llm_type,
    state: State,
) -> list:
    """Handle GraphRecursionError with graceful fallback using LLM summary.
    When the agent hits the recursion limit, this function generates a final output
    using only the observations already gathered, without calling any tools.

    Args:
        messages: Messages accumulated during agent execution before hitting limit
        agent_name: Name of the agent that hit the limit
        state: Current workflow state

    Returns:
        list: Messages including the accumulated messages plus the fallback summary

    Raises:
        Exception: If the fallback LLM call fails
    """
    logger.warning(
        f"Recursion limit reached for {agent_name} agent."
        f"Attempting graceful fallback with {len(messages)} accumulated messages."
    )
    if len(messages) == 0:
        return messages

    # cleared_messages = messages.copy()
    # while len(cleared_messages) > 0 and is_system_message(cleared_messages[-1]):
    #     cleared_messages = cleared_messages[:-1]
    cleared_messages = copy.deepcopy(messages)
    while len(cleared_messages) > 0 and cleared_messages[-1].type == "system":
        cleared_messages = cleared_messages[:-1]

    # Prepare state for prompt template
    fallback_state = {
        "locale": state.get("locale", "zh-CN"),
    }
    # Apply the recursion_fallback prompt template
    # system_prompt = get_system_prompt_template(agent_name, fallback_state, None)
    limit_prompt = get_system_prompt_template("recursion_fallback", fallback_state, None)
    # cleared_messages[0] = SystemMessage(content=f"{cleared_messages[0].content}\n\n{limit_prompt}")
    # cleared_messages.insert(1, SystemMessage(content=limit_prompt))
    cleared_messages.append(
        HumanMessage(
            content=limit_prompt,
            name="user",
        )
    )
    fallback_messages = cleared_messages

    # Get the LLM without tools (strip all tools from binding)
    fallback_llm = get_llm_by_type(llm_type)
    logger.debug(f"fallback inputs {fallback_messages}")

    # Call the LLM with the updated messages
    fallback_response = fallback_llm.invoke(fallback_messages)
    fallback_content = fallback_response.content
    logger.info(
        f"Graceful fallback succeeded for {agent_name} agent."
        f"Generated summary of {len(fallback_content)} characters."
    )

    # Sanitize response
    fallback_content = sanitize_tool_response(str(fallback_content))
    # Update the step with the fallback result
    # current_step.execution_res = fallback_content
    # Return the accumulated messages plus the fallback response
    result_messages = list(cleared_messages)
    result_messages.append(AIMessage(content=fallback_content, name=agent_name))
    return result_messages


async def _execute_agent_step(
    state: State,
    agent,
    agent_name: str,
    config: RunnableConfig = None,
    tool_returns_cache: list = None,
):
    """Helper function to execute a step using the specified agent."""
    logger.debug(f"[_execute_agent_step] Starting execution for agent: {agent_name}")
    current_plan = state.get("current_plan")
    current_plan_cached = state.get("current_plan_cached")
    plan_title = current_plan.title
    observations = state.get("observations", [])
    search_results = state.get("search_results", [])
    curator_rule_splitter_views = state.get("curator_rule_splitter_views", [])
    workflow_type = state.get("workflow_type", "")
    workflow_type_str = "A 定点调查"
    if workflow_type == "B":
        workflow_type_str = "B 并行对比"
    elif workflow_type == "C":
        workflow_type_str = "C 扫描穷举"
    elif workflow_type == "D":
        workflow_type_str = "D 条件推理"

    configurable = Configuration.from_runnable_config(config)
    report_style = configurable.report_style
    logger.info(
        f"[_execute_agent_step] Plan title: {plan_title}, observations count: {len(observations)}, workflow_type: {workflow_type}, workflow_type_str: {workflow_type_str}"
    )

    # Find the first unexecuted step
    current_step = None
    completed_steps = []
    for idx, step in enumerate(current_plan.steps):
        if not step.execution_res:
            current_step = step
            logger.debug(f"[_execute_agent_step] Found unexecuted step at index {idx}: {step.title}")
            break
        else:
            completed_steps.append(step)

    if not current_step:
        logger.warning(f"[_execute_agent_step] No unexecuted step found in {len(current_plan.steps)} total steps")
        return Command(
            update=preserve_state_meta_fields(state),
            goto="reporter",
        )

    is_searcher = agent_name == "researcher"
    is_curator = agent_name == "curator"
    is_analyst = agent_name == "analyst"
    logger.info(f"[_execute_agent_step] Executing step: {current_step.title}, agent: {agent_name}")
    logger.debug(f"[_execute_agent_step] Completed steps so far: {len(completed_steps)}")

    # Format completed steps information
    if is_curator:
        searcher_annotations, raw_tool_returns = search_results[-1] if search_results else ("", "")
        completed_steps_info = (
            f"# 检索结果摘要\n\n{searcher_annotations}\n\n# 原始工具返回\n\n{raw_tool_returns}\n\n"
        )
        if not search_results:
            logger.warning(
                "[_execute_agent_step] curator: search_results is empty (researcher produced no annotation); continuing with empty evidence view"
            )
    elif is_searcher:
        if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
            step_index = 1
            completed_steps_info = ""
            for searcher_annotations, _ in search_results:
                if searcher_annotations:
                    completed_steps_info += (
                        f"# 已完成步骤{str(step_index)}-检索\n\n{searcher_annotations}\n\n"
                    )
                    step_index += 1
        else:
            completed_steps_info = ""
    else:  # is_analyst
        if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
            missing_conditions = state.get("missing_conditions", "无。")
            step_index = 1
            if workflow_type == "D":
                completed_steps_info = f"# 调查原始问题未提供关键条件列表\n\n{missing_conditions}\n\n"
            else:
                completed_steps_info = ""

            completed_cached_steps = []
            if current_plan_cached:
                for idx, step in enumerate(current_plan_cached.steps):
                    if step.execution_res and step.step_type == StepType.RESEARCH:
                        completed_cached_steps.append(step)
            completed_steps += completed_cached_steps

            for step in completed_steps:
                if (
                    "researcher agent error" in step.execution_res.lower()
                    and step.execution_res.startswith("[ERROR]")
                ):
                    logger.error(f"[_execute_agent_step] Completed Step skip agent error")
                    completed_steps_info += f"因为该步骤的Agent执行任务失败，没有获取任何相关信息。\n\n"
                    continue
                completed_steps_info += f"# 已完成步骤 {step_index}: {step.title}\n\n"
                completed_steps_info += f"## 步骤背景\n\n{step.background}\n\n"
                completed_steps_info += f"## 信息质量评估报告\n{step.execution_res}\n\n"
                step_index += 1

            arbitration_result = state.get("arbitration_result", "")
            if arbitration_result:
                completed_steps_info += f"# 已完成步骤 {step_index}: 矛盾信息仲裁\n\n"
                completed_steps_info += f"{arbitration_result}\n\n"
                step_index += 1
        elif report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
            completed_steps_info = f"# 逐审查点完整记录\n\n"
            # 逐审查点完整记录
            all_atomic_rules = state.get("atomic_rules", [])
            for atomic_rules in all_atomic_rules:
                completed_steps_info += f"{atomic_rules}\n\n"
        else:
            completed_steps_info = ""

    original_topic = state.get("original_topic")
    if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
        original_topic_title = "调查原始问题"
        workflow_prompt = f"# 当前工作流类型\n{workflow_type_str}\n\n"
    elif report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
        original_topic_title = "信用卡业务宣传文本"
        workflow_prompt = ""
    else:
        original_topic_title = "调查原始问题"
        workflow_prompt = ""

    if is_searcher:
        agent_chn_name = "检索"
    elif is_curator:
        agent_chn_name = "评估"
    else:
        agent_chn_name = "分析"

    agent_input = {"messages": []}
    if is_curator:
        agent_input["messages"].append(
            HumanMessage(
                content=(
                    f"# {original_topic_title}\n\n{original_topic}\n\n{completed_steps_info}"
                    f"# 当前步骤 - {agent_chn_name}\n\n"
                    f"## {agent_chn_name}标题\n\n{current_step.title}\n\n"
                    f"## {agent_chn_name}背景\n\n{current_step.background}\n\n"
                    f"## {agent_chn_name}内容\n\n{current_step.description}\n\n"
                )
            )
        )
    else:
        agent_input["messages"].append(
            HumanMessage(
                content=(
                    f"# {original_topic_title}\n\n{original_topic}\n\n{workflow_prompt}{completed_steps_info}"
                    f"# 当前步骤 - {agent_chn_name}\n\n"
                    f"## {agent_chn_name}标题\n\n{current_step.title}\n\n"
                    f"## {agent_chn_name}背景\n\n{current_step.background}\n\n"
                    f"## {agent_chn_name}内容\n\n{current_step.description}\n\n"
                )
            )
        )

    replan_iterations = state.get("replan_iterations", 0)
    MAX_ITERATIONS = 2
    if (
        report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value
        and replan_iterations >= MAX_ITERATIONS
    ):
        agent_input["messages"].append(HumanMessage(content="注意：当前规划迭代次数已经达到最大值"))

    # Invoke the agent
    default_recursion_limit = 25
    try:
        env_value_str = os.getenv("NODE_RECURSION_LIMIT", str(default_recursion_limit))
        parsed_limit = int(env_value_str)
        if parsed_limit > 0:
            recursion_limit = parsed_limit
            logger.info(f"Recursion limit set to: {recursion_limit}")
        else:
            logger.warning(
                f"NODE_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
                f"Using default value {default_recursion_limit}."
            )
            recursion_limit = default_recursion_limit
    except ValueError:
        raw_env_value = os.getenv("NODE_RECURSION_LIMIT")
        logger.warning(
            f"Invalid NODE_RECURSION_LIMIT value: '{raw_env_value}'. "
            f"Using default value {default_recursion_limit}."
        )
        recursion_limit = default_recursion_limit

    logger.debug(f"Agent input: {agent_input}")
    # Validate message content before invoking agent
    try:
        validated_messages = validate_message_content(agent_input["messages"])
        agent_input["messages"] = validated_messages
    except Exception as validation_error:
        logger.error(f"Error validating agent input messages: {validation_error}")

    modified_state = copy.deepcopy(state)
    modified_state["messages"] = agent_input["messages"]
    accumulated_messages = []
    try:
        # Use astream (async) from the start to capture messages in real-time
        # This allows us to retrieve accumulated messages even if recursion limit is hit
        # NOTE: astream is required for MCP tools which only support async invocation
        async for chunk in agent.astream(
            input={"messages": apply_prompt_template(agent_name, modified_state, configurable)},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            if isinstance(chunk, dict) and "messages" in chunk:
                accumulated_messages = chunk["messages"]
        # If we get here, execution completed successfully
        llm_result = {"messages": accumulated_messages}
    except GraphRecursionError:
        # Check if recursion fallback is enabled
        configurable = Configuration.from_runnable_config(config) if config else Configuration()
        if configurable.enable_recursion_fallback:
            try:
                # Call fallback with accumulated messages (function returns list of messages)
                llm_type = get_agent_llm_type(agent_name, configurable.enable_deep_thinking)
                fallback_response_messages = await _handle_recursion_limit_fallback(
                    messages=accumulated_messages,
                    agent_name=agent_name,
                    llm_type=llm_type,
                    state=state,
                )
                # Create result dict so the code can continue normally from line 1178
                llm_result = {"messages": fallback_response_messages}
            except Exception as fallback_error:
                # If fallback fails, log and fall through to standard error handling
                logger.error(
                    f"Recursion fallback failed for {agent_name} agent: {fallback_error}. "
                    "Falling back to standard error handling."
                )
                raise
        else:
            # Fallback disabled, let error propagate to standard handler
            logger.info(
                f"Recursion limit reached but graceful fallback is disabled. Using standard error handling."
            )
            raise
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = f"Error executing {agent_name} agent for step '{current_step.title}': {str(e)}"
        logger.exception(error_message)
        logger.error(f"Full traceback:\n{error_traceback}")
        # Enhanced error diagnostics for content-related errors
        if "Field required" in str(e) and "content" in str(e):
            logger.error(f"Message content validation error detected")
            for i, msg in enumerate(agent_input.get("messages", [])):
                logger.error(
                    f"Message {i}: type={type(msg).__name__}, "
                    f"has_content={hasattr(msg, 'content')}, "
                    f"content_type={type(msg.content).__name__ if hasattr(msg, 'content') else 'N/A'}, "
                    f"content_len={len(str(msg.content)) if hasattr(msg, 'content') and msg.content else 0}"
                )
        return Command(
            update=preserve_state_meta_fields(state),
            goto="__end__",
        )

    # Include all messages from agent result to preserve intermediate tool calls/results
    # This ensures multiple web_search calls all appear in the stream, not just the final result
    agent_messages = llm_result["messages"]
    response_content = agent_messages[-1].content
    logger.info(
        f"Step '{current_step.title}' execution completed by {agent_name}\n"
        f"{agent_name.capitalize()} returned {len(agent_messages)} messages. "
        f"Message types: {[type(msg).__name__ for msg in agent_messages]}"
    )

    if is_analyst:
        parsed_analyst_res = _parse_analyst_output(response_content)
        # Validate explicitly that response content is valid JSON before proceeding to parse it
        if parsed_analyst_res is None:
            logger.error(
                f"{agent_name} response does not appear to be valid XML or JSON after cleanup\n\n{response_content}"
            )
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__",
            )
        replanning_needed = bool(parsed_analyst_res and parsed_analyst_res.get("replan"))
        current_step.execution_res = response_content
        if (
            report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value
            and replanning_needed
            and replan_iterations <= MAX_ITERATIONS
        ):
            replanning_reason = parsed_analyst_res.get("replan_reason", "")
            logger.info("Analyst requested replanning due to insufficient evidence.")
            replan_iterations += 1
            messages = list(state.get("messages", [])) or []
            skip_idx = 0
            for idx, m in enumerate(messages):
                if not is_user_message(m):
                    break
                skip_idx = idx
            skip_idx += 1
            delete_messages = [RemoveMessage(id=m.id) for m in messages[skip_idx:]]
            return Command(
                update={
                    **preserve_state_meta_fields(state),
                    "replan": replanning_needed,
                    "replan_reason": replanning_reason,
                    "messages": delete_messages,
                    "observations": observations,
                    "replan_iterations": replan_iterations,
                    "current_plan_cached": current_plan,
                    "current_plan": "",
                    "planner_override_occurred": False,
                    "structure_validation_retried": False,
                },
                goto="planner",
            )
        else:
            analyst_result = prepare_reporter_input(parsed_analyst_res, report_style, workflow_type)
            return Command(
                update={
                    **preserve_state_meta_fields(state),
                    "replan": replanning_needed,
                    "replan_reason": "",
                    "messages": agent_messages,
                    "observations": observations + [analyst_result],
                    "replan_iterations": replan_iterations,
                },
                goto="reporter",
            )
    elif is_searcher:
        if tool_returns_cache is None:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__",
            )
        existing_citations = state.get("citations", [])
        existing_maps = state.get("document_chunk_maps", {})
        existing_metadata = state.get("document_metadata", {})
        curator_tool_input = format_tool_cache_for_curator(tool_returns_cache, current_step.title)
        # —— 提取 chunk_maps 和文档元数据 ——
        new_chunk_maps = extract_chunk_maps_from_cache(tool_returns_cache)
        # —— 合并到全局 chunk_maps（State 中已有的 + 本步骤须新增的）——
        for doc_title, cmap in new_chunk_maps.items():
            if doc_title in existing_maps:
                existing_maps[doc_title].update(cmap)
            else:
                existing_maps[doc_title] = cmap
        # Document metadata (全局累积)
        new_doc_metadata = extract_document_metadata_from_cache(tool_returns_cache)
        existing_metadata.update(new_doc_metadata)
        # Extract citations from tool call results (local_search, crawl, fetch)
        new_citations = extract_citations_from_cache(tool_returns_cache)
        # Citations (跨步骤 merge)
        merged_citations = merge_citations(existing_citations, new_citations)
        if new_citations:
            logger.info(
                f"Extracted {len(new_citations)} new citations from {agent_name} agent. Total citations: {len(merged_citations)}"
            )
        logger.info(f"curator_tool_input [{len(curator_tool_input)} chars]")
        return Command(
            update={
                **preserve_state_meta_fields(state),
                "messages": agent_messages,
                "search_results": search_results + [(response_content, curator_tool_input)],
                # 全局 chunk_map 累积
                "document_chunk_maps": existing_maps,
                # 全局文档元数据累积
                "document_metadata": existing_metadata,
                "citations": merged_citations,  # Store merged citations based on existing state and new tool results
            },
            goto="curator",
        )
    else:  # curator
        if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
            try:
                curator_output = parse_curator_output(response_content)
                resolved = resolve_all_evidence_chunks(
                    curator_output=curator_output,
                    document_chunk_maps=state.get("document_chunk_maps", {}),
                )
                # 使用 resolved_chunks 替换 body 中的具体内容（原文段落，未经精简）
                curator_rule_splitter_view = build_rule_splitter_view_with_resolved(
                    curator_output, resolved
                )
                curator_analyst_view = generate_analysis_view(curator_output, resolved)
            except CuratorParseError as e:
                logger.error(f"Curator 输出解析失败: {e}\n\n{response_content}\n重新推理...")
                curator_analyst_view = ""
                curator_rule_splitter_view = ""

            if not curator_analyst_view and not curator_rule_splitter_view:
                try:
                    async for chunk in agent.astream(
                        input={"messages": apply_prompt_template(agent_name, modified_state, configurable)},
                        config={"recursion_limit": recursion_limit},
                        stream_mode="values",
                    ):
                        if isinstance(chunk, dict) and "messages" in chunk:
                            accumulated_messages = chunk["messages"]
                    # If we get here, execution completed successfully
                    result = {"messages": accumulated_messages}
                except Exception as e:
                    import traceback

                    error_traceback = traceback.format_exc()
                    error_message = f"Error executing {agent_name} agent for step '{current_step.title}': {str(e)}"
                    logger.exception(error_message)
                    logger.error(f"Full traceback:\n{error_traceback}")
                    # Enhanced error diagnostics for content-related errors
                    if "Field required" in str(e) and "content" in str(e):
                        logger.error(f"Message content validation error detected")
                        for i, msg in enumerate(agent_input.get("messages", [])):
                            logger.error(
                                f"Message {i}: type={type(msg).__name__}, "
                                f"has_content={hasattr(msg, 'content')}, "
                                f"content_type={type(msg.content).__name__ if hasattr(msg, 'content') else 'N/A'}, "
                                f"content_len={len(str(msg.content)) if hasattr(msg, 'content') and msg.content else 0}"
                            )
                    return Command(
                        update=preserve_state_meta_fields(state),
                        goto="__end__",
                    )
                response_messages = result["messages"]
                response_content = response_messages[-1].content
                try:
                    curator_output = parse_curator_output(response_content)
                    resolved = resolve_all_evidence_chunks(
                        curator_output=curator_output,
                        document_chunk_maps=state.get("document_chunk_maps", {}),
                    )
                    # 使用 resolved_chunks 替换 body 中的具体内容（原文段落，未经精简）
                    curator_rule_splitter_view = build_rule_splitter_view_with_resolved(
                        curator_output, resolved
                    )
                    curator_analyst_view = generate_analysis_view(curator_output, resolved)
                except CuratorParseError as e:
                    logger.error(f"Curator 第二次输出解析失败: {e}\n\n{response_content}")
                    curator_analyst_view = "[ERROR] researcher agent error, 两次输出解析均失败。"
                    curator_rule_splitter_view = ""

            current_step.execution_res = curator_analyst_view
            return {
                **preserve_state_meta_fields(state),
                "messages": agent_messages,
                "curator_rule_splitter_views": curator_rule_splitter_views + [curator_rule_splitter_view],
                "observations": observations,
            }
        else:
            current_step.execution_res = response_content
            current_curator_info = (
                f"# 当前分析信息\n\n"
                f"## 分析标题\n\n{current_step.title}\n\n"
                f"## 分析背景\n\n{current_step.background}\n\n"
                f"## 相关总行指引和法规依据\n\n{response_content}"
            )
            return {
                **preserve_state_meta_fields(state),
                "messages": agent_messages,
                "observations": observations,
                "curator_rule_splitter_views": curator_rule_splitter_views + [current_curator_info],
            }


async def researcher_node(state: State, config: RunnableConfig) -> Command[Literal["curator"]]:
    """Searcher node that do search"""
    logger.info("researcher_node is researching.")
    logger.debug(f"[researcher_node] Starting researcher agent")
    configurable = Configuration.from_runnable_config(config)
    logger.debug(f"[researcher_node] Max search results: {configurable.max_search_results}")

    # Build tools list based on configuration
    # Add retriever tool if resources are available (always add, higher priority)
    retriever_tool = get_retriever_tool(
        configurable.max_search_results,
        configurable.report_style,
        state.get("resources", []),
    )
    if retriever_tool:
        logger.debug(f"[researcher_node] Adding retriever tool to tools list")
    else:
        return Command(
            update=preserve_state_meta_fields(state),
            goto="__end__",
        )
    tools = [retriever_tool, crawl_tool, fetch_tool]
    logger.info(f"[researcher_node] Researcher tools count: {len(tools)}")
    logger.debug(
        f"[researcher_node] Researcher tools: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]}"
    )
    tool_returns_cache = []
    agent_type = "researcher"
    agent = create_agent(agent_type, config, tools, tool_returns_cache)
    return await _execute_agent_step(state, agent, agent_type, config, tool_returns_cache)


async def curator_node(state: State, config: RunnableConfig):
    """Curator node"""
    logger.info("Curator node is analyzing.")
    logger.debug(f"[curator_node] Starting curator agent")
    agent_type = "curator"
    agent = create_agent(agent_type, config, [])
    # curator uses no tools - pure LLM reasoning
    return await _execute_agent_step(state, agent, agent_type, config)


async def analyst_node(state: State, config: RunnableConfig) -> Command[Literal["planner", "reporter"]]:
    """Analyst node that performs reasoning and analysis without code execution.
    This node handles tasks like:
    - Cross-validating information from multiple sources
    - Synthesizing research findings
    - Comparative analysis
    - Pattern recognition and trend analysis
    - General reasoning tasks that don't require code
    """
    logger.info("Analyst node is analyzing.")
    logger.debug(f"[analyst_node] Starting analyst agent for reasoning/analysis tasks")
    agent_type = "analyst"
    agent = create_agent(agent_type, config, [])
    # Analyst uses no tools - pure LLM reasoning
    return await _execute_agent_step(state, agent, agent_type, config)


async def get_response(agent, state, agent_input, agent_name, configurable):
    # Validate message content before invoking agent
    try:
        validated_messages = validate_message_content(agent_input["messages"])
        agent_input["messages"] = validated_messages
    except Exception as validation_error:
        logger.error(f"Error validating agent input messages: {validation_error}")

    modified_state = copy.deepcopy(state)
    modified_state["messages"] = agent_input["messages"]
    accumulated_messages = []
    try:
        # Use astream (async) from the start to capture messages in real-time
        # This allows us to retrieve accumulated messages even if recursion limit is hit
        # NOTE: astream is required for MCP tools which only support async invocation
        async for chunk in agent.astream(
            input={"messages": apply_prompt_template(agent_name, modified_state, configurable)},
            # config={"configurable": {"output_token_limit": max_tokens}},
            stream_mode="values",
        ):
            if isinstance(chunk, dict) and "messages" in chunk:
                accumulated_messages = chunk["messages"]
        # response_content = accumulated_messages[-1].content
        if accumulated_messages:
            response_content = accumulated_messages[-1].content.strip()
            return response_content
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = f"Error executing {agent_name} agent: {str(e)}"
        logger.exception(error_message)
        logger.error(f"Full traceback:\n{error_traceback}")
        # Enhanced error diagnostics for content-related errors
        if "Field required" in str(e) and "content" in str(e):
            logger.error(f"Message content validation error detected")
            for i, msg in enumerate(modified_state.get("messages", [])):
                logger.error(
                    f"Message {i}: type={type(msg).__name__}, "
                    f"has_content={hasattr(msg, 'content')}, "
                    f"content_type={type(msg.content).__name__ if hasattr(msg, 'content') else 'N/A'}, "
                    f"content_len={len(str(msg.content)) if hasattr(msg, 'content') and msg.content else 0}"
                )
        return None


async def rule_splitter_node(state: State, config: RunnableConfig) -> dict:
    """业务原子规则拆分专家"""

    def extract_content_after_heading(text: str) -> str:
        """从多行字符串中提取 "# 相关总行指引和法规依据" 标题后面的所有内容。
        参数:
            text (str): 输入的多行字符串
        返回:
            str: 标题之后的所有内容（不含标题），若未找到则返回空字符串
        """
        # 定义正则表达式：匹配标题后的所有内容（包括换行）
        pattern = r"#\s*相关总行指引和法规依据\s*\n(.*)"
        # 使用 re.DOTALL 使 . 匹配包括换行符在内的所有字符
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # 返回标题之后的所有内容，去除可能的前导空白
            content = match.group(1).lstrip("\n")
            # 去掉紧接标题后的换行
            return content
        else:
            # 标题未找到，返回空字符串
            return ""

    logger.info("Rule Splitter node is running.")
    all_views = state.get("curator_rule_splitter_views", [])
    atomic_rules = state.get("atomic_rules", [])
    agent_type = "rule_splitter"
    if not all_views:
        return {
            "atomic_rules": atomic_rules,
            **preserve_state_meta_fields(state),
        }
    idx_str = str(len(atomic_rules) + 1)
    latest_view = all_views[-1] if all_views else ""
    configurable = Configuration.from_runnable_config(config)
    original_topic = state.get("original_topic")
    if latest_view:
        report_style = configurable.report_style
        if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
            agent_input = {
                "messages": [
                    HumanMessage(
                        content=f"\n\n# 调查原始问题\n\n{original_topic}\n\n# 信息质量评估报告\n\n{latest_view}",
                        name="user",
                    )
                ]
            }
        elif report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
            agent_input = {
                "messages": [
                    HumanMessage(
                        content=f"\n\n# 信用卡业务宣传文本\n\n{original_topic}\n\n{latest_view}",
                        name="user",
                    )
                ]
            }
            state = {
                **state,
                "cp_edge_constraints": render_cp_edge_constraints_for_assessor(),
                "element_class_mapping": render_element_class_mapping(),
            }
        else:
            agent_input = {
                "messages": [
                    HumanMessage(
                        content=f"\n\n# 调查原始问题\n\n{original_topic}\n\n# 信息质量评估报告\n\n{latest_view}",
                        name="user",
                    )
                ]
            }
        rule_splitter = create_agent(agent_type, config, [])
        this_filtering_content = await get_response(
            rule_splitter, state, agent_input, agent_type, configurable
        )
        if this_filtering_content is not None and this_filtering_content:
            if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
                this_filtering_content = this_filtering_content.replace(
                    "## 限定性特征", f"## 限定性特征 {idx_str}"
                )
                this_filtering_content = this_filtering_content.replace(
                    "## 关联业务原子规则清单", f"## 关联业务原子规则清单 {idx_str}"
                )
                this_filtering_content = this_filtering_content.replace(
                    "**业务原子规则", f"**业务原子规则{idx_str}"
                )
            elif report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
                latest_view_str = extract_content_after_heading(latest_view)
                this_filtering_content = this_filtering_content.strip()
                this_filtering_content = (
                    f"\n\n---\n\n### 合规判定结果 {idx_str}\n\n{this_filtering_content}\n\n"
                    f"## 相关总行指引和法规依据 {idx_str}\n\n{latest_view_str}\n\n---\n\n"
                )
            logger.debug(f"rule_splitter response_content reasoning {this_filtering_content}")
            return {
                "atomic_rules": atomic_rules + [this_filtering_content],
                **preserve_state_meta_fields(state),
            }
    logger.warning(f"No split rules found for rule_splitter, Curator 输出解析失败")
    return {
        "atomic_rules": atomic_rules,
        **preserve_state_meta_fields(state),
    }


async def arbitrator_node(state: State, config: RunnableConfig) -> dict:
    """业务矛盾仲裁专家"""
    logger.info("Arbitrator node is running to resolve conflicts.")
    all_atomic_rules = state.get("atomic_rules", [])
    arbitration_result = state.get("arbitration_result", "")
    if not all_atomic_rules:
        logger.warning("No atomic rules found for arbitration.")
        return {"arbitration_result": arbitration_result}
    all_atomic_rules_str = "\n\n".join(all_atomic_rules)
    configurable = Configuration.from_runnable_config(config)
    original_topic = state.get("original_topic")
    workflow_type = state.get("workflow_type", "")
    workflow_type_str = "A 定点调查"
    if workflow_type == "B":
        workflow_type_str = "B 并行对比"
    elif workflow_type == "C":
        workflow_type_str = "C 扫描穷举"
    elif workflow_type == "D":
        workflow_type_str = "D 条件推理"

    agent_type = "arbitrator"
    arbitrator = create_agent(
        agent_type,
        config,
        [],
    )
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"# 调查原始问题\n\n{original_topic}\n\n# 当前工作流类型\n\n{workflow_type_str}\n\n{all_atomic_rules_str}",
                name="user",
            )
        ]
    }
    sub = {
        **state,
        "arbitrator_bucketing_guide": render_arbitrator_bucketing_guide(),
    }
    response_content = await get_response(arbitrator, sub, agent_input, agent_type, configurable)
    if response_content is not None and response_content:
        logger.debug(f"arbitrator response_content {response_content}")
        return {
            "arbitration_result": response_content,
            **preserve_state_meta_fields(state),
        }
    logger.warning(f"No atomic rules found for arbitration. response_content {response_content}")
    return {
        "arbitration_result": arbitration_result,
        **preserve_state_meta_fields(state),
    }


def _extract_xml_block(text: str, tag: str) -> str:
    """提取 <tag>...</tag> 标签内的文本（非贪婪，取第一个匹配）。无匹配返回空串。"""
    if not text:
        return ""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_analyst_output(text: str) -> dict[str, Any] | None:
    """
    解析 Analyst 输出（XML 标签 + 小 JSON 混合契约）。
    新契约 (analyst.md 2026-06-11 起):
      <analyst_meta>{小 JSON...}</analyst_meta>
      <analysis_text>Markdown...</analysis_text>
      <contradiction_text>...</contradiction_text>
      <risk_text>...</risk_text>

    兼容路径：未发现 <analyst_meta> 标签时回退到旧版"单一大 JSON"解析，保证旧提示词缓存 / 模型未遵循新契约时不致硬失败。
    返回与旧契约同构的 dict（文本块以 analysis_text 等 key 并入，便于下游 (reporter / replanning 判断) 无感切换。
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
