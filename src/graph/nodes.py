import json
import logging
import os
from functools import partial
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt

from src.agents import create_agent
from src.citations import extract_citations_from_messages, merge_citations
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type, get_llm_token_limit_by_type
from src.prompts.planner_model import Plan
from src.prompts.template import apply_prompt_template
from src.tools import (
    crawl_tool,
    get_retriever_tool
)
from src.utils.context_manager import ContextManager, validate_message_content
from src.utils.json_utils import repair_json_output, sanitize_tool_response

from .types import State
from .utils import (
    build_clarified_topic_from_history,
    get_message_content,
    reconstruct_clarification_history,
)

logger = logging.getLogger(__name__)


@tool
def handoff_to_planner(
        research_topic: Annotated[str, "The topic of the research task to be handed off."],
        locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    return


@tool
def handoff_after_clarification(
        locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
        research_topic: Annotated[
            str, "The clarified research topic based on all clarification rounds."
        ],
):
    """Handoff to planner after clarification rounds are complete."""
    return


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
    return {
        "locale": state.get("locale", "en-US"),
        "research_topic": state.get("research_topic", ""),
        "clarified_research_topic": state.get("clarified_research_topic", ""),
        "clarification_history": state.get("clarification_history", []),
        "enable_clarification": state.get("enable_clarification", False),
        "max_clarification_rounds": state.get("max_clarification_rounds", 3),
        "clarification_rounds": state.get("clarification_rounds", 0),
        "resources": state.get("resources", []),
    }


def background_investigation_node(state: State, config: RunnableConfig):
    logger.info("background investigation node is running.")
    configurable = Configuration.from_runnable_config(config)

    if not configurable.enable_web_search:
        logger.info("Web search is disabled, skipping background investigation.")
        return {"background_investigation_results": json.dumps([], ensure_ascii=False)}

    query = state.get("clarified_research_topic") or state.get("research_topic")
    background_investigation_results = []

    return {
        "background_investigation_results": json.dumps(
            background_investigation_results, ensure_ascii=False
        )
    }


def extract_plan_content(plan_data: str | dict | Any) -> str:
    if isinstance(plan_data, str):
        return plan_data
    elif hasattr(plan_data, 'content') and isinstance(plan_data.content, str):
        return plan_data.content
    elif isinstance(plan_data, dict):
        if "content" in plan_data:
            if isinstance(plan_data["content"], str):
                return plan_data["content"]
            if isinstance(plan_data["content"], dict):
                return json.dumps(plan_data["content"], ensure_ascii=False)
            else:
                return str(plan_data["content"])
        else:
            return json.dumps(plan_data)
    else:
        return str(plan_data)


def planner_node(state: State, config: RunnableConfig) -> dict:
    """Planner node that generates the full plan."""
    logger.info("Planner generating full plan with locale: %s", state.get("locale", "en-US"))
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state.get("plan_iterations", 0)

    # Prepare context for Planner
    if state.get("enable_clarification", False) and state.get("clarified_research_topic"):
        modified_state = state.copy()
        modified_state["messages"] = [
            {"role": "user", "content": state["clarified_research_topic"]}
        ]
        modified_state["research_topic"] = state["clarified_research_topic"]
        messages = apply_prompt_template("planner", modified_state, configurable, state.get("locale", "en-US"))
    else:
        messages = apply_prompt_template("planner", state, configurable, state.get("locale", "en-US"))

    if state.get("enable_background_investigation") and state.get("background_investigation_results"):
        messages += [
            {
                "role": "user",
                "content": (
                        "background investigation results of user query:\n"
                        + state["background_investigation_results"]
                        + "\n"
                ),
            }
        ]

    # Model Selection (Qwen3-235B-A22B recommended)
    llm = get_llm_by_type(AGENT_LLM_MAP.get("planner", "basic"))

    full_response = ""
    response = llm.stream(messages)
    for chunk in response:
        full_response += chunk.content

    logger.info(f"Planner response: {full_response}")

    try:
        curr_plan = json.loads(repair_json_output(full_response))
        curr_plan_content = extract_plan_content(curr_plan)
        new_plan_dict = json.loads(repair_json_output(curr_plan_content))
        new_plan = Plan.model_validate(new_plan_dict)
    except Exception as e:
        logger.warning(f"Failed to parse plan: {str(e)}")
        # If parsing fails, create a generic fallback plan to avoid breaking the graph
        new_plan = Plan(
            thought="Fallback plan due to parsing error.",
            title="Fallback Investigation",
            steps=[]
        )

    # Just return the updated state. The router `route_from_planner` will dictate the next node.
    return {
        "messages": [AIMessage(content=full_response, name="planner")],
        "current_plan": new_plan,
        "plan_iterations": plan_iterations + 1,
        **preserve_state_meta_fields(state),
    }


def coordinator_node(state: State, config: RunnableConfig) -> Command:
    """Coordinator node that communicate with customers and handle clarification."""
    logger.info("Coordinator talking.")
    configurable = Configuration.from_runnable_config(config)

    enable_clarification = state.get("enable_clarification", False)
    initial_topic = state.get("research_topic", "")
    clarified_topic = initial_topic

    # ...(保留原有 coordinator_node 逻辑，处理澄清与工具调用) ...
    # 为了精简，保留您提供的 coordinator 原有实现逻辑。此处略过大段未变动代码。
    # 核心是返回 Command(update={...}, goto="planner" 或 "__end__")

    goto = "planner"  # 简化示意
    return Command(
        update={
            "research_topic": initial_topic,
            **preserve_state_meta_fields(state)
        },
        goto=goto,
    )


async def researcher_node(state: State, config: RunnableConfig) -> dict:
    """Researcher node that executes ONE research step."""
    logger.info("Researcher node is researching (Information Retrieval).")
    configurable = Configuration.from_runnable_config(config)

    current_plan = state.get("current_plan")
    observations = state.get("observations", [])

    # 1. Find the first unexecuted research step
    current_step = None
    for step in current_plan.steps:
        if not step.execution_res and step.step_type == "research":
            current_step = step
            break

    if not current_step:
        logger.warning("Researcher called but no pending research step found.")
        return {}

    # 2. Prepare Tools (RAG Knowledge base)
    tools = []
    retriever_tool = get_retriever_tool(state.get("resources", []))
    if retriever_tool:
        tools.append(retriever_tool)

    # 3. Setup Agent (qwen3-next-80B recommended)
    agent_type = "researcher"
    locale = state.get("locale", "en-US")
    llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP[agent_type])
    pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)

    agent = create_agent(
        agent_type, agent_type, tools, agent_type, pre_model_hook,
        interrupt_before_tools=configurable.interrupt_before_tools, locale=locale
    )

    # 4. Agent Execution
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"# Current Research Step\n\n## Title\n{current_step.title}\n\n## Description\n{current_step.description}\n\n"
                        f"IMPORTANT: Use tools to fetch precise banking rules. Do not hallucinate."
            )
        ]
    }

    try:
        result = await agent.ainvoke(input=agent_input, config={"recursion_limit": 15})
        response_content = sanitize_tool_response(str(result["messages"][-1].content))
        current_step.execution_res = response_content

        agent_messages = result.get("messages", [])
        new_citations = extract_citations_from_messages(agent_messages)
        merged_citations = merge_citations(state.get("citations", []), new_citations)

        return {
            "current_plan": current_plan,  # Update plan with execution_res
            "observations": observations + [response_content],
            "citations": merged_citations,
            **preserve_state_meta_fields(state),
        }
    except Exception as e:
        logger.error(f"Researcher error: {e}")
        current_step.execution_res = f"Error during research: {str(e)}"
        return {
            "current_plan": current_plan,
            "observations": observations + [current_step.execution_res],
            **preserve_state_meta_fields(state),
        }


async def rule_splitter_node(state: State, config: RunnableConfig) -> dict:
    """业务原子规则拆分专家 (qwen3-next-80B)"""
    logger.info("Rule Splitter node is running.")
    observations = state.get("observations", [])
    atomic_rules = state.get("atomic_rules", [])

    if not observations:
        return {}

    # Get the latest observation to split
    latest_obs = observations[-1]

    # 模拟构建拆分专家的 Prompt
    messages = [
        SystemMessage(
            content="你是民生银行信用卡业务规则原子处理专家。请严格按照要求，将以下检索到的文档片段拆分为清晰的原子规则(IF-THEN 格式)。"),
        HumanMessage(content=f"## 待处理信息：\n{latest_obs}")
    ]

    llm = get_llm_by_type(AGENT_LLM_MAP.get("rule_splitter", "researcher"))
    response = await llm.ainvoke(messages)

    new_rules = response.content
    logger.info(f"Rule Splitter extracted rules: {new_rules[:100]}...")

    return {
        "atomic_rules": atomic_rules + [new_rules],
        **preserve_state_meta_fields(state)
    }


async def arbitrator_node(state: State, config: RunnableConfig) -> dict:
    """业务矛盾仲裁专家 (Qwen3-235B-A22B)"""
    logger.info("Arbitrator node is running to resolve conflicts.")
    atomic_rules = state.get("atomic_rules", [])

    if not atomic_rules:
        logger.warning("No atomic rules found for arbitration.")
        return {"arbitration_result": "无可用规则进行仲裁。"}

    rules_text = "\n\n".join([f"规则片段 {i + 1}:\n{rule}" for i, rule in enumerate(atomic_rules)])

    messages = [
        SystemMessage(
            content="你是严谨的银行信用卡业务信息处理专家。请对以下提取的【所有业务规则】进行比对，识别矛盾点，并执行仲裁路径（红线禁令优先等），输出 Markdown 格式的《矛盾信息仲裁报告》。"),
        HumanMessage(content=f"## 业务原子规则清单：\n{rules_text}")
    ]

    llm = get_llm_by_type(AGENT_LLM_MAP.get("arbitrator", "planner"))
    response = await llm.ainvoke(messages)

    arbitration_result = response.content
    logger.info("Arbitration completed.")

    return {
        "arbitration_result": arbitration_result,
        **preserve_state_meta_fields(state)
    }


async def analyst_node(state: State, config: RunnableConfig) -> dict:
    """业务分析专家 (Qwen3-235B-A22B) - No tools, pure logic synthesis."""
    logger.info("Analyst node is analyzing the aggregated data.")

    current_plan = state.get("current_plan")
    arbitration_result = state.get("arbitration_result", "")
    original_question = state.get("clarified_research_topic") or state.get("research_topic")

    # Mark analysis steps as complete in the plan
    for step in current_plan.steps:
        if step.step_type == "analysis" and not step.execution_res:
            step.execution_res = "Analyzed by global analyst node."

    messages = [
        SystemMessage(
            content="你是民生银行信用卡业务分析专家。请基于用户原始问题和《矛盾信息仲裁报告》，进行综合推理并给出确切结论。如果发现信息严重缺失无法得出结论，请在报告末尾明确输出 JSON 标记 `{\"replanning_needed\": true}`。"),
        HumanMessage(content=f"## 调查原始问题：\n{original_question}\n\n## 矛盾信息仲裁报告：\n{arbitration_result}")
    ]

    llm = get_llm_by_type(AGENT_LLM_MAP.get("analyst", "planner"))
    response = await llm.ainvoke(messages)
    analysis_output = response.content

    # Check if replanning is needed based on LLM output
    replanning_needed = False
    if "replanning_needed\": true" in analysis_output.lower() or "replanning_needed\":true" in analysis_output.lower():
        replanning_needed = True
        logger.info("Analyst requested replanning due to insufficient evidence.")

    return {
        "current_plan": current_plan,
        "analyst_observations": analysis_output,
        "replanning_needed": replanning_needed,
        **preserve_state_meta_fields(state)
    }


def reporter_node(state: State, config: RunnableConfig) -> dict:
    """Reporter node that writes the final report."""
    logger.info("Reporter writing final report")
    configurable = Configuration.from_runnable_config(config)

    original_question = state.get("clarified_research_topic") or state.get("research_topic")
    analyst_observations = state.get("analyst_observations", "")
    citations = state.get("citations", [])

    # Format the prompt exactly as required by the Report Structure image
    input_messages = [
        SystemMessage(
            content="你是一位拥有多年从业经验的民生银行信用卡业务专家。任务是针对“调查原始问题”，利用“analyst task observations”和“可用参考来源”生成最终业务报告。严格遵守 Markdown 格式，不捏造 URL。"),
        HumanMessage(
            content=f"### 调查原始问题\n{original_question}\n\n### Analyst Observations (核心结论与风险提示)\n{analyst_observations}")
    ]

    if citations:
        citation_list = "\n### 可用参考来源\n"
        for i, citation in enumerate(citations, 1):
            title = citation.get("title", "Untitled")
            url = citation.get("url", "")
            citation_list += f"- [{i}] **{title}** ({url})\n"

        input_messages.append(HumanMessage(content=citation_list))

    # Compress Context if needed
    llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP.get("reporter", "basic"))
    compressed_state = ContextManager(llm_token_limit).compress_messages({"messages": input_messages})

    llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "basic"))
    response = llm.invoke(compressed_state.get("messages", input_messages))

    logger.info(f"Final report generated successfully.")

    return {
        "final_report": response.content,
        "citations": citations,
    }