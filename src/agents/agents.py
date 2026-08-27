import asyncio
import inspect
import logging
import json
from typing import Any, Callable
from functools import partial

from langchain.agents import create_agent as langchain_create_agent
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage

from src.agents.tool_interceptor import wrap_tools_with_interceptor
from src.config.agents import get_agent_llm_type
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type, get_llm_token_limit_by_type
from src.utils.context_manager import ContextManager
from src.rag.retriever import ToolCallRecord, ToolCallArtifact

logger = logging.getLogger(__name__)


class PreModelHookMiddleware(AgentMiddleware):

    def __init__(self, pre_model_hook: Callable):
        self._pre_model_hook = pre_model_hook

    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Execute the pre-model hook."""
        if not self._pre_model_hook:
            return None

        try:
            result = self._pre_model_hook(state, runtime)
            return result
        except Exception as e:
            logger.error(
                f"Pre-model hook execution failed in before_model: {e}",
                exc_info=True
            )
            return None

    async def abefore_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Async version of before_model."""
        if not self._pre_model_hook:
            return None

        try:
            # Check if the hook is async
            if inspect.iscoroutinefunction(self._pre_model_hook):
                result = await self._pre_model_hook(state, runtime)
            else:
                # Run synchronous hook in thread pool to avoid blocking event loop
                result = await asyncio.to_thread(self._pre_model_hook, state, runtime)
            return result
        except Exception as e:
            logger.error(
                f"Pre-model hook execution failed in abefore_model: {e}",
                exc_info=True
            )
            return None


def make_tool_saver_hook(cache: list[ToolCallRecord], context_manager: ContextManager):
    seen_ids: set[str] = set()

    def hook(state, *args, **kwargs) -> list:
        if isinstance(state, list):
            messages = state
        elif isinstance(state, dict) and "messages" in state:
            messages = state["messages"]
        else:
            return context_manager.compress_messages(state, *args, **kwargs)

        logger.info(f"hooking {len(messages)} messages")

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

        logger.info(f"cached {len(cache)} messages")

        return context_manager.compress_messages(state, *args, **kwargs)

    return hook


# Create agents using configured LLM types
def create_agent(
        agent_type: str,
        config: RunnableConfig,
        tools: list,
        tool_returns_cache: list[ToolCallRecord] = []
):
    """Factory function to create agents with consistent configuration.

    Args:
        agent_type: Type of agent (researcher, coder, etc.)
        config: RunnableConfig
        tools: List of tools available to the agent
        interrupt_before_tools: Optional list of tool names to interrupt before execution

    Returns:
        A configured agent graph
    """
    logger.debug(
        f"Creating agent '{agent_type}' with {len(tools)} tools"
    )

    configurable = Configuration.from_runnable_config(config)

    llm_type = get_agent_llm_type(agent_type, configurable.enable_deep_thinking)

    # Wrap tools with interrupt logic if specified
    processed_tools = tools
    interrupt_before_tools = configurable.interrupt_before_tools
    if interrupt_before_tools:
        logger.info(
            f"Creating agent '{agent_type}' with tool-specific interrupts: {interrupt_before_tools}"
        )
        logger.debug(f"Wrapping {len(tools)} tools for agent '{agent_type}'")
        processed_tools = wrap_tools_with_interceptor(tools, interrupt_before_tools)
        logger.debug(f"Agent '{agent_type}' tool wrapping completed")
    else:
        logger.debug(f"Agent '{agent_type}' has no interrupt-before-tools configured")

    llm_token_limit = get_llm_token_limit_by_type(llm_type)

    if processed_tools:
        middleware = [
            PreModelHookMiddleware(make_tool_saver_hook(tool_returns_cache, ContextManager(llm_token_limit, 3)))]
    else:
        middleware = [PreModelHookMiddleware(partial(ContextManager(llm_token_limit, 3).compress_messages))]

    agent = langchain_create_agent(
        name=agent_type,
        model=get_llm_by_type(llm_type),
        tools=processed_tools,
        middleware=middleware,
        debug=False
    )

    logger.info(f"Agent '{agent_type}' created successfully")

    return agent
