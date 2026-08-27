# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Literal

from google.api_core.retry import retry_unary

# Define available LLM types
LLMType = Literal["basic", "reasoning"]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = {
    "coordinator": "basic",
    "background_investigator": "basic",
    "ontology_mapper": "basic",
    "planner": "basic",
    "researcher": "basic",
    "curator": "basic",
    "rule_splitter": "basic",
    "arbitrator": "basic",
    "analyst": "basic",
    "coder": "basic",
    "reporter": "basic",
    "podcast_script_writer": "basic",
    "ppt_composer": "basic",
    "prose_writer": "basic",
    "prompt_enhancer": "basic",
}

def get_agent_llm_type(agent_name: str, enable_deep_thinking=True) -> LLMType:
    llm_type = AGENT_LLM_MAP[agent_name]

    return llm_type
