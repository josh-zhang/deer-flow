# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import dataclasses
import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape
from langchain.agents import AgentState

from src.config.configuration import Configuration
from src.config.report_style import ReportStyle

# Initialize Jinja2 environment
bi_env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__) + "/BI"),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)
cg_env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__) + "/CG"),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)
cp_env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__) + "/CP"),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_prompt_template(report_style: str, prompt_name: str) -> str:
    """
    Load and return a prompt template using Jinja2.

    Args:
        report_style: bi | cp | cg
        prompt_name: Name of the prompt template file (without .md extension)

    Returns:
        The template string with proper variable substitution syntax
    """
    if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
        this_env = bi_env
    elif report_style == ReportStyle.CUSTOMER_PROTECTION.value:
        this_env = cp_env
    else:
        this_env = cg_env

    try:
        # Try locale-specific template first (e.g., researcher.zh_CN.md)
        try:
            template = this_env.get_template(f"{prompt_name}.zh_CN.md")
            return template.render()
        except TemplateNotFound:
            # Fallback to English template if locale-specific not found
            template = this_env.get_template(f"{prompt_name}.md")
            return template.render()
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}")


def apply_prompt_template(
    report_style: str, prompt_name: str, state: AgentState, configurable: Configuration = None
) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.

    Args:
        report_style: bi | cp | cg
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute
        configurable: Configuration object with additional variables

    Returns:
        List of messages with the system prompt as the first message
    """
    try:
        system_prompt = get_system_prompt_template(report_style, prompt_name, state, configurable)
        return [{"role": "system", "content": system_prompt}] + state["messages"]
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}")

def get_system_prompt_template(
    report_style: str, prompt_name: str, state: AgentState, configurable: Configuration = None
) -> str:
    """
    Render and return the system prompt template with state and configuration variables.
    This function loads a Jinja2-based prompt template (with optional locale-specific
    variants), applies variables from the agent state and Configuration object, and
    returns the fully rendered system prompt string.
    Args:
        report_style: bi | cp | cg
        prompt_name: Name of the prompt template to load (without .md extension).
        state: Current agent state containing variables available to the template.
        configurable: Optional Configuration object providing additional template variables.
    Returns:
        The rendered system prompt string after applying all template variables.
    """
    # Convert state to dict for template rendering
    if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
        this_env = bi_env
    elif report_style == ReportStyle.CUSTOMER_PROTECTION.value:
        this_env = cp_env
    else:
        this_env = cg_env

    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        **state,
    }

    # Add configurable variables
    if configurable:
        state_vars.update(dataclasses.asdict(configurable))

    try:

        # Try locale-specific template first
        try:
            template = this_env.get_template(f"{prompt_name}.zh_CN.md")
        except TemplateNotFound:
            # Fallback to English template
            template = this_env.get_template(f"{prompt_name}.md")

        system_prompt = template.render(**state_vars)
        return f"{system_prompt}\n\n输出数学公式时必须使用 Unicode 字符，例如 '→'。"
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}: {e}")