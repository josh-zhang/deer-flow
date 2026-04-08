"""
curator_node.py
在 LangGraph 中集成 Curator：LLM 输出 Markdown → 解析器 → CuratorOutput → 视图
"""

import logging

from langchain_core.messages import AIMessage

from curator_models import CuratorOutput
from curator_parser import CuratorParseError, parse_curator_markdown
from curator_views import (
    build_analyst_view,
    build_rule_splitter_view,
    build_searcher_view,
)

logger = logging.getLogger(__name__)

# 最大重试次数（解析失败时要求 LLM 修正格式后重新输出）
MAX_PARSE_RETRIES = 1

# 解析失败时追加的修正提示
REPAIR_PROMPT = """你上一次输出的格式存在解析问题：{error}

请严格按照输出格式规范重新输出完整结果。特别注意：
1. 每个区块必须以 `### ` 三级标题开头
2. 每条业务依据必须以 `**业务依据 N**` 开头（N 从 1 连续编号）
3. 字段格式必须为 `- *字段名*：值`
4. `- *具体内容*：` 后必须换行，内容从下一行开始
5. 存疑-低的具体内容固定写 `[存疑-低，不输出具体内容]`
"""


def invoke_curator_with_retry(
    llm,
    system_prompt: str,
    user_message: str,
    max_retries: int = MAX_PARSE_RETRIES,
) -> tuple[CuratorOutput, str]:
    """
    调用 Curator LLM 获取 Markdown → 解析为 CuratorOutput。
    解析失败时自动重试（追加错误提示让 LLM 修正格式）。

    Returns:
        (CuratorOutput, raw_markdown)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for attempt in range(1 + max_retries):
        response: AIMessage = llm.invoke(messages)
        raw_markdown = response.content

        try:
            output = parse_curator_markdown(raw_markdown)
            # ── 基本校验 ──
            _validate_output(output)
            return output, raw_markdown
        except (CuratorParseError, ValidationError) as e:
            logger.warning(f"Curator 输出解析失败 (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                # 追加修正提示，要求 LLM 重新输出
                messages.append({"role": "assistant", "content": raw_markdown})
                messages.append({
                    "role": "user",
                    "content": REPAIR_PROMPT.format(error=str(e)),
                })
            else:
                logger.error(f"Curator 解析在 {max_retries + 1} 次尝试后仍然失败")
                raise


class ValidationError(Exception):
    pass


def _validate_output(output: CuratorOutput) -> None:
    """业务级校验（代码层面保证数据一致性）"""
    for ev in output.evidences:
        # 存疑-低必须无 content
        if ev.relevance.value == "存疑-低" and ev.content is not None:
            raise ValidationError(
                f"业务依据 {ev.id} 为存疑-低但 content 不为 null"
            )
        # 非存疑-低必须有 content
        if ev.relevance.value != "存疑-低" and not ev.content:
            raise ValidationError(
                f"业务依据 {ev.id} 为{ev.relevance.value}但 content 为空"
            )

    # 覆盖度标的数量校验
    if len(output.coverage.target_details) != len(output.evaluation_basis.specific_targets):
        raise ValidationError(
            f"覆盖度标的数({len(output.coverage.target_details)}) "
            f"≠ 评估基准标的数({len(output.evaluation_basis.specific_targets)})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LangGraph 节点
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def curator_node(state: dict) -> dict:
    """Evidence Curator LangGraph 节点"""
    step_idx = state["current_step_index"]
    step = state["plan_steps"][step_idx]
    searcher_output = state["messages"][-1].content

    user_msg = (
        f"# 调查原始问题\n\n{state['original_question']}\n\n"
        f"# 当前步骤\n\n"
        f"- title: {step['title']}\n"
        f"- background: {step['background']}\n"
        f"- description: {step['description']}\n\n"
        f"# Searcher 输出\n\n{searcher_output}"
    )

    curator_output, raw_md = invoke_curator_with_retry(
        llm=curator_llm,
        system_prompt=CURATOR_SYSTEM_PROMPT,
        user_message=user_msg,
    )

    # 构建三个视图
    return {
        "curator_outputs": {
            **state.get("curator_outputs", {}),
            step_idx: curator_output,
        },
        "rule_splitter_views": {
            **state.get("rule_splitter_views", {}),
            step_idx: build_rule_splitter_view(curator_output),
        },
        "analyst_views": {
            **state.get("analyst_views", {}),
            step_idx: build_analyst_view(curator_output),
        },
        "searcher_views": {
            **state.get("searcher_views", {}),
            step_idx: build_searcher_view(curator_output),
        },
    }