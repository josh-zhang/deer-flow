"""Curator JSON 解析器（极简版）。

全部逻辑 = json.loads + Pydantic 校验 + 计数统计。
没有任何正则表达式。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from curator_models import CuratorOutput, Relevance

logger = logging.getLogger(__name__)


class CuratorParseError(Exception):
    """Curator 解析失败"""


def parse_curator_output(text: str) -> CuratorOutput:
    """
    将 Evidence Curator 的 JSON 文本解析为 CuratorOutput。

    鲁棒性保障链：
    1. json.loads() — 确定性 JSON 解析
    2. _try_repair() — 截断修复兜底
    3. Pydantic model_validate() — 类型校验 + 默认值 + 模糊枚举匹配
    4. 计数字段由代码统计
    """
    text = text.strip()

    # ── Step 1: JSON 反序列化 ──
    data = _load_json(text)

    # ── Step 2: Pydantic 校验 ──
    try:
        output = CuratorOutput.model_validate(data)
    except Exception as e:
        raise CuratorParseError(f"Pydantic 校验失败: {e}") from e

    # ── Step 3: 填充 ID 和计数 ──
    for idx, evidence in enumerate(output.evidences):
        evidence.id = idx + 1

    output.retained_count = len(output.evidences)
    output.direct_count = sum(
        1 for e in output.evidences if e.relevance == Relevance.DIRECT
    )
    output.indirect_count = sum(
        1 for e in output.evidences if e.relevance == Relevance.INDIRECT
    )
    output.uncertain_high_count = sum(
        1 for e in output.evidences if e.relevance == Relevance.UNCERTAIN_HIGH
    )
    output.uncertain_low_count = sum(
        1 for e in output.evidences if e.relevance == Relevance.UNCERTAIN_LOW
    )

    # discarded_count：尝试从 discarded 文本中计数表格行
    output.discarded_count = _count_discarded(output.discarded)
    output.total_input_count = output.retained_count + output.discarded_count

    return output


def _load_json(text: str) -> dict:
    """尝试解析 JSON，失败时进行截断修复。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 兜底 1：去除 markdown 代码块包裹
    cleaned = text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # 去首行 ```json 和末行 ```
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # 兜底 2：json-repair 截断修复
    try:
        import json_repair
        return json_repair.loads(text)
    except Exception:
        pass

    raise CuratorParseError("JSON 解析失败，所有修复策略均未成功")


def _count_discarded(discarded_text: str) -> int:
    """从 discarded 自由文本中估算丢弃数量（尽力而为）。"""
    if "无丢弃" in discarded_text or not discarded_text.strip():
        return 0
    # 计算表格数据行数（含 | 但不含 --- 的行，排除表头）
    lines = discarded_text.strip().split("\n")
    data_rows = 0
    for line in lines:
        line = line.strip()
        if "|" in line and "---" not in line and "序号" not in line:
            data_rows += 1
    return max(data_rows, 0)