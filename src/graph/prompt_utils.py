"""
V2 Investigation Pipeline — Prompt Rendering & Validation Utilities
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Jinja Environment Setup
# ─────────────────────────────────────────────

_PROMPT_DIR = Path(__file__).parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPT_DIR)),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


# ═══════════════════════════════════════════════════
#  Prompt Rendering Functions
# ═══════════════════════════════════════════════════

def render_planner_prompt(
    workflow_type: str,
    max_step_num: int = 5,
) -> str:
    """
    用 Jinja 渲染 Planner Prompt。
    根据 workflow_type 动态注入对应的分解策略 + 示例 + 验证清单。
    """
    template = _jinja_env.get_template("planner.md.j2")
    return template.render(
        workflow_type=workflow_type,
        max_step_num=max_step_num,
        CURRENT_TIME=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def render_analyst_prompt(workflow_type: str) -> str:
    """
    用 Jinja 渲染 Analyst Prompt。
    根据 workflow_type 动态注入对应的分析框架 + 输出模板。
    """
    template = _jinja_env.get_template("analyst.md.j2")
    return template.render(workflow_type=workflow_type)


def render_reporter_prompt(workflow_type: str) -> str:
    """
    用 Jinja 渲染 Reporter Prompt。
    根据 workflow_type 动态注入对应的报告模板。
    """
    template = _jinja_env.get_template("reporter.md.j2")
    return template.render(workflow_type=workflow_type)


def render_searcher_prompt(workflow_type: str) -> str:
    """
    渲染 Searcher Prompt。
    当前为静态 Prompt（含所有 workflow 策略），workflow_type 作为输入参数传递。
    未来可改为 Jinja 动态渲染。
    """
    return load_prompt("searcher.md")


def render_arbitrator_prompt(workflow_type: str) -> str:
    """
    渲染 Arbitrator Prompt。
    Arbitrator Prompt 以静态为主，但包含 workflow_type 感知段落。
    可选：用 Jinja 动态渲染"工作流类型感知"章节。
    """
    # 方案 A：简单字符串替换
    raw = load_prompt("arbitrator.md")
    return raw.replace("{{ workflow_type }}", workflow_type)

    # 方案 B（推荐）：若改为 .md.j2 文件
    # template = _jinja_env.get_template("arbitrator.md.j2")
    # return template.render(workflow_type=workflow_type)


def load_static_planner_prompt(max_step_num: int = 5) -> str:
    """
    加载静态全量 Planner Prompt（含 A/B/C/D 全部四套策略）。
    仅在 workflow_confidence='low' 时作为降级方案使用。
    """
    raw = load_prompt("planner_static_full.md")
    return raw.replace("{{ max_step_num }}", str(max_step_num)).replace(
        "{{ CURRENT_TIME }}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def load_prompt(filename: str) -> str:
    """加载静态 Prompt 文件。"""
    filepath = _PROMPT_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    return filepath.read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════
#  Plan Structure Validation (Layer 2)
# ═══════════════════════════════════════════════════

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

        # 检查维度清单是否结构一致（标的数量相同）
        target_counts = []
        for s in research_steps:
            desc = getattr(s, "description", "")
            targets = _extract_targets_from_description(desc)
            target_counts.append(len(targets))

        if len(set(target_counts)) > 1:
            return False, (
                f"Workflow B: dimension counts not aligned across research steps: "
                f"{target_counts}"
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
            return False, "Workflow D: missing negative/exception clause search step"

        # missing_conditions 空值为 soft warning（用户可能提供了全部条件）
        missing = getattr(plan, "missing_conditions", [])
        if not missing:
            logger.warning(
                "Workflow D: missing_conditions is empty. "
                "This may be correct if user provided all conditions."
            )

    # ── 工作流 A：无特殊结构要求 ──
    # elif workflow_type == "A": pass

    return True, ""


# ═══════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════

def get_research_steps(plan: Any) -> list:
    """从 Plan 中提取所有 research 类型的步骤。"""
    if plan is None:
        return []
    steps = getattr(plan, "steps", [])
    return [s for s in steps if getattr(s, "step_type", "") == "research"]


def get_analysis_step(plan: Any):
    """从 Plan 中提取 analysis 步骤（应为最后一步且唯一）。"""
    if plan is None:
        return None
    steps = getattr(plan, "steps", [])
    for s in steps:
        if getattr(s, "step_type", "") == "analysis":
            return s
    return None


def generate_searcher_summary(
    step_title: str,
    step_description: str,
    curator_output: dict,
) -> str:
    """
    从 Curator 输出中生成精简摘要（供后续 Searcher 参考）。

    摘要格式：
    - 检索主题：{title}
    - 已覆盖标的：[...]
    - 未覆盖标的：[...]
    - 已发现文档：[文档名: 主题概括]
    - 待追踪线索：[...]
    """
    # 从 Curator 输出中提取覆盖度信息
    coverage = curator_output.get("coverage", {})
    covered = coverage.get("covered_targets", [])
    uncovered = coverage.get("uncovered_targets", [])

    # 从 Curator 输出中提取文档列表
    evidences = curator_output.get("evidences", [])
    doc_list = []
    for ev in evidences:
        source = ev.get("source", "未知来源")
        summary = ev.get("summary", "")
        doc_list.append(f"  - {source}: {summary[:50]}")

    # 提取待追踪线索
    leads = coverage.get("leads", [])

    lines = [
        f"### 步骤摘要：{step_title}",
        f"- **检索内容**：{step_description}",
        f"- **已覆盖标的**：{', '.join(covered) if covered else '无'}",
        f"- **未覆盖标的**：{', '.join(uncovered) if uncovered else '无'}",
        f"- **已发现文档**：",
    ]
    if doc_list:
        lines.extend(doc_list)
    else:
        lines.append("  - 无")
    lines += [
        f"- **待追踪线索**：{', '.join(leads) if leads else '无'}",
    ]

    return "\n".join(lines)


def _extract_targets_from_description(description: str) -> list[str]:
    """
    从 Step description 中按分号拆分出标的列表。

    示例：
      "'中信银行白金卡'的年费标准；年费减免条件；年费收取时间"
      → ["年费标准", "年费减免条件", "年费收取时间"]
    """
    if not description:
        return []

    # 尝试用中文分号分割
    parts = re.split(r"[；;]", description)

    # 清理：去除首段（通常是实体+业务名，不是标的）
    targets = []
    for part in parts:
        part = part.strip().rstrip("。.")
        if part:
            targets.append(part)

    return targets