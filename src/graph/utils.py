from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.config.report_style import (
    REPORT_STYLE_KEY,
    RISK_LEVEL_CN,
    RISK_LEVEL_ORDER,
)

ASSISTANT_SPEAKER_NAMES = {
    "coordinator": "coordinator",
    "planner": "planner",
    "researcher": "researcher",
    "curator": "curator",
    "rule_splitter": "rule_splitter",
    "arbitrator": "arbitrator",
    "reporter": "reporter",
    "background_investigator": "background_investigator",
}

WORKFLOW_TYPE_LABEL: dict[str, str] = {
    "A": "设定框架",
    "B": "并行对比",
    "C": "清单穷举",
}

CONFIDENCE_DESCRIPTION: dict[str, str] = {
    "sufficient": "核心结论均有充分并直接支撑, 证据链条完整。",
    "partial": "部分核心结论依赖间接依据, 有条件分支未解决。",
    "insufficient": "部分结论基于过期依据支撑或存在问题未完整覆盖。",
}

NOT_APPLICABLE = "（不适用）"


def _join_list(items: list) -> str:
    """将列表渲染为中文顿号分隔文本"""
    if not items:
        return "（无）"
    return "、".join(str(i) for i in items)


def _extract_message_content(message: Any) -> str:
    """Return the text content of a message."""
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return ""


def _is_user_message(message: Any) -> bool:
    """Return True if the message originated from the end user."""
    if isinstance(message, dict):
        role = message.get("role") or ""
    else:
        role = getattr(message, "role", "") or ""
    if role in ("user", "human"):
        return True
    if role in ("assistant", "system"):
        return False
    name = getattr(message, "name", "") or ""
    return name not in ASSISTANT_SPEAKER_NAMES


def _get_last_user_message(messages: list[Any]) -> tuple[Any, str]:
    """Return the latest user-authored message and its content."""
    for message in reversed(messages):
        if _is_user_message(message):
            return message, _extract_message_content(message)
    return None, ""


def _flatten_history(messages: list[Any]) -> list[str]:
    """Flatten message history into readable content strings."""
    sequence: list[str] = []
    for message in messages:
        if not message:
            continue
        content = _extract_message_content(message)
        if not content:
            continue
        sequence.append(content)
    return sequence


def build_clarified_topic_from_history(
    clarification_history: list[dict],
) -> tuple[str, list]:
    """
    Extract clarified topic string from an ordered clarification history.

    Returns:
        (topic, followups)
    """
    sequence: list = []
    for item in clarification_history:
        if not isinstance(item, dict):
            continue
        if len(sequence) == 0:
            sequence.append(item)
    if not sequence:
        return "", []
    topic = _extract_message_content(sequence[0]).strip()
    followups = [_extract_message_content(m) for m in sequence[1:]]
    return topic, followups


def reconstruct_clarification_history(
    history: list[Any],
) -> list[str]:
    fallback_history = _flatten_history(history or [])
    if fallback_history:
        return fallback_history
    return []


ATTACHED_TEXT_MAX_CHARS = 50_000


def summarize_evidence_section(
    state: dict,
    max_chars_per_file: int = ATTACHED_TEXT_MAX_CHARS,
) -> str:
    idx = 1
    sections: list[str] = []
    for f in state.get("attached_files") or []:
        if not isinstance(f, dict):
            continue
        kind = f.get("kind") or ""
        name = f.get("name", "unnamed")
        if kind == "text":
            content = f.get("text") or ""
            truncated = False
            if len(content) > max_chars_per_file:
                content = content[:max_chars_per_file]
                truncated = True
            suffix = (
                f"\n...(truncated, original {f.get('size_bytes', 0)} bytes)"
                if truncated
                else ""
            )
            sections.append(f"### 附件[{idx}]: {name}\n{content}{suffix}")
            idx += 1
        else:
            sections.append(f"- [非文本附件: {name}]")
    return "\n".join(sections)


def _extract_images(state: dict) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for f in state.get("attached_files") or []:
        if f.get("kind") != "image":
            continue
        b64 = f.get("b64", "")
        if not b64:
            continue
        out.append(
            {
                "name": f.get("name", "unnamed"),
                "mime": f.get("mime", "application/octet-stream"),
                "b64": b64,
            }
        )
    return out


def parse_evidence_summary(summary: str) -> dict[str, str]:
    result = {
        "material_except": "",
        "internal_rule": "",
        "external_rule": "",
    }
    if not summary:
        return result
    parts = [p.strip() for p in summary.split("。")]
    for part in parts:
        if part.startswith("[涉及文章]"):
            result["material_except"] = part[len("[涉及文章]"):].strip()
        elif part.startswith("[行内依据]"):
            result["internal_rule"] = part[len("[行内依据]"):].strip()
        elif part.startswith("[外部依据]"):
            result["external_rule"] = part[len("[外部依据]"):].strip()
    # 判定: 无法简单模板识别=涉及文章
    if not any(result.values()):
        result["material_except"] = summary.strip()
    return result


@dataclass
class FlowEvidence:
    """
    证据摘要解析结果
    """

    review_point: str = ""
    internal_citation: str = ""
    external_citation: str = ""


@dataclass
class FormattedFinding:
    """格式化后的事单风险发现"""

    index: int
    review_point_name: str
    risk_level: str
    material_excerpt: str
    recommendation: str
    internal_citation: str
    external_citation: str
    description: str


@dataclass
class ReporterInput:
    """Reporter 的完整输入"""

    stats_section: str
    findings_section: str
    compliance_section: str

    def to_prompt_text(self) -> str:
        """组装为 === 分隔的最终文本"""
        return "\n\n".join(
            [
                f"=== 审查统计 ===\n{self.stats_section}",
                f"=== 风险发现 ===\n{self.findings_section}",
                f"=== 合规审查 ===\n{self.compliance_section}",
            ]
        )


def render_conclusions(conclusions: dict[str, Any]) -> str:
    """conclusions 数据渲染为可读文本, 为空则提示缺失"""
    if not conclusions:
        return "（无结论）\n"
    lines: list[str] = []
    for i, c in enumerate(conclusions, 1):
        lines.append(f"结论 {i}:")
        lines.append(f"  陈述: {c.get('statement', '')}")
        lines.append(f"  置信度: {c.get('confidence', '')}")
        # 支撑来源
        sources = c.get("supporting_sources", [])
        if sources:
            lines.append(f"  支撑来源: {', '.join(f'{s}' for s in sources)}")
        # 条件前提
        conds = c.get("conditions") or []
        if conds:
            lines.append(f"  条件前提: {'; '.join(map(str, conds))}")
        # 范围说明: {scope_note} (已渲染, 仅供历史参考)
        if c.get("scope_note"):
            lines.append(f"  范围说明: {c['scope_note']}")
        if c.get("expired_only"):
            lines.append("  ⚠️ 该结论仅基于已过期依据, 仅供历史参考")
    return "\n".join(lines)


def compute_stats(
    risk_findings: list[dict[str, Any]],
    observations: list,
    confidence: float,
) -> str:
    """生成审查统计文本"""
    high = sum(1 for f in risk_findings if f.get("risk_level") == "high")
    medium = sum(1 for f in risk_findings if f.get("risk_level") == "medium")
    low = sum(1 for f in risk_findings if f.get("risk_level") == "low")
    total = len(risk_findings)
    # 合规项: observations
    compliance_items = [
        obs
        for obs in observations
        if "合规" in obs and "已缺失" not in obs
    ]
    compliance_count = len(compliance_items)
    # 主要风险领域: 从 findings 的
    # review_point_name 提取顶层名称 (去重保序)
    areas: list[str] = []
    for f in risk_findings:
        name = f.get("review_point_name", "")
        top_level = name.split("—")[0].strip()
        if top_level and top_level not in areas:
            areas.append(top_level)
    risk_areas = ", ".join(areas) if areas else "无"
    return (
        f"### 审查统计\n风险总计: {total}\n"
        f"- 高风险: {high}\n中风险: {medium}\n"
        f"- 需人工复核: {low}\n"
        f"- 合规项: {compliance_count}\n"
        f"- 主要风险领域: {risk_areas}"
    )


def format_comparison_table(table: dict[str, Any]) -> str:
    """渲染 Markdown 对比表格"""
    if not table or not isinstance(table, dict):
        return ""
    dimensions: list[str] = table.get("dimensions", [])
    obj_names: list = table.get("objects", [])
    key_differences: list[str] = table.get("key_differences", [])
    if not dimensions or not obj_names:
        return ""
    lines: list[str] = []
    header = "| 对比维度 | " + " | ".join(obj_names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(obj_names) + 1))
    for dim in dimensions:
        row = f"| {dim} "
        for name in obj_names:
            vals = dim.get(name) if isinstance(dim, dict) else None
            cell = str(vals) if vals else "知识库中未查到"
            row += f"| {cell} "
        row += "|"
        lines.append(row)
    if key_differences:
        lines.append("")
        lines.append("**关键差异**:")
        for d in key_differences:
            lines.append(f"- {d}")
    return "\n".join(lines)


def sort_and_number_findings(
    risk_findings: list[dict[str, Any]],
) -> list[FormattedFinding]:
    """
    按风险等级排序 (high > medium > low 并编号)
    """
    sorted_findings = sorted(
        risk_findings,
        key=lambda f: RISK_LEVEL_ORDER.get(f.get("risk_level", "low"), 99),
    )
    result = []
    for idx, finding in enumerate(sorted_findings, start=1):
        evidence = parse_evidence_summary(
            finding.get("evidence_summary", ""))
        result.append(FormattedFinding(
            index=idx,
            review_point_name=finding.get("review_point", "未指明审查点"),
            risk_level=RISK_LEVEL_CN.get(
                finding.get("risk_level", "low"), "需人工复核"),
            material_excerpt=evidence.get("material_except") or "（无）",
            internal_citation=evidence.get("internal_rule") or "（无）",
            external_citation=evidence.get("external_rule") or "（无）",
            description=finding.get("description", ""),
            recommendation=finding.get("recommendation", ""),
        ))
    return result


def format_finding_block(f: FormattedFinding) -> str:
    return (
        f"### [{f.index}] {f.review_point_name} [{f.risk_level}]\n"
        f"- 涉及条款: {f.material_excerpt}\n"
        f"- 行内依据: {f.internal_citation}\n"
        f"- 外部依据: {f.external_citation}\n"
        f"- 风险描述: {f.description}\n"
        f"- 整改建议: {f.recommendation}"
    )


def render_enumeration_list(enum_list: dict[str, Any]) -> str:
    """无完整数据列表"""
    items = enum_list.get("items") or []
    if not items:
        return "（无列表项）"
    lines: list[str] = []
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            conditions = item.get("condition", "")
            if conditions:
                lines.append(f"- 条件{i}: {conditions}")
            confirmed = item.get("confirmed_items") or []
            if confirmed:
                lines.append(
                    f"  - 已确认项目 ({len(confirmed)}): "
                    + ", ".join(map(str, confirmed))
                )
            uncertain = item.get("uncertain_items") or []
            if uncertain:
                lines.append(
                    f"  - 存疑项目 ({len(uncertain)}): "
                    + ", ".join(map(str, uncertain))
                )
        else:
            lines.append(f"- {item}")
    note = enum_list.get("completeness_note")
    if note:
        lines.append(f"- 完整性说明: {note}")
    return "\n".join(lines)


def render_scope_coverage(scope: dict) -> str:
    covered = scope.get("covered", [])
    not_covered = scope.get("not_covered", [])
    if not covered and not not_covered:
        return "（无覆盖信息）"
    lines = []
    if covered:
        lines.append("### 已覆盖")
        lines.extend(f"- {item}" for item in covered)
    if not_covered:
        lines.append("### 未覆盖")
        lines.extend(f"- {item}" for item in not_covered)
    return "\n".join(lines)


def prepare_bi_reporter_input(
    parsed: dict[str, Any],
    wf: str,
) -> str:
    parts: list[str] = []
    # 调查覆盖范围
    sc = parsed.get("scope_coverage") or {}

    if isinstance(sc, dict):
        c_parts: list[str] = ["### 调查覆盖范围"]
        covered = sc.get("covered") or []
        not_covered = sc.get("not_covered") or []
        if covered:
            c_parts.append(f"- 已覆盖: {', '.join(map(str, covered))}")
        else:
            c_parts.append("- 已覆盖: （无）")
        if not_covered:
            c_parts.append(f"- 未覆盖: {', '.join(map(str, not_covered))}")
        else:
            c_parts.append("- 未覆盖: （无）")
        conf = parsed.get("overall_confidence")
        if conf:
            c_parts.append(f"### 整体置信度: {conf}")

        parts.append("\n".join(c_parts))

    unc = parsed.get("uncovered_aspects") or []
    if unc:
        parts.append("### 未覆盖点\n" + "\n".join(f"- {u}" for u in unc))
    conclusions = parsed.get("conclusions") or []
    if conclusions:
        c_lines = ["### 业务结论"]
        for i, c in enumerate(conclusions, 1):
            if isinstance(c, dict):
                line = (
                    f"{i}. {c.get('statement', '')} "
                    f"(置信度: {c.get('confidence', '')})"
                )
                srcs = c.get("supporting_sources") or []
                if srcs:
                    line += f"\n  - 支撑来源: {', '.join(f'{s}' for s in srcs)}"
                conds = c.get("conditions") or []
                if conds:
                    line += f"\n  - 条件前提: {'; '.join(map(str, conds))}"
                if c.get("scope_note"):
                    line += f"\n  - 范围说明: {c['scope_note']}"
                if c.get("expired_only"):
                    line += "\n  ⚠️ 该结论仅基于已过期依据, 仅供历史参考"
                c_lines.append(line)
            else:
                c_lines.append(f"{i}. {c}")
        parts.append("\n".join(c_lines))
    # 条件状态表
    conditions = parsed.get("conditions") or []
    if conditions:
        unknown_need_user = "未知 (需用户确认)"
        unknown_need_system = "未知 (需系统查证)"
        label_map = {
            "known": "已知 (可靠)",
            "unknown_need_user": unknown_need_user,
            "unknown_need_system": unknown_need_system,
            "inferred": "已知 (推断)",
        }
        t_lines = ["| 条件 | 状态 | 备注 |", "| --- | --- | --- |"]
        for cond in conditions:
            cond_line = cond.get("condition", "")
            status = (
                cond.get("status")
                or cond.get("value")
                or cond.get("status_label")
                or "—"
            )
            status_label = label_map.get(status, status)
            value = cond.get("value", "")
            t_lines.append(f"| {cond_line} | {status_label} | {value} |")
        parts.append("\n".join(t_lines))
    # — 工作流定位建议 —
    if wf == "B":
        table = parsed.get("comparison_table")
        if isinstance(table, dict) and table:
            md = format_comparison_table(table)
            if md:
                parts.append("### 对比表格 (已渲染)\n" + md)
    elif wf == "C":
        enum = parsed.get("enumeration_list")
        if isinstance(enum, dict) and enum:
            e_lines = ["### 完整列表"]
            items = enum.get("items") or []
            for item in items:
                if isinstance(item, dict):
                    label = (
                        item.get("condition")
                        or item.get("item")
                        or json.dumps(item, ensure_ascii=False)
                    )
                    if item.get("confirmed_items"):
                        e_lines.append(
                            f"- [√] {label} "
                            f"(已确认项目: {len(item['confirmed_items'])})"
                        )
                    elif item.get("uncertain_items"):
                        e_lines.append(
                            f"- [?] {label} "
                            f"(仅疑似项目: {len(item['uncertain_items'])})"
                        )
                    else:
                        e_lines.append(f"- [ ] {label}")
                else:
                    e_lines.append(f"- {item}")
            if enum.get("completeness_note"):
                e_lines.append(f"- 完整性说明: {enum['completeness_note']}")
            parts.append("\n".join(e_lines))
    else:
        chain = parsed.get("condition_chain")
        if isinstance(chain, dict) and chain:
            d_lines = ["### 条件逻辑概览"]
            if chain.get("target"):
                d_lines.append(f"- 判定目标: {chain['target']}")
            conds = chain.get("conditions") or []
            for c in conds:
                d_lines.append(f"- 条件: {c}")
            if chain.get("reasoning_summary"):
                d_lines.append(f"- 推理概述: {chain['reasoning_summary']}")
            negs = chain.get("negative_rules") or []
            if negs:
                d_lines.append("- 否定性规则: ")
                d_lines.extend(f"  - {n}" for n in negs)
            parts.append("\n".join(d_lines))

    for title, key, empty_hint in (
        ("分析过程", "analysis_text", "（无）"),
        ("矛盾信息处理", "contradiction_text", "未发现矛盾信息。"),
        ("风险提示", "risk_text", "未发现需要提示的风险。"),
    ):
        parts.append(f"### {title}\n{parsed.get(key) or empty_hint}")
    return "\n\n".join(parts)


def format_compliance_section(observations: list[str]) -> str:
    if not observations:
        return "无合规项记录。"
    return "### 合规项\n" + "\n".join(f"- {ob}" for ob in observations)


def prepare_cp_reporter_input(
    analyst_output: dict[str, Any],
) -> str:
    risk_findings = analyst_output.get("risk_findings", [])
    observations = analyst_output.get("observations", [])
    confidence = analyst_output.get("confidence_score", 0.0)

    # 1. 审查统计
    stats = compute_stats(risk_findings, observations, confidence)

    # 2. 风险发现 (排序 + 编号 + 格式化)
    formatted = sort_and_number_findings(risk_findings)
    findings_text = (
        "### 风险发现\n"
        + "\n\n".join(format_finding_block(f) for f in formatted)
        if formatted
        else "### 风险发现\n（无）"  # 风险校验为空
    )

    # 3. 合规项
    compliance_text = format_compliance_section(observations)

    return ReporterInput(
        stats_section=stats,
        findings_section=findings_text,
        compliance_section=compliance_text,
    ).to_prompt_text()
