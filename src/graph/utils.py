from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config.report_style import ReportStyle

RISK_LEVEL_CN = {
    "high": "高风险",
    "medium": "中风险",
    "low": "需人工复核",
}

RISK_LEVEL_SORT_KEY = {"high": 0, "medium": 1, "low": 2}


@dataclass
class ParsedEvidence:
    """从 evidence_summary 字段中拆分出的三部分"""
    material_excerpt: str = ""
    internal_citation: str = ""
    external_citation: str = ""


@dataclass
class FormattedFinding:
    """格式化后的单条风险发现"""
    index: int
    review_point_name: str
    risk_level_cn: str
    material_excerpt: str
    internal_citation: str
    external_citation: str
    description: str
    recommendation: str


@dataclass
class ReporterInput:
    """Reporter 的完整输入包"""
    stats_section: str
    findings_section: str
    compliance_section: str

    def to_prompt_text(self) -> str:
        """组装为 ===== 分隔的最终文本"""
        return "\n".join([
            self.stats_section,
            self.findings_section,
            self.compliance_section
        ])


ASSISTANT_SPEAKER_NAMES = {
    "coordinator",
    "planner",
    "researcher",
    "curator",
    "rule_splitter",
    "arbitrator",
    "reporter",
    "background_investigator",
}

WORKFLOW_TYPE_LABEL: dict[str, str] = {
    "A": "定点调查",
    "B": "并行对比",
    "C": "扫描穷举",
    "D": "条件推理",
}

CONFIDENCE_DESCRIPTION: dict[str, str] = {
    "sufficient": "核心结论均有充分非过期直接依据支撑，链条完整",
    "partial": (
        "部分核心结论依赖间接依据、有条件分支未解决、"
        "部分结论仅有过期依据支撑或原始问题范围未完整覆盖"
    ),
    "insufficient": "核心结论缺乏非过期直接依据或存在关键断点",
}

_NOT_APPLICABLE = "（不适用）"


def _join_list(items: list[str], fallback: str = "（无）") -> str:
    """将字符串列表用顿号连接。处理空列表和“不适用”标记。"""
    if not items:
        return fallback
    if items == [_NOT_APPLICABLE]:
        return "（不适用——原始问题不涉及多个子类型）"
    return "、".join(items)


CP_REVIEW_POINTS = """# 信用卡消保审查要点(略)"""


def get_message_content(message: Any) -> str:
    """Extract message content from dict or LangChain message."""
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def is_user_message(message: Any) -> bool:
    """Return True if the message originated from the end user."""
    if isinstance(message, dict):
        role = (message.get("role") or "").lower()
        if role in {"user", "human"}:
            return True
        if role in {"assistant", "system"}:
            return False
        name = (message.get("name") or "").lower()
        if name and name in ASSISTANT_SPEAKER_NAMES:
            return False
        return role == "" and name not in ASSISTANT_SPEAKER_NAMES

    message_type = (getattr(message, "type", "") or "").lower()
    name = (getattr(message, "name", "") or "").lower()
    if message_type == "human":
        return not (name and name in ASSISTANT_SPEAKER_NAMES)

    role_attr = getattr(message, "role", None)
    if isinstance(role_attr, str) and role_attr.lower() in {"user", "human"}:
        return True

    additional_role = getattr(message, "additional_kwargs", {}).get("role")
    if isinstance(additional_role, str) and additional_role.lower() in {
        "user",
        "human",
    }:
        return True

    return False


def get_latest_user_message(messages: list[Any]) -> tuple[Any, str]:
    """Return the latest user-authored message and its content."""
    for message in reversed(messages or []):
        if is_user_message(message):
            content = get_message_content(message)
            if content:
                return message, content
    return None, ""


def build_clarified_topic_from_history(
    clarification_history: list[str],
) -> tuple[str, list[str]]:
    """Construct clarified topic string from an ordered clarification history."""
    sequence = [item for item in clarification_history if item]
    if not sequence:
        return "", []
    if len(sequence) == 1:
        return sequence[0], sequence
    head, *tail = sequence
    clarified_string = f"{head} - {', '.join(tail)}"
    return clarified_string, sequence


def reconstruct_clarification_history(
    messages: list[Any],
    fallback_history: list[str] | None = None,
    base_topic: str = "",
) -> list[str]:
    sequence: list[str] = []
    for message in messages or []:
        if not is_user_message(message):
            continue
        content = get_message_content(message)
        if not content:
            continue
        if sequence and sequence[-1] == content:
            continue
        sequence.append(content)

    if sequence:
        return sequence

    fallback = [item for item in (fallback_history or []) if item]
    if fallback:
        return fallback

    base_topic = (base_topic or "").strip()
    return [base_topic] if base_topic else []


ATTACHED_TEXT_MAX_CHARS = 50_000


def format_attached_files_for_prompt(
    attached_files: list[dict], max_chars_per_file: int = ATTACHED_TEXT_MAX_CHARS
) -> str:
    sections: list[str] = []
    idx = 1
    for f in attached_files:
        name = f.get("name", "unnamed")
        kind = f.get("kind")
        if kind == "text":
            content = f.get("text") or ""
            truncated = False
            if len(content) > max_chars_per_file:
                content = content[:max_chars_per_file]
                truncated = True
            suffix = (
                f"\n\n...[truncated, original {f.get('size_bytes', 0)} bytes]"
                if truncated
                else ""
            )
            sections.append(f"## 附件{idx}: {name}\n\n{content}{suffix}")
            idx += 1
        else:
            sections.append(f"- [unknown attachment: {name}]")

    return "\n\n".join(sections)


def get_attached_images(state: Any) -> list[dict[str, str]]:
    """Return image attachments shaped for multimodal LangChain messages.

    Each entry: {"name", "mime", "b64"}. Empty list when there are no
    image attachments.
    """
    files = (state or {}).get("attached_files") or []
    out: list[dict[str, str]] = []
    for f in files:
        if f.get("kind") != "image":
            continue
        b64 = f.get("b64")
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


def _parse_evidence_summary(summary: str) -> dict[str, str]:
    result = {"material_excerpt": "", "internal_rule": "", "external_rule": ""}
    if not summary:
        return result

    parts = [p.strip() for p in summary.split("|")]
    for part in parts:
        if part.startswith("【涉及文案】"):
            result["material_excerpt"] = part[len("【涉及文案】"):].strip()
        elif part.startswith("【行内依据】"):
            result["internal_rule"] = part[len("【行内依据】"):].strip()
        elif part.startswith("【外部依据】"):
            result["external_rule"] = part[len("【外部依据】"):].strip()

    # 兜底: 无标签则整段视为涉及文案
    if not any(result.values()):
        result["material_excerpt"] = summary.strip()

    return result


def _extract_high_risk_domains(findings: list[dict[str, Any]]) -> str:
    """提取高风险项对应的审查点名称（去重），用顿号连接。"""
    seen: list[str] = []
    for f in findings:
        if f.get("risk_level") == "high":
            name = f.get("point_name", "")
            if name and name not in seen:
                seen.append(name)
    return "、".join(seen) if seen else "无高风险项"


def prepare_reporter_input(
    analyst_output: dict[str, Any], report_style: str, workflow_type: str
) -> str:
    if report_style == ReportStyle.BANK_BUSINESS_ANALYSIS.value:
        return prepare_bi_reporter_input(analyst_output, workflow_type)
    elif report_style == ReportStyle.CUSTOMER_RIGHTS_PROTECTION_REVIEW.value:
        return prepare_cp_reporter_input(analyst_output)
    else:
        return ""


def _render_scope_coverage(scope: dict[str, Any]) -> str:
    """将 scope_coverage 渲染为可读文本。"""
    if not scope:
        return "（无调查覆盖范围数据）"

    full_scope: list[str] = scope.get("full_scope", [])
    covered: list[str] = scope.get("covered", [])
    not_covered: list[str] = scope.get("not_covered", [])

    lines: list[str] = [
        f"原始问题完整范围: {_join_list(full_scope)}",
        f"本次调查已覆盖: {_join_list(covered)}",
    ]

    if not_covered and not_covered != [_NOT_APPLICABLE]:
        lines.append(f"本次调查未覆盖: {_join_list(not_covered)}")
    else:
        lines.append("本次调查未覆盖: 无 (已完整覆盖)")

    return "\n".join(lines)


def _render_conclusions(conclusions: list[dict[str, Any]]) -> str:
    """将 conclusions 数组渲染为可读文本。"""
    if not conclusions:
        return "（无结论）"

    lines: list[str] = []
    for i, c in enumerate(conclusions, 1):
        lines.append(f"结论 {i}:")
        lines.append(f"  陈述: {c.get('statement', '')}")
        lines.append(f"  置信度: {c.get('confidence', '未知')}")

        # 支撑来源
        sources = c.get("supporting_sources", [])
        if sources:
            lines.append(f"  支撑来源: {', '.join(f'《{s}》' for s in sources)}")
        else:
            lines.append("  支撑来源: (无)")

        # 条件前提
        conditions = c.get("conditions")
        if conditions:
            lines.append(f"  条件前提: {'; '.join(conditions)}")
        else:
            lines.append("  条件前提: 无")

        # 范围说明
        scope_note = c.get("scope_note")
        if scope_note:
            lines.append(f"  范围说明: {scope_note}")

        # 过期依据标记
        if c.get("expired_only", False):
            lines.append(
                "  ⚠️ 时效性警告: 该结论仅由已过期依据支撑，当前规则可能已变更"
            )

        lines.append("")

    return "\n".join(lines)


def _render_comparison_table(table: dict[str, Any]) -> str:
    """将 comparison_table 渲染为 Markdown 表格。"""
    if not table:
        return "（无对比数据）"

    dimensions: list[str] = table.get("dimensions", [])
    objects: dict[str, dict] = table.get("objects", {})
    key_differences: list[str] = table.get("key_differences", [])

    obj_names = list(objects.keys())
    if not dimensions or not obj_names:
        return "（对比表格数据不完整）"

    lines: list[str] = []

    # 表头
    lines.append("| 对比维度 | " + " | ".join(obj_names) + " |")
    lines.append("|:---|" + "|".join([":---"] * len(obj_names)) + "|")

    # 表体
    for dim in dimensions:
        row = f"| {dim} |"
        for obj_name in obj_names:
            val = objects.get(obj_name, {}).get(dim, "知识库中未查到")
            row += f" {val} |"
        lines.append(row)

    lines.append("")

    # 关键差异
    if key_differences:
        lines.append("关键差异:")
        for i, diff in enumerate(key_differences, 1):
            lines.append(f"  {i}. {diff}")

    return "\n".join(lines)


def _render_enumeration_list(enum_list: dict[str, Any]) -> str:
    """将 enumeration_list 渲染为可读文本。"""
    if not enum_list:
        return "（无穷举数据）"

    lines: list[str] = []

    condition = enum_list.get("condition", "")
    if condition:
        lines.append(f"穷举条件: {condition}")
        lines.append("")

    # 已确认项目
    confirmed = enum_list.get("confirmed_items", [])
    if confirmed:
        lines.append(f"已确认项目（共 {len(confirmed)} 项）:")
        for i, item in enumerate(confirmed, 1):
            name = item.get("name", "未知")
            source = item.get("source", "")
            source_str = f"（来源: {source}）" if source else ""
            lines.append(f"  {i}. {name}{source_str}")
        lines.append("")

    # 存疑项目
    uncertain = enum_list.get("uncertain_items", [])
    if uncertain:
        lines.append(f"存疑项目（共 {len(uncertain)} 项）:")
        for i, item in enumerate(uncertain, 1):
            name = item.get("name", "未知")
            reason = item.get("reason", "")
            lines.append(f"  {i}. {name} - 存疑原因: {reason}")
        lines.append("")

    # 来源已过期项目
    expired_items = enum_list.get("expired_source_items", [])
    if expired_items:
        lines.append(f"来源已过期项目（共 {len(expired_items)} 项）:")
        for i, item in enumerate(expired_items, 1):
            name = item.get("name", "未知")
            source = item.get("source", "")
            note = item.get("note", "")
            source_str = f"（来源: {source}）" if source else ""
            note_str = f" - {note}" if note else ""
            lines.append(f"  {i}. {name}{source_str}{note_str}")
        lines.append("")

    note = enum_list.get("completeness_note", "")
    if note:
        lines.append(f"完备性说明: {note}")

    return "\n".join(lines)


def _render_condition_chain(chain: dict[str, Any]) -> str:
    """将 condition_chain 渲染为可读文本。"""
    if not chain:
        return "（无条件推理数据）"

    lines: list[str] = []

    target = chain.get("target", "")
    if target:
        lines.append(f"判定目标: {target}")
        lines.append("")

    conditions = chain.get("conditions", [])
    if conditions:
        status_map = {
            "known": "已知",
            "unknown_need_user": "未知（需用户确认）",
            "unknown_need_system": "未知（需系统查询）",
            "inferred": "已知（推断）",
        }
        lines.append("条件清单:")
        lines.append("| 条件 | 状态 | 值 |")
        lines.append("|---|---|---|")
        for cond in conditions:
            name = cond.get("name", "")
            status = cond.get("status", "")
            value = cond.get("value") or "-"
            status_label = status_map.get(status, status)
            lines.append(f"| {name} | {status_label} | {value} |")
        lines.append("")

    reasoning = chain.get("reasoning_summary", "")
    if reasoning:
        lines.append(f"推理概述: {reasoning}")
        lines.append("")

    negative_rules = chain.get("negative_rules", [])
    if negative_rules:
        lines.append("否定性规则:")
        for rule in negative_rules:
            lines.append(f"  - {rule}")

    return "\n".join(lines)


def format_compliance_section(observations: list[str]) -> str:
    if not observations:
        return "无合规项记录。"

    return "### 合规项\n" + "\n".join(f"- {obs}" for obs in observations)


def compute_stats(
    risk_findings: list[dict[str, Any]],
    observations: list[str],
    confidence_score: float,
) -> str:
    """生成审查统计文本"""
    high = sum(1 for f in risk_findings if f.get("risk_level") == "high")
    medium = sum(1 for f in risk_findings if f.get("risk_level") == "medium")
    low = sum(1 for f in risk_findings if f.get("risk_level") == "low")
    total = len(risk_findings)

    # 合规项：observations 中包含"合规"且不含"记录缺失"的条目
    compliance_items = [
        obs for obs in observations
        if "合规" in obs and "记录缺失" not in obs
    ]
    compliance_count = len(compliance_items)

    # 主要风险领域：从 findings 的 review_point_name 提取顶层名称（去重保序）
    areas: list[str] = []
    for f in risk_findings:
        name = f.get("review_point_name", "")
        top_level = name.split("—")[0].strip()
        if top_level and top_level not in areas:
            areas.append(top_level)
    risk_areas = "、".join(areas[:5]) if areas else "无"

    return (
        f"### 审查统计\n风险项总计：{total}\n"
        f"高风险：{high}\n"
        f"中风险：{medium}\n"
        f"需人工复核：{low}\n"
        f"合规项：{compliance_count}\n"
        f"整体置信度：{confidence_score}\n"
        f"主要风险领域：{risk_areas}"
    )


def sort_and_number_findings(
    risk_findings: list[dict[str, Any]],
) -> list[FormattedFinding]:
    """按风险等级排序（high > medium > low）并编号。
    同等级内保持原始顺序。
    """
    sorted_findings = sorted(
        risk_findings,
        key=lambda f: RISK_LEVEL_SORT_KEY.get(f.get("risk_level", "low"), 99),
    )
    result = []
    for idx, finding in enumerate(sorted_findings, start=1):
        evidence = parse_evidence_summary(finding.get("evidence_summary", ""))
        result.append(
            FormattedFinding(
                index=idx,
                review_point_name=finding.get("review_point_name", "未知审查点"),
                risk_level_cn=RISK_LEVEL_CN.get(
                    finding.get("risk_level", "low"), "需人工复核"
                ),
                material_excerpt=evidence.material_excerpt or "（未提供）",
                internal_citation=evidence.internal_citation or "（无）",
                external_citation=evidence.external_citation or "（无）",
                description=finding.get("description", ""),
                recommendation=finding.get("recommendation", ""),
            )
        )
    return result


def format_finding_block(f: FormattedFinding) -> str:
    """将单条 FormattedFinding 格式化为文本块"""
    return (
        f"【{f.index}】\n"
        f"审查点名称: {f.review_point_name}\n"
        f"风险等级: {f.risk_level_cn}\n"
        f"涉及文案: {f.material_excerpt}\n"
        f"行内依据: {f.internal_citation}\n"
        f"外部依据: {f.external_citation}\n"
        f"风险描述: {f.description}\n"
        f"整改建议: {f.recommendation}"
    )


def parse_evidence_summary(summary: str) -> ParsedEvidence:
    result = ParsedEvidence()
    if not summary:
        return result

    parts = [p.strip() for p in summary.split("|")]
    for part in parts:
        if part.startswith("【涉及文案】"):
            result.material_excerpt = part[len("【涉及文案】"):].strip()
        elif part.startswith("【行内依据】"):
            result.internal_citation = part[len("【行内依据】"):].strip()
        elif part.startswith("【外部依据】"):
            result.external_citation = part[len("【外部依据】"):].strip()

    return result


def _render_comparison_table_md(table: dict[str, Any]) -> str:
    dims = [str(d) for d in (table.get("dimensions") or [])]
    objects = table.get("objects") or {}
    if not dims or not isinstance(objects, dict) or not objects:
        return ""

    obj_names = [str(k) for k in objects.keys()]
    lines = [
        "| 对比维度 | " + " | ".join(obj_names) + " |",
        "|:---| " + "|".join([":---"] * len(obj_names)) + "|",
    ]
    for i, dim in enumerate(dims):
        row = [dim]
        for name in obj_names:
            vals = objects.get(name)
            cell = ""
            if isinstance(vals, dict):
                cell = str(vals.get(dim, "知识库中未查到"))
            elif isinstance(vals, list):
                cell = str(vals[i]) if i < len(vals) else "知识库中未查到"
            elif vals is not None:
                cell = str(vals)
            row.append(cell.replace("\n", " ").replace("|", "/") or "知识库中未查到")
        lines.append("| " + " | ".join(row) + " |")

    diffs = table.get("key_differences") or []
    if diffs:
        lines.append("")
        lines.append("**关键差异**:")
        for d in diffs:
            lines.append(f"- {d}")

    return "\n".join(lines)


def prepare_bi_reporter_input(
    parsed: dict[str, Any], wf: str
) -> str:
    parts: list[str] = []

    sc = parsed.get("scope_coverage") or {}
    if isinstance(sc, dict) and sc:
        parts.append(
            f"### 调查覆盖范围\n"
            f"- 完整范围（full_scope）: {'、'.join(map(str, sc.get('full_scope') or [])) or '（不适用）'}\n"
            f"- 已覆盖（covered）: {'、'.join(map(str, sc.get('covered') or [])) or '（不适用）'}\n"
            f"- 未覆盖（not_covered）: {'、'.join(map(str, sc.get('not_covered') or [])) or '无'}"
        )

    conf = parsed.get("overall_confidence")
    if conf:
        parts.append(f"### 整体置信度\n{conf}")

    unc = parsed.get("uncovered_aspects") or []
    if unc:
        parts.append("### 未覆盖方面\n" + "\n".join(f"- {u}" for u in unc))

    conclusions = parsed.get("conclusions") or []
    if conclusions:
        c_lines = ["### 业务结论"]
        for i, c in enumerate(conclusions, 1):
            if not isinstance(c, dict):
                c_lines.append(f"{i}. {c}")
                continue
            line = f"{i}. {c.get('statement', '')} (置信度: {c.get('confidence', '')})"
            srcs = c.get("supporting_sources") or []
            if srcs:
                line += f"\n    - 支撑来源: {'、'.join(f'《{s}》' for s in srcs)}"
            conds = c.get("conditions")
            if conds:
                line += f"\n    - 条件前提: {'；'.join(map(str, conds))}"
            if c.get("scope_note"):
                line += f"\n    - 范围说明: {c['scope_note']}"
            if c.get("expired_only"):
                line += "\n    - ⚠️ 该结论仅基于已过期依据，仅供历史参考"
            c_lines.append(line)
        parts.append("\n".join(c_lines))

    # —— 工作流特定结构 ——
    if wf == "B":
        table = parsed.get("comparison_table")
        if isinstance(table, dict):
            md = _render_comparison_table_md(table)
            if md:
                parts.append("### 对比表格（已渲染）\n" + md)
    elif wf == "C":
        enum = parsed.get("enumeration_list")
        if isinstance(enum, dict) and enum:
            e_lines = ["### 穷举列表"]
            if enum.get("condition"):
                e_lines.append(f"- 穷举条件: {enum['condition']}")
            for label, key in (
                ("已确认项目", "confirmed_items"),
                ("存疑项目", "uncertain_items"),
                ("仅过期来源项目", "expired_source_items"),
            ):
                items = enum.get(key) or []
                if items:
                    e_lines.append(f"- {label}: ")
                    e_lines.extend(f"  - {it}" for it in items)
            if enum.get("completeness_note"):
                e_lines.append(f"- 完备性说明: {enum['completeness_note']}")
            parts.append("\n".join(e_lines))
    elif wf == "D":
        chain = parsed.get("condition_chain")
        if isinstance(chain, dict) and chain:
            d_lines = ["### 条件推理链"]
            if chain.get("target"):
                d_lines.append(f"- 判定目标: {chain['target']}")
            conds = chain.get("conditions") or []
            if conds:
                d_lines.append("- 条件清单: ")
                d_lines.extend(f"  - {c}" for c in conds)
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


def prepare_cp_reporter_input(
    analyst_output: dict[str, Any]
) -> str:
    risk_findings = analyst_output.get("risk_findings", [])
    observations = analyst_output.get("observations", [])
    confidence = analyst_output.get("confidence_score", 0.0)

    # 1. 审查统计
    stats = compute_stats(risk_findings, observations, confidence)

    # 2. 风险发现（排序 + 编号 + 格式化）
    formatted = sort_and_number_findings(risk_findings)
    findings_text = "### 风险发现\n"
    findings_text += (
        "\n\n".join(format_finding_block(f) for f in formatted)
        if formatted
        else "无风险发现。"
    )

    # 3. 合规项
    compliance_text = format_compliance_section(observations)

    return ReporterInput(
        stats_section=stats,
        findings_section=findings_text,
        compliance_section=compliance_text
    ).to_prompt_text()
