"""
视图构建器：从 CuratorOutput 生成面向不同下游消费者的 Markdown 视图
在 LangGraph 的 State 转换 / 消息构建层调用，不在 Agent 提示词中
"""

from curator_models import CuratorOutput, EvidenceItem, Relevance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _format_evidence_full(ev: EvidenceItem) -> str:
    """完整视图：包含全部字段 + 具体内容"""
    merged_note = f"（合并自 Searcher 片段 {', '.join(map(str, ev.merged_from))}）" if ev.merged_from else ""
    lines = [
        f"**业务依据 {ev.id}**{merged_note}",
        f"- *来源*：{ev.source}",
        f"- *相关性*：{ev.relevance.value} — {ev.relevance_reason}",
        f"- *信息完整性*：{ev.completeness}",
        f"- *信息质量*：{'、'.join(ev.quality_tags)}",
        f"- *内容类型*：{ev.content_type.value}",
        f"- *是否包含条件分支*：{'是' if ev.has_conditional_branch else '否'}",
        f"- *要点概述*：{ev.summary}",
        f"- *具体内容*：\n\n{ev.content}" if ev.content else "- *具体内容*：[存疑-低，仅保留要点概述]",
    ]
    return "\n".join(lines)


def _format_evidence_analyst_direct(ev: EvidenceItem) -> str:
    """Analyst 视图 - 直接相关：全部字段 + 具体内容"""
    return _format_evidence_full(ev)


def _format_evidence_analyst_indirect(ev: EvidenceItem) -> str:
    """Analyst 视图 - 间接相关：精简字段 + 仅要点概述（已含「」原文引用）"""
    lines = [
        f"**业务依据 {ev.id}**",
        f"- *来源*：{ev.source}",
        f"- *相关性*：{ev.relevance.value} — {ev.relevance_reason}",
        f"- *信息完整性*：{ev.completeness}",
        f"- *信息质量*：{'、'.join(ev.quality_tags)}",
        f"- *是否包含条件分支*：{'是' if ev.has_conditional_branch else '否'}",
        f"- *要点概述*：{ev.summary}",
    ]
    return "\n".join(lines)


def _format_evidence_analyst_uncertain(ev: EvidenceItem) -> str:
    """Analyst 视图 - 存疑-高/存疑-低：最精简（来源 + 相关性 + 质量 + 概述）"""
    lines = [
        f"**业务依据 {ev.id}**",
        f"- *来源*：{ev.source}",
        f"- *相关性*：{ev.relevance.value} — {ev.relevance_reason}",
        f"- *信息质量*：{'、'.join(ev.quality_tags)}",
        f"- *要点概述*：{ev.summary}",
    ]
    return "\n".join(lines)


def _format_contradictions(output: CuratorOutput) -> str:
    if not output.contradictions:
        return "### 矛盾提示\n\n未发现跨片段矛盾。"
    lines = ["### 矛盾提示\n"]
    lines.append("| 矛盾编号 | 涉及依据 | 矛盾描述 |")
    lines.append("|:---|:---|:---|")
    for c in output.contradictions:
        ids_str = " ↔ ".join([f"业务依据 {i}" for i in c.evidence_ids])
        lines.append(f"| {c.id} | {ids_str} | {c.description} |")
    return "\n".join(lines)


def _format_coverage(output: CuratorOutput) -> str:
    lines = [
        "### 检索覆盖度自评\n",
        f"- **覆盖度**：{output.coverage.overall.value}",
        "- **逐标的覆盖详情**：",
    ]
    for tc in output.coverage.target_details:
        lines.append(f"  - {tc.target}：{tc.status.value}")
    lines.append(
        f"  - 否定性规则/例外条款：{'有相关依据' if output.coverage.negative_rules_covered else '无相关依据'}"
    )
    lines.append(f"- **未覆盖方面**：{output.coverage.uncovered_aspects}")
    lines.append(f"- **可能原因**：{output.coverage.possible_reasons}")
    return "\n".join(lines)


def _format_discarded(output: CuratorOutput) -> str:
    if not output.discarded_items:
        return "### 丢弃清单\n\n无丢弃片段。"
    lines = ["### 丢弃清单\n"]
    lines.append("| 序号 | 来源文档 | 内容摘要 | 丢弃原因 |")
    lines.append("|:---|:---|:---|:---|")
    for d in output.discarded_items:
        lines.append(f"| {d.id} | {d.source} | {d.content_summary} | {d.discard_reason} |")
    return "\n".join(lines)


def _format_evaluation_basis(output: CuratorOutput) -> str:
    targets_str = "\n".join([f"  {i+1}. {t}" for i, t in enumerate(output.evaluation_basis.specific_targets)])
    return (
        f"### 评估基准\n\n"
        f"- **检索内容**：{output.evaluation_basis.search_content}\n"
        f"- **具体标的**：\n{targets_str}\n"
        f"- **关联信息需求**：{output.evaluation_basis.related_info_needs}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  三大视图构建器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_rule_splitter_view(output: CuratorOutput) -> str:
    """
    Rule Splitter 视图（最完整）
    包含：评估基准 + 全部依据（含具体内容，存疑-低除外）+ 矛盾提示
    不含：丢弃清单、覆盖度自评、后续步骤检索参考
    """
    sections = []

    # 评估基准
    sections.append(_format_evaluation_basis(output))

    # 依据清单
    summary_line = (
        f"### 相关业务依据清单\n\n"
        f"**结论**：从 Searcher 筛选的 {output.total_input_count} 条片段中，"
        f"保留 {output.retained_count} 条"
        f"（直接相关 {output.direct_count} 条，间接相关 {output.indirect_count} 条，"
        f"存疑-高 {output.uncertain_high_count} 条，存疑-低 {output.uncertain_low_count} 条），"
        f"丢弃 {output.discarded_count} 条。\n"
    )
    sections.append(summary_line)

    for ev in output.evidences:
        sections.append(_format_evidence_full(ev))
        sections.append("")  # 空行分隔

    # 矛盾提示
    sections.append(_format_contradictions(output))

    return "\n\n".join(sections)


def build_analyst_view(output: CuratorOutput) -> str:
    """
    Analyst 视图（分层精简）
    - 直接相关：完整信息 + 具体内容全文
    - 间接相关：精简信息 + 仅要点概述（含「」原文引用）
    - 存疑-高 / 存疑-低：最精简（来源 + 相关性 + 质量 + 要点概述）
    + 矛盾提示 + 丢弃清单 + 覆盖度自评
    """
    sections = []

    # 评估基准
    sections.append(_format_evaluation_basis(output))

    # 依据清单（分层）
    summary_line = (
        f"### 相关业务依据清单\n\n"
        f"**结论**：从 Searcher 筛选的 {output.total_input_count} 条片段中，"
        f"保留 {output.retained_count} 条"
        f"（直接相关 {output.direct_count} 条，间接相关 {output.indirect_count} 条，"
        f"存疑-高 {output.uncertain_high_count} 条，存疑-低 {output.uncertain_low_count} 条），"
        f"丢弃 {output.discarded_count} 条。\n\n"
        f"*注：直接相关依据包含具体内容全文；间接相关依据仅呈现要点概述（含原文关键语句引用）；"
        f"存疑依据仅呈现要点概述。*"
    )
    sections.append(summary_line)

    for ev in output.evidences:
        if ev.relevance == Relevance.DIRECT:
            sections.append(_format_evidence_analyst_direct(ev))
        elif ev.relevance == Relevance.INDIRECT:
            sections.append(_format_evidence_analyst_indirect(ev))
        else:  # UNCERTAIN_HIGH, UNCERTAIN_LOW
            sections.append(_format_evidence_analyst_uncertain(ev))
        sections.append("")

    # 矛盾提示
    sections.append(_format_contradictions(output))

    # 丢弃清单
    sections.append(_format_discarded(output))

    # 覆盖度自评
    sections.append(_format_coverage(output))

    return "\n\n".join(sections)


def build_searcher_view(output: CuratorOutput) -> str:
    """
    后续 Searcher 视图（最精简）
    仅包含：后续步骤检索参考
    """
    ref = output.searcher_reference
    docs_str = "\n".join(
        [f"  {i+1}. {d.name} — {d.topic}" for i, d in enumerate(ref.discovered_documents)]
    ) or "  （无）"

    return (
        f"### 已完成步骤检索摘要\n\n"
        f"- **检索主题**：{ref.search_topic}\n"
        f"- **已覆盖标的**：{'、'.join(ref.covered_targets) if ref.covered_targets else '无'}\n"
        f"- **未覆盖标的**：{'、'.join(ref.uncovered_targets) if ref.uncovered_targets else '无'}\n"
        f"- **已发现文档清单**：\n{docs_str}\n"
        f"- **待追踪线索**：{ref.pending_leads}"
    )