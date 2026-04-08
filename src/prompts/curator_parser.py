"""
curator_parser.py
将 Evidence Curator 的 Markdown 输出确定性解析为 CuratorOutput 结构化对象。

设计原则：
- 纯字符串操作 + 正则，无 LLM 调用
- 所有计数字段由代码统计，不信任 LLM 输出
- 解析失败时抛出明确异常，便于上层重试或降级
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from curator_models import (
    ContentType,
    ContradictionNote,
    CoverageAssessment,
    CoverageStatus,
    CuratorOutput,
    DiscardedItem,
    DiscoveredDocument,
    EvaluationBasis,
    EvidenceItem,
    OverallCoverage,
    Relevance,
    SearcherReference,
    TargetCoverage,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  常量映射
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RELEVANCE_MAP: dict[str, Relevance] = {
    "直接相关": Relevance.DIRECT,
    "间接相关": Relevance.INDIRECT,
    "存疑-高": Relevance.UNCERTAIN_HIGH,
    "存疑-低": Relevance.UNCERTAIN_LOW,
}

CONTENT_TYPE_MAP: dict[str, ContentType] = {
    "条款规则": ContentType.REGULATION,
    "费率表": ContentType.FEE_TABLE,
    "操作流程": ContentType.PROCEDURE,
    "FAQ": ContentType.FAQ,
    "产品说明": ContentType.PRODUCT_SPEC,
    "其他": ContentType.OTHER,
}

COVERAGE_STATUS_MAP: dict[str, CoverageStatus] = {
    "有直接依据": CoverageStatus.DIRECT_EVIDENCE,
    "仅有间接依据": CoverageStatus.INDIRECT_ONLY,
    "无依据": CoverageStatus.NO_EVIDENCE,
}

OVERALL_COVERAGE_MAP: dict[str, OverallCoverage] = {
    "充分覆盖": OverallCoverage.FULL,
    "部分覆盖": OverallCoverage.PARTIAL,
    "未覆盖": OverallCoverage.NONE,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  异常
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CuratorParseError(Exception):
    """Curator Markdown 解析失败"""

    def __init__(self, section: str, detail: str):
        self.section = section
        self.detail = detail
        super().__init__(f"[{section}] {detail}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _split_sections(text: str) -> dict[str, str]:
    """
    按 '### ' 三级标题切分 Markdown 为 {标题: 内容} 字典。
    """
    pattern = re.compile(r"^### (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        raise CuratorParseError("整体结构", "未找到任何 '### ' 三级标题")

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[title] = text[start:end].strip()
    return sections


def _extract_field(text: str, field_name: str) -> Optional[str]:
    """
    从 Markdown 文本中提取 `- *field_name*：value` 或 `- **field_name**：value` 格式的值。
    返回 value 部分（去首尾空白）；未找到返回 None。
    """
    # 匹配 - *字段名*：值  或  - **字段名**：值
    pattern = re.compile(
        rf"^-\s+\*+{re.escape(field_name)}\*+\s*[：:]\s*(.*)$",
        re.MULTILINE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _extract_numbered_list(text: str) -> list[str]:
    """提取有序编号列表项（1. xxx  2. xxx ...）"""
    items = re.findall(r"^\s*\d+\.\s+(.+)$", text, re.MULTILINE)
    return [item.strip() for item in items]


def _parse_markdown_table(text: str) -> list[dict[str, str]]:
    """
    解析标准 Markdown 表格为 list[dict]。
    第一行为表头，第二行为分隔线（跳过），后续行为数据。
    """
    lines = [
        line.strip()
        for line in text.strip().splitlines()
        if line.strip() and line.strip().startswith("|")
    ]
    if len(lines) < 3:  # 表头 + 分隔 + 至少一行数据
        return []

    def split_row(line: str) -> list[str]:
        cells = line.split("|")
        # 去掉首尾空 cell（因为 |a|b| split 后首尾为 ''）
        return [c.strip() for c in cells if c.strip() != ""]

    headers = split_row(lines[0])
    rows: list[dict[str, str]] = []
    for line in lines[2:]:  # 跳过分隔行
        cells = split_row(line)
        row = {}
        for j, h in enumerate(headers):
            row[h] = cells[j] if j < len(cells) else ""
        rows.append(row)
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  各区块解析器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _parse_evaluation_basis(section_text: str) -> EvaluationBasis:
    search_content = _extract_field(section_text, "检索内容") or ""
    related_info = _extract_field(section_text, "关联信息需求") or ""
    targets = _extract_numbered_list(section_text)
    return EvaluationBasis(
        search_content=search_content,
        specific_targets=targets,
        related_info_needs=related_info,
    )


def _split_evidence_blocks(section_text: str) -> list[str]:
    """将依据清单区块按 '**业务依据 N**' 切分为各条依据的文本块。"""
    pattern = re.compile(r"^\*\*业务依据\s+\d+\*\*", re.MULTILINE)
    matches = list(pattern.finditer(section_text))
    if not matches:
        return []
    blocks: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        blocks.append(section_text[start:end].strip())
    return blocks


def _parse_single_evidence(block: str, evidence_id: int) -> EvidenceItem:
    """解析单条业务依据的文本块。"""

    # ── 来源 ──
    source = _extract_field(block, "来源") or ""

    # ── 合并自（可选）──
    merged_from: Optional[list[int]] = None
    merged_raw = _extract_field(block, "合并自")
    if merged_raw:
        nums = re.findall(r"\d+", merged_raw)
        merged_from = [int(n) for n in nums] if nums else None

    # ── 相关性 ──
    relevance_raw = _extract_field(block, "相关性") or ""
    # 格式: "直接相关 — 理由" 或 "存疑-高 — 理由"
    relevance_parts = re.split(r"\s*[—\-–]\s*", relevance_raw, maxsplit=1)
    relevance_label = relevance_parts[0].strip()
    relevance_reason = relevance_parts[1].strip() if len(relevance_parts) > 1 else ""
    relevance = RELEVANCE_MAP.get(relevance_label, Relevance.UNCERTAIN_HIGH)

    # ── 信息完整性（存疑-低可能缺失）──
    completeness = _extract_field(block, "信息完整性") or ""

    # ── 信息质量 ──
    quality_raw = _extract_field(block, "信息质量") or "清晰"
    quality_tags = [tag.strip() for tag in re.split(r"[、,，]", quality_raw) if tag.strip()]

    # ── 内容类型 ──
    content_type_raw = _extract_field(block, "内容类型") or "其他"
    content_type = CONTENT_TYPE_MAP.get(content_type_raw.strip(), ContentType.OTHER)

    # ── 条件分支 ──
    branch_raw = _extract_field(block, "是否包含条件分支") or "否"
    has_branch = branch_raw.strip() == "是"

    # ── 要点概述 ──
    summary = _extract_field(block, "要点概述") or ""

    # ── 具体内容 ──
    content: Optional[str] = None
    if relevance != Relevance.UNCERTAIN_LOW:
        content_match = re.search(
            r"^-\s+\*具体内容\*\s*[：:]\s*$",
            block,
            re.MULTILINE,
        )
        if content_match:
            content_start = content_match.end()
            content = block[content_start:].strip()
            # 如果内容为占位符标记，清理
            if content.startswith("[存疑-低"):
                content = None
        else:
            # 尝试匹配行内内容（短内容情况）
            inline_match = re.search(
                r"^-\s+\*具体内容\*\s*[：:]\s*(.+)$",
                block,
                re.MULTILINE,
            )
            if inline_match:
                val = inline_match.group(1).strip()
                if val.startswith("[存疑-低"):
                    content = None
                else:
                    content = val
    # 存疑-低强制 null
    if relevance == Relevance.UNCERTAIN_LOW:
        content = None

    return EvidenceItem(
        id=evidence_id,
        source=source,
        merged_from=merged_from,
        relevance=relevance,
        relevance_reason=relevance_reason,
        completeness=completeness,
        quality_tags=quality_tags,
        content_type=content_type,
        has_conditional_branch=has_branch,
        summary=summary,
        content=content,
    )


def _parse_contradictions(section_text: str) -> list[ContradictionNote]:
    if "未发现" in section_text:
        return []
    rows = _parse_markdown_table(section_text)
    notes: list[ContradictionNote] = []
    for row in rows:
        cid_raw = row.get("矛盾编号", "0")
        cid = int(re.search(r"\d+", cid_raw).group()) if re.search(r"\d+", cid_raw) else len(notes) + 1
        involved_raw = row.get("涉及依据", "")
        evidence_ids = [int(n) for n in re.findall(r"\d+", involved_raw)]
        desc = row.get("矛盾描述", "")
        notes.append(ContradictionNote(id=cid, evidence_ids=evidence_ids, description=desc))
    return notes


def _parse_discarded(section_text: str) -> list[DiscardedItem]:
    if "无丢弃" in section_text:
        return []
    rows = _parse_markdown_table(section_text)
    items: list[DiscardedItem] = []
    for row in rows:
        did_raw = row.get("序号", "0")
        did = int(re.search(r"\d+", did_raw).group()) if re.search(r"\d+", did_raw) else len(items) + 1
        items.append(
            DiscardedItem(
                id=did,
                source=row.get("来源文档", ""),
                content_summary=row.get("内容摘要", ""),
                discard_reason=row.get("丢弃原因", ""),
            )
        )
    return items


def _parse_coverage(section_text: str) -> CoverageAssessment:
    # 整体覆盖度
    overall_raw = _extract_field(section_text, "覆盖度") or "部分覆盖"
    overall = OVERALL_COVERAGE_MAP.get(overall_raw.strip(), OverallCoverage.PARTIAL)

    # 逐标的覆盖详情
    target_details: list[TargetCoverage] = []
    detail_block = section_text
    # 提取缩进列表项: "  - xxx：yyy"
    detail_lines = re.findall(r"^\s+-\s+(.+?)[：:](.+)$", detail_block, re.MULTILINE)
    negative_covered = False
    for target_name, status_raw in detail_lines:
        target_name = target_name.strip()
        status_raw = status_raw.strip()
        if "否定性规则" in target_name or "例外条款" in target_name:
            negative_covered = "有" in status_raw
            continue
        status = COVERAGE_STATUS_MAP.get(status_raw, CoverageStatus.NO_EVIDENCE)
        target_details.append(TargetCoverage(target=target_name, status=status))

    uncovered = _extract_field(section_text, "未覆盖方面") or "无"
    reasons = _extract_field(section_text, "可能原因") or ""

    return CoverageAssessment(
        overall=overall,
        target_details=target_details,
        negative_rules_covered=negative_covered,
        uncovered_aspects=uncovered,
        possible_reasons=reasons,
    )


def _parse_searcher_reference(section_text: str) -> SearcherReference:
    topic = _extract_field(section_text, "本步骤检索主题") or ""
    covered_raw = _extract_field(section_text, "已覆盖标的") or "无"
    uncovered_raw = _extract_field(section_text, "未覆盖标的") or "无"

    def split_targets(raw: str) -> list[str]:
        if raw.strip() in ("无", ""):
            return []
        return [t.strip() for t in re.split(r"[、;；,，]", raw) if t.strip()]

    covered = split_targets(covered_raw)
    uncovered = split_targets(uncovered_raw)

    # 已发现文档清单
    docs: list[DiscoveredDocument] = []
    doc_lines = re.findall(r"^\s*\d+\.\s+(.+?)\s*[—\-–]\s*(.+)$", section_text, re.MULTILINE)
    for name, topic_str in doc_lines:
        docs.append(DiscoveredDocument(name=name.strip(), topic=topic_str.strip()))

    leads = _extract_field(section_text, "待追踪线索") or "无"

    return SearcherReference(
        search_topic=topic,
        covered_targets=covered,
        uncovered_targets=uncovered,
        discovered_documents=docs,
        pending_leads=leads,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# 区块标题到期望的映射键
_SECTION_KEYS = {
    "评估基准": "evaluation_basis",
    "相关业务依据清单": "evidences",
    "矛盾提示": "contradictions",
    "丢弃清单": "discarded",
    "检索覆盖度自评": "coverage",
    "后续步骤检索参考": "searcher_reference",
}


def parse_curator_markdown(markdown_text: str) -> CuratorOutput:
    """
    将 Evidence Curator 的 Markdown 输出解析为 CuratorOutput。

    Raises:
        CuratorParseError: 关键区块缺失或解析失败时抛出
    """
    sections = _split_sections(markdown_text)

    # ── 评估基准 ──
    basis_text = sections.get("评估基准")
    if not basis_text:
        raise CuratorParseError("评估基准", "未找到该区块")
    evaluation_basis = _parse_evaluation_basis(basis_text)

    # ── 业务依据清单 ──
    evidence_text = sections.get("相关业务依据清单")
    if not evidence_text:
        raise CuratorParseError("相关业务依据清单", "未找到该区块")
    evidence_blocks = _split_evidence_blocks(evidence_text)
    evidences = [
        _parse_single_evidence(block, idx + 1) for idx, block in enumerate(evidence_blocks)
    ]

    # ── 矛盾提示 ──
    contradiction_text = sections.get("矛盾提示", "未发现")
    contradictions = _parse_contradictions(contradiction_text)

    # ── 丢弃清单 ──
    discarded_text = sections.get("丢弃清单", "无丢弃")
    discarded_items = _parse_discarded(discarded_text)

    # ── 覆盖度自评 ──
    coverage_text = sections.get("检索覆盖度自评")
    if not coverage_text:
        raise CuratorParseError("检索覆盖度自评", "未找到该区块")
    coverage = _parse_coverage(coverage_text)

    # ── 后续步骤检索参考 ──
    ref_text = sections.get("后续步骤检索参考")
    if not ref_text:
        raise CuratorParseError("后续步骤检索参考", "未找到该区块")
    searcher_reference = _parse_searcher_reference(ref_text)

    # ── 代码计算计数字段 ──
    direct_count = sum(1 for e in evidences if e.relevance == Relevance.DIRECT)
    indirect_count = sum(1 for e in evidences if e.relevance == Relevance.INDIRECT)
    uncertain_high = sum(1 for e in evidences if e.relevance == Relevance.UNCERTAIN_HIGH)
    uncertain_low = sum(1 for e in evidences if e.relevance == Relevance.UNCERTAIN_LOW)
    retained = len(evidences)
    discarded = len(discarded_items)
    total = retained + discarded

    return CuratorOutput(
        evaluation_basis=evaluation_basis,
        total_input_count=total,
        retained_count=retained,
        direct_count=direct_count,
        indirect_count=indirect_count,
        uncertain_high_count=uncertain_high,
        uncertain_low_count=uncertain_low,
        discarded_count=discarded,
        evidences=evidences,
        contradictions=contradictions,
        discarded_items=discarded_items,
        coverage=coverage,
        searcher_reference=searcher_reference,
    )