"""
curator_parser.py
将 Evidence Curator 的 Markdown 输出确定性解析为 CuratorOutput 结构化对象。

设计原则：
  - 纯字符串操作 + 正则，无 LLM 调用
  - 所有计数字段由代码统计，不信任 LLM 输出
  - 解析失败时抛出明确异常（定位到区块 + 原因），便于上层重试或降级

修正记录：
  Bug #1 — 相关性字段：改为前缀匹配已知标签，而非按连字符 split
  Bug #2 — 具体内容：统一取"标记位置到块尾"，覆盖行内和换行两种情况
  Bug #3 — 文档清单：分隔符要求两侧至少一个空格，避免文档名内连字符误切
  Bug #4 — 表格解析：显式检测分隔行（含 - 的行），而非硬编码跳过第 2 行
"""

from __future__ import annotations

import re
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

# 按长度降序排列，确保 "存疑-高" 在 "存疑" 之前匹配（前缀匹配安全）
RELEVANCE_LABELS_ORDERED: list[str] = sorted(
    RELEVANCE_MAP.keys(), key=len, reverse=True
)

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
#  通用工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _split_sections(text: str) -> dict[str, str]:
    """
    按 '### ' 三级标题切分 Markdown 为 {标题: 内容} 字典。
    标题去除首尾空白，内容为该标题与下一个三级标题之间的文本。
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
    从文本中提取 `- *field_name*：value` 或 `- **field_name**：value` 格式的行内值。
    仅返回同一行内冒号之后的内容（不跨行）。
    未找到返回 None。
    """
    pattern = re.compile(
        rf"^-\s+\*{{1,2}}{re.escape(field_name)}\*{{1,2}}\s*[：:]\s*(.*)$",
        re.MULTILINE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _extract_numbered_list(text: str) -> list[str]:
    """提取有序编号列表项 (1. xxx  2. xxx ...)"""
    items = re.findall(r"^\s*\d+\.\s+(.+)$", text, re.MULTILINE)
    return [item.strip() for item in items]


def _is_table_separator(line: str) -> bool:
    """
    判断一行是否为 Markdown 表格分隔行。
    分隔行特征：由 |、-、:、空格组成，且包含至少一个 -。
    示例：|:---|:---|:---|  或  | --- | --- |
    """
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    # 去掉 | 后，剩余字符只含 -、:、空格
    inner = stripped.replace("|", "")
    return bool(inner.strip()) and all(c in "-: " for c in inner)


def _parse_markdown_table(text: str) -> list[dict[str, str]]:
    """
    解析标准 Markdown 表格为 list[dict]。
    第一行为表头，分隔行（自动检测）被跳过，后续行为数据行。

    [Bug #4 修正] 不再硬编码跳过第 2 行，而是显式检测分隔行。
    """
    lines = [
        line.strip()
        for line in text.strip().splitlines()
        if line.strip() and line.strip().startswith("|")
    ]
    if len(lines) < 3:
        return []

    def split_row(line: str) -> list[str]:
        """按 | 分割并去除首尾空 cell"""
        cells = line.split("|")
        return [c.strip() for c in cells if c.strip() != ""]

    headers = split_row(lines[0])

    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        # [Bug #4 修正] 跳过所有分隔行
        if _is_table_separator(line):
            continue
        cells = split_row(line)
        row = {}
        for j, h in enumerate(headers):
            row[h] = cells[j] if j < len(cells) else ""
        rows.append(row)
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  相关性字段专用解析器（Bug #1 修正）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _parse_relevance_field(raw: str) -> tuple[Relevance, str]:
    """
    解析相关性字段值。

    输入格式: "存疑-高 — 该卡指代不明但含分期费率关键词"
    输出: (Relevance.UNCERTAIN_HIGH, "该卡指代不明但含分期费率关键词")

    [Bug #1 修正] 不使用 re.split 按连字符切割，改为前缀匹配已知标签。
    RELEVANCE_LABELS_ORDERED 按长度降序排列，确保 "存疑-高" 在 "间接相关" 等
    更短标签之前尝试匹配（虽然这些标签之间不存在前缀关系，降序排列是防御性措施）。
    """
    raw = raw.strip()

    for label in RELEVANCE_LABELS_ORDERED:
        if raw.startswith(label):
            rest = raw[len(label) :].strip()
            # 剥离标签与理由之间的分隔符: " — " / " – " / " - "
            # 注意：这里只剥离开头的分隔符，不会误伤理由文本中的连字符
            rest = re.sub(r"^[—–\-]\s*", "", rest).strip()
            return RELEVANCE_MAP[label], rest

    # 兜底：无法识别标签时归为存疑-高（保守策略，避免误丢）
    return Relevance.UNCERTAIN_HIGH, raw


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
    """
    将依据清单区块按 '**业务依据 N**' 切分为各条依据的文本块。
    每个块从 **业务依据 N** 开头，到下一个 **业务依据 M** 之前或区块末尾。
    """
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


def _extract_content(block: str) -> Optional[str]:
    """
    从单条依据文本块中提取"具体内容"字段的多行文本。

    [Bug #2 修正]
    统一策略：定位 `- *具体内容*：` 的位置，取从该位置到块尾的全部文本。
    无论内容是在冒号后紧跟（行内开始）还是在下一行开始，都能正确提取。

    返回 None 表示未找到该字段或内容为存疑-低占位符。
    """
    # 匹配 "- *具体内容*：" 行——注意冒号后可能紧跟内容也可能换行
    pattern = re.compile(
        r"^-\s+\*具体内容\*\s*[：:]\s*",
        re.MULTILINE,
    )
    m = pattern.search(block)
    if not m:
        return None

    # 从冒号之后（含可能的行内文本）取到块尾
    raw_content = block[m.end() :].strip()

    if not raw_content:
        return None

    # 检查是否为存疑-低占位符
    if raw_content.startswith("[存疑-低") or raw_content.startswith("[存疑－低"):
        return None

    return raw_content


def _parse_single_evidence(block: str, evidence_id: int) -> EvidenceItem:
    """
    解析单条业务依据的文本块，提取所有元数据字段和具体内容。
    """

    # ── 来源 ──
    source = _extract_field(block, "来源") or ""

    # ── 合并自（可选字段，仅合并时存在）──
    merged_from: Optional[list[int]] = None
    merged_raw = _extract_field(block, "合并自")
    if merged_raw:
        nums = re.findall(r"\d+", merged_raw)
        merged_from = [int(n) for n in nums] if nums else None

    # ── 相关性（Bug #1 修正：使用前缀匹配）──
    relevance_raw = _extract_field(block, "相关性") or ""
    relevance, relevance_reason = _parse_relevance_field(relevance_raw)

    # ── 信息完整性（存疑-低依据可能不输出此字段）──
    completeness = _extract_field(block, "信息完整性") or ""

    # ── 信息质量 ──
    quality_raw = _extract_field(block, "信息质量") or "清晰"
    quality_tags = [
        tag.strip()
        for tag in re.split(r"[、,，]", quality_raw)
        if tag.strip()
    ]

    # ── 内容类型 ──
    content_type_raw = (_extract_field(block, "内容类型") or "其他").strip()
    content_type = CONTENT_TYPE_MAP.get(content_type_raw, ContentType.OTHER)

    # ── 是否包含条件分支 ──
    branch_raw = (_extract_field(block, "是否包含条件分支") or "否").strip()
    has_branch = branch_raw == "是"

    # ── 要点概述 ──
    summary = _extract_field(block, "要点概述") or ""

    # ── 具体内容（Bug #2 修正：统一提取逻辑）──
    content: Optional[str] = None
    if relevance == Relevance.UNCERTAIN_LOW:
        # 存疑-低强制无具体内容
        content = None
    else:
        content = _extract_content(block)

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
    """解析矛盾提示区块。无矛盾时返回空列表。"""
    if "未发现" in section_text:
        return []
    rows = _parse_markdown_table(section_text)
    notes: list[ContradictionNote] = []
    for row in rows:
        cid_raw = row.get("矛盾编号", "0")
        cid_match = re.search(r"\d+", cid_raw)
        cid = int(cid_match.group()) if cid_match else len(notes) + 1

        involved_raw = row.get("涉及依据", "")
        evidence_ids = [int(n) for n in re.findall(r"\d+", involved_raw)]

        desc = row.get("矛盾描述", "")
        notes.append(
            ContradictionNote(id=cid, evidence_ids=evidence_ids, description=desc)
        )
    return notes


def _parse_discarded(section_text: str) -> list[DiscardedItem]:
    """解析丢弃清单区块。无丢弃时返回空列表。"""
    if "无丢弃" in section_text:
        return []
    rows = _parse_markdown_table(section_text)
    items: list[DiscardedItem] = []
    for row in rows:
        did_raw = row.get("序号", "0")
        did_match = re.search(r"\d+", did_raw)
        did = int(did_match.group()) if did_match else len(items) + 1
        items.append(
            DiscardedItem(
                id=did,
                source=row.get("来源文档", ""),
                content_summary=row.get("内容摘要", row.get("内容摘要（≤30字）", "")),
                discard_reason=row.get("丢弃原因", ""),
            )
        )
    return items


def _parse_coverage(section_text: str) -> CoverageAssessment:
    """解析检索覆盖度自评区块。"""
    # ── 整体覆盖度 ──
    overall_raw = (_extract_field(section_text, "覆盖度") or "部分覆盖").strip()
    overall = OVERALL_COVERAGE_MAP.get(overall_raw, OverallCoverage.PARTIAL)

    # ── 逐标的覆盖详情 ──
    # 匹配缩进列表项: "  - xxx：yyy"（必须有前导空白，排除顶层字段）
    detail_lines = re.findall(
        r"^\s+-\s+(.+?)\s*[：:]\s*(.+)$", section_text, re.MULTILINE
    )

    target_details: list[TargetCoverage] = []
    negative_covered = False

    for target_name, status_raw in detail_lines:
        target_name = target_name.strip()
        status_raw = status_raw.strip()

        # 识别否定性规则专用行
        if "否定性规则" in target_name or "例外条款" in target_name:
            negative_covered = "有" in status_raw
            continue

        status = COVERAGE_STATUS_MAP.get(status_raw, CoverageStatus.NO_EVIDENCE)
        target_details.append(TargetCoverage(target=target_name, status=status))

    # ── 未覆盖方面 & 可能原因 ──
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
    """解析后续步骤检索参考区块。"""
    topic = _extract_field(section_text, "本步骤检索主题") or ""
    covered_raw = _extract_field(section_text, "已覆盖标的") or "无"
    uncovered_raw = _extract_field(section_text, "未覆盖标的") or "无"

    def split_targets(raw: str) -> list[str]:
        if raw.strip() in ("无", ""):
            return []
        return [t.strip() for t in re.split(r"[、;；,，]", raw) if t.strip()]

    covered = split_targets(covered_raw)
    uncovered = split_targets(uncovered_raw)

    # ── 已发现文档清单 ──
    # [Bug #3 修正] 分隔符要求两侧各至少一个空格: "文档名 — 主题"
    # 避免文档名内含连字符（如 "分期-手续费管理办法"）时被误切
    doc_lines = re.findall(
        r"^\s*\d+\.\s+(.+?)\s+[—–-]\s+(.+)$",
        section_text,
        re.MULTILINE,
    )
    docs = [
        DiscoveredDocument(name=name.strip(), topic=topic_str.strip())
        for name, topic_str in doc_lines
    ]

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


def parse_curator_markdown(markdown_text: str) -> CuratorOutput:
    """
    将 Evidence Curator 的 Markdown 输出解析为 CuratorOutput。

    所有计数字段由代码统计（不信任 LLM 计数）。
    解析失败时抛出 CuratorParseError，包含区块名和具体原因。
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
        _parse_single_evidence(block, idx + 1)
        for idx, block in enumerate(evidence_blocks)
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

    # ── 代码计算计数字段（绝不信任 LLM 的自述计数）──
    direct_count = sum(1 for e in evidences if e.relevance == Relevance.DIRECT)
    indirect_count = sum(1 for e in evidences if e.relevance == Relevance.INDIRECT)
    uncertain_high = sum(
        1 for e in evidences if e.relevance == Relevance.UNCERTAIN_HIGH
    )
    uncertain_low = sum(
        1 for e in evidences if e.relevance == Relevance.UNCERTAIN_LOW
    )
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