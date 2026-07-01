from __future__ import annotations

from .planner_model import (
    CPPlannerInput, CPSearcherInput, CPEvaluatorInput,
    CPPointAnalystInput, CPReporterInput,
    CPPointAnalystOutput, ToolReturn
)
from .curator_models import CuratorOutput, EvidenceItem, Relevance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 内部工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_content_boundary(body: str) -> tuple[str, str]:
    """
    将 evidence body 文本分为 metadata 部分和 content 部分。

    约定：body 中 "段落编号：" 行为分界线。
    如果找不到，整个 body 视为 metadata（存疑-低等无 content 的情况）。

    返回 (metadata_text, content_text)，content_text 可能为空字符串。
    """
    # 查找 "段落编号：" 行
    lines = body.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("段落编号") and ("：" in stripped or ":" in stripped):
            metadata = "\n".join(lines[:i])
            # 段落编号可能紧跟在冒号后，也可能在下一行
            colon_pos = stripped.find("：")
            if colon_pos == -1:
                colon_pos = stripped.find(":")
            after_colon = stripped[colon_pos + 1:].strip()
            if after_colon:
                content = after_colon + "\n" + "\n".join(lines[i + 1:])
            else:
                content = "\n".join(lines[i + 1:])
            return metadata.strip(), content.strip()
    return body.strip(), ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 视图 1: 分析视图 → Analyst
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


_INDIRECT_CONTENT_PREVIEW_LIMIT = 300


def _format_evidence_for_analyst(e: EvidenceItem) -> str:
    """
    分析视图（改造后）：
    - 直接相关 → 完整 body
    - 间接相关 / 存疑 → metadata + 具体内容截断预览
    """
    if e.relevance == Relevance.DIRECT:
        return f"**业务依据 {e.id}**\n{e.body}"

    metadata, content = _find_content_boundary(e.body)

    parts = [f"**业务依据 {e.id}**", metadata]

    # ← 新增：具体内容截断预览，供 Analyst 交叉核实摘引
    if content:
        preview = content[:_INDIRECT_CONTENT_PREVIEW_LIMIT]
        if len(content) > _INDIRECT_CONTENT_PREVIEW_LIMIT:
            preview += "\n…[具体内容已截断，以上为前部预览]"
        parts.append(f"具体内容预览：\n{preview}")

    return "\n".join(parts)



def generate_analysis_view(output: CuratorOutput) -> str:
    """Analyst 视图：直接相关含全文，其余仅 metadata + 预览 + 统计概览。"""
    parts: list[str] = []

    # ── 统计概览（新增）──
    stats = (
        f"共保留 {output.retained_count} 条依据"
        f"（直接相关 {output.direct_count}，"
        f"间接相关 {output.indirect_count}，"
        f"存疑 {output.uncertain_count}）"
    )
    if output.expired_count > 0:
        stats += f"，其中 {output.expired_count} 条已过期"
    if output.tool_extracted_count > 0:
        stats += f"，{output.tool_extracted_count} 条为工具截取"
    parts.append(f"【依据统计】{stats}。")

    # ── 以下不变 ──
    parts.append(f"\n【评估基准】\n{output.evaluation_basis}")

    parts.append("\n【相关业务依据清单】")
    for e in output.evidences:
        parts.append("")
        parts.append(_format_evidence_for_analyst(e))

    parts.append(f"\n【矛盾提示】\n{output.contradictions}")
    parts.append(f"\n【丢弃清单】\n{output.discarded}")
    parts.append(f"\n【检索覆盖度自评】\n{output.coverage}")

    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 视图 2: 完整视图 → Rule Splitter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_rule_splitter_view(
    output: CuratorOutput,
    resolved_list: list[ResolvedEvidence] | None = None,
) -> str:
    """
    Rule Splitter 完整视图。

    两种模式：
    - 有 resolved_list：用 chunk_map 原文替换 Curator 精简后的具体内容
      → Rule Splitter 拿到的是未经 Curator 精简的完整段落原文
    - 无 resolved_list：回退到 Curator body 原始内容（向后兼容）
    """
    parts: list[str] = []

    # ── 评估基准 ──
    parts.append(f"## 评估基准\n\n{output.evaluation_basis}")

    # ── 相关业务依据清单 ──
    parts.append("\n## 相关业务依据清单")

    if resolved_list:
        # ── 有 resolved_list：逐条用 chunk_map 原文重建 ──
        resolved_map = {r.evidence_id: r for r in resolved_list}

        for evidence in output.evidences:
            parts.append("")
            resolved = resolved_map.get(evidence.id)

            if resolved and resolved.resolved_chunks:
                # 用 chunk_map 原文重建具体内容
                content_lines = []
                for chunk in resolved.resolved_chunks:
                    content_lines.append(
                        f"**[{chunk['chunk_index']}]**\n{chunk['chunk_content']}"
                    )
                rebuilt_content = "\n\n".join(content_lines)
                parts.append(
                    f"### 业务依据 {evidence.id}\n\n"
                    f"{resolved.body_metadata}\n"
                    f"具体内容：\n{rebuilt_content}"
                )
            else:
                # 回退：chunk_map 未命中，使用 Curator body 原始内容
                parts.append(f"### 业务依据 {evidence.id}\n\n{evidence.body}")
    else:
        # ── 无 resolved_list：直接使用 Curator body（向后兼容）──
        for evidence in output.evidences:
            parts.append("")
            parts.append(f"### 业务依据 {evidence.id}\n\n{evidence.body}")

    # ── 矛盾提示 ──
    parts.append(f"\n## 矛盾提示\n\n{output.contradictions}")

    return "\n".join(parts)


def build_evaluator_input(step, searcher_annotation, raw_tool_returns) -> str:
    """组装 CP Evidence Evaluator 的输入文本"""

    input_parts = []

    # 1. 当前审查点信息
    input_parts.append(f"# 当前审查点信息\n")
    input_parts.append(f"- 检索标题：{step.title}")
    input_parts.append(f"- 检索背景：{step.background}")
    input_parts.append(f"- 检索内容：{step.description}")

    # 2. 检索结果摘要（Searcher 的 Markdown 输出）
    input_parts.append(f"\n# 检索结果摘要\n")
    input_parts.append(searcher_annotation)

    # 3. 原始工具返回（框架从 Searcher 消息历史中提取）
    input_parts.append(f"\n# 原始工具返回\n")
    for i, tool_return in enumerate(raw_tool_returns, 1):
        input_parts.append(f"## {tool_return.tool_name}_{i} 结果\n")
        input_parts.append(tool_return.content)

    return "\n".join(input_parts)



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. Reporter 输入格式化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def format_citations_for_reporter(
    citations: list[dict[str, Any]],
) -> str:
    """
    将 citations 格式化为 Reporter 提示词中的"可用参考来源"段落。

    输出示例:
        # 可用参考来源

        正文中需要引用时，在句末使用 `[[n]](#ref-n)`，n 为下方序号。
        仅限引用下方列出的来源，严禁编造 URL。

        [1] 信用卡分期业务管理办法 - http://xxx
        [2] 白金卡产品说明书 - http://yyy
    """
    ordered = _dedupe_citations_preserve_order(citations)

    if not ordered:
        return (
            "# 可用参考来源\n\n"
            "当前无可用参考来源。报告中请勿使用 `[[n]](#ref-n)` 引用标记。\n"
            "如需引用文档，仅用《文档名》指代。"
        )

    lines = [
        "# 可用参考来源",
        "",
        "正文中需要引用时，在句末使用 `[[n]](#ref-n)`，n 为下方序号。",
        "仅限引用下方列出的来源，严禁编造未列出的来源或 URL。",
        "",
    ]

    for i, c in enumerate(ordered, 1):
        title = (c.get("title") or "未知文档").strip()
        url = (c.get("url") or "").strip()

        if url:
            lines.append(f"[{i}] {title} - {url}")
        else:
            lines.append(f"[{i}] {title}")

    return "\n".join(lines)


def _dedupe_citations_preserve_order(
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """去重并保持插入顺序。"""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        key = _citation_dedup_key(c.get("title", ""), c.get("url"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out



def build_point_analyst_input(material_text, step, review_point, evaluator_output) -> str:
    """组装 CP Point Analyst 的输入文本"""

    input_parts = []

    # 1. 宣传文本全文
    input_parts.append(f"# 信用卡业务宣传文本\n")
    input_parts.append(material_text)

    # 2. 当前审查点信息
    input_parts.append(f"\n# 当前审查点信息\n")
    input_parts.append(f"- 审查点编号：{review_point.id}")
    input_parts.append(f"- 审查点名称：{review_point.checklist_item_title}")
    input_parts.append(f"- 触发原因：{review_point.triggered_reason}")
    input_parts.append(f"- 触发片段：{review_point.triggered_excerpt}")

    # 3. 行规法规依据（Evaluator 的 Markdown 输出）
    input_parts.append(f"\n# 行规法规依据\n")
    input_parts.append(evaluator_output)

    return "\n".join(input_parts)


# ══════════════════════════════════════════════════════════════════════
# Planner
# ══════════════════════════════════════════════════════════════════════

def build_planner_user_message(input_data: CPPlannerInput) -> str:
    """组装 Planner 的 User Message"""
    return f"# 信用卡业务宣传文本\n\n{input_data.material_text}"


# ══════════════════════════════════════════════════════════════════════
# Searcher
# ══════════════════════════════════════════════════════════════════════

def build_searcher_user_message(input_data: CPSearcherInput) -> str:
    """组装 Searcher 的 User Message"""
    return (
        f"# 当前步骤\n\n"
        f"- **检索标题**：{input_data.search_title}\n"
        f"- **检索背景**：{input_data.search_background}\n"
        f"- **检索内容**：{input_data.search_description}"
    )


# ══════════════════════════════════════════════════════════════════════
# Evidence Evaluator
# ══════════════════════════════════════════════════════════════════════

def build_evaluator_user_message(input_data: CPEvaluatorInput) -> str:
    """
    组装 Evaluator 的 User Message。
    包含：审查点信息 + 检索摘要 + 原始工具返回。
    """
    parts: list[str] = []

    # Part 1: 审查点信息
    parts.append(
        f"# 当前审查点信息\n\n"
        f"- **检索标题**：{input_data.search_title}\n"
        f"- **检索背景**：{input_data.search_background}\n"
        f"- **检索内容**：{input_data.search_description}"
    )

    # Part 2: 检索摘要
    parts.append(f"\n# 检索结果摘要\n\n{input_data.searcher_annotation}")

    # Part 3: 原始工具返回
    parts.append("\n# 原始工具返回\n")
    if not input_data.raw_tool_returns:
        parts.append("*无工具返回结果。*")
    else:
        for tr in input_data.raw_tool_returns:
            label = _format_tool_label(tr)
            parts.append(f"## {label}\n\n{tr.content}\n")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════
# Point Analyst
# ══════════════════════════════════════════════════════════════════════

def build_point_analyst_user_message(input_data: CPPointAnalystInput) -> str:
    """
    组装 Point Analyst 的 User Message。
    包含：宣传文本 + 审查点信息 + 行规法规依据。
    """
    baseline_label = "是" if input_data.is_baseline else "否"
    return (
        f"# 信用卡业务宣传文本\n\n"
        f"{input_data.material_text}\n\n"
        f"# 当前审查点信息\n\n"
        f"- **审查点编号**：{input_data.review_point_id}\n"
        f"- **审查点名称**：{input_data.review_topic}\n"
        f"- **是否为基线审查点**：{baseline_label}\n"
        f"- **触发原因**：{input_data.triggered_reason}\n"
        f"- **触发片段**：{input_data.triggered_excerpt}\n\n"
        f"# 行规法规依据\n\n"
        f"{input_data.evaluator_output}"
    )


# ══════════════════════════════════════════════════════════════════════
# Final Reporter
# ══════════════════════════════════════════════════════════════════════

def build_reporter_user_message(input_data: CPReporterInput) -> str:
    """
    组装 Final Reporter 的 User Message。
    包含：宣传文本 + 分析结果 JSON 数组 + 参考来源。
    """
    parts: list[str] = []

    # Part 1: 宣传文本
    parts.append(f"# 信用卡业务宣传文本\n\n{input_data.material_text}")

    # Part 2: 分析结果
    parts.append("\n# 审查点分析结果\n")
    assessments_json = _serialize_assessments(input_data.point_assessments)
    parts.append(f"```json\n{assessments_json}\n```")

    # Part 3: 参考来源
    parts.append("\n# 可用参考来源\n")
    if not input_data.available_citations:
        parts.append("*无可用参考来源。*")
    else:
        for c in input_data.available_citations:
            parts.append(f"- [{c.index}] {c.document_name} — {c.url}")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════

def _format_tool_label(tr: ToolReturn) -> str:
    if tr.tool_name == "local_search_tool":
        return f"local_search_tool 第{tr.call_index}次调用（query: \"{tr.query}\"）"
    elif tr.tool_name == "crawl_tool":
        return f"crawl_tool 第{tr.call_index}次调用（url: {tr.url}）"
    return f"{tr.tool_name} 第{tr.call_index}次调用"


def _serialize_assessments(assessments: list[CPPointAnalystOutput]) -> str:
    """将 Point Analyst 输出列表序列化为 JSON 字符串"""
    import json
    items = []
    for pa in assessments:
        items.append(pa.model_dump(mode="json"))
    return json.dumps(items, ensure_ascii=False, indent=2)
