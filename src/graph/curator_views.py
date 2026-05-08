"""从 CuratorOutput 生成下游 Agent 所需的三种视图。

核心操作只有两种：
1. 字符串原样传递（5 个文本块字段）
2. 按 relevance 枚举过滤 evidence（唯一的分支逻辑）

没有任何"解析"动作，因此不可能因输入变化而失败。
"""

from __future__ import annotations

from .curator_models import CuratorOutput, EvidenceItem, Relevance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 内部工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_content_boundary(body: str) -> tuple[str, str]:
    """
    将 evidence body 文本分为 metadata 部分和 content 部分。

    约定：body 中 "具体内容：" 行以下为 content。
    如果找不到，整个 body 视为 metadata（存疑-低等无 content 的情况）。

    返回 (metadata_text, content_text)，content_text 可能为空字符串。
    """
    # 查找 "具体内容：" 或 "具体内容:" 行
    lines = body.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("具体内容") and ("：" in stripped or ":" in stripped):
            metadata = "\n".join(lines[:i])
            # 具体内容可能紧跟在冒号后，也可能在下一行
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


def _format_evidence_for_analyst(e: EvidenceItem) -> str:
    """
    分析视图：
    - 直接相关 → 完整 body（含具体内容）
    - 间接相关 / 存疑 → 仅 metadata（不含具体内容）
    """
    if e.relevance == Relevance.DIRECT:
        return f"**业务依据 {e.id}**\n{e.body}"
    else:
        metadata, _ = _find_content_boundary(e.body)
        return f"**业务依据 {e.id}**\n{metadata}"


def generate_analysis_view(output: CuratorOutput) -> str:
    """Analyst 视图：直接相关含全文，其余仅 metadata。"""
    parts = []

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

def generate_full_view(output: CuratorOutput) -> str:
    """Rule Splitter 视图：所有 evidence 完整输出。"""
    parts = []

    parts.append(f"\n## 评估基准\n{output.evaluation_basis}")

    parts.append("\n## 相关业务依据清单")
    for e in output.evidences:
        parts.append("")
        parts.append(f"### 业务依据 {e.id}\n\n{e.body}")

    parts.append(f"\n## 矛盾提示\n{output.contradictions}")

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


