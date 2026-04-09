"""从 CuratorOutput 生成下游 Agent 所需的三种视图。

核心操作只有两种：
1. 字符串原样传递（5 个文本块字段）
2. 按 relevance 枚举过滤 evidence（唯一的分支逻辑）

没有任何"解析"动作，因此不可能因输入变化而失败。
"""

from __future__ import annotations

from curator_models import CuratorOutput, EvidenceItem, Relevance


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
# 视图 1: 摘要视图 → 后续 Searcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_summary_view(output: CuratorOutput, step_index: int) -> str:
    """直接传递 searcher_reference 文本块。"""
    return f"--- 步骤 {step_index} 检索摘要 ---\n{output.searcher_reference}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 视图 2: 分析视图 → Analyst
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


def generate_analysis_view(output: CuratorOutput, step_index: int) -> str:
    """Analyst 视图：直接相关含全文，其余仅 metadata。"""
    parts = [f"=== 步骤 {step_index} 信息质量评估报告（分析视图）==="]

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
# 视图 3: 完整视图 → Rule Splitter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_full_view(output: CuratorOutput, step_index: int) -> str:
    """Rule Splitter 视图：所有 evidence 完整输出。"""
    parts = [f"=== 步骤 {step_index} 信息质量评估报告（完整视图）==="]

    parts.append(f"\n【评估基准】\n{output.evaluation_basis}")

    parts.append("\n【相关业务依据清单】")
    for e in output.evidences:
        parts.append("")
        parts.append(f"**业务依据 {e.id}**\n{e.body}")

    parts.append(f"\n【矛盾提示】\n{output.contradictions}")

    return "\n".join(parts)