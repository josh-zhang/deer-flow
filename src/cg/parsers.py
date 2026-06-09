from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.cg.types import (
    AudienceReaction,
    AuditReport,
    ChannelCopy,
    ComplianceRuleSet,
    CopyStrategy,
    MasterCopy,
    ProductFacts,
    TasteScore,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class CGParseError(Exception):
    """CG 解析失败"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  核心 JSON 提取与解析工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_json_str(text: str) -> str | None:
    """
    从 LLM 输出中提取 JSON 字符串。

    尝试策略：
    1. 整段即 JSON
    2. <json>...</json> 标签内容
    3. ```json ... ``` 代码块
    4. 首个 {/[ 到末尾的括号匹配
    """
    text = text.strip()

    # 策略 1：整段即 JSON
    if (text.startswith("{") and text.endswith("}")) or \
            (text.startswith("[") and text.endswith("]")):
        return text

    # 策略 2：<json>...</json> 标签
    xml_match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
    if xml_match:
        candidate = xml_match.group(1).strip()
        if candidate.startswith(("{", "[")):
            return candidate

    # 策略 3：代码块提取
    code_pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
    for m in re.findall(code_pattern, text, re.DOTALL):
        m = m.strip()
        if m.startswith(("{", "[")):
            return m

    # 策略 4：首尾括号定位
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        end = text.rfind(end_char)
        if end <= start:
            continue
        return text[start:end + 1]

    return None


def _load_json(text: str) -> dict | list:
    """尝试解析 JSON，失败时使用 json_repair 兜底。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 去除 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # json_repair 兜底
    try:
        import json_repair
        return json_repair.loads(text)
    except Exception:
        pass

    raise CGParseError(f"JSON 解析失败，所有修复策略均未成功。文本预览: {text[:200]}")


def parse_model_from_text(text: str, model_class: type[T]) -> T:
    """
    从 LLM 输出文本中提取 JSON 并解析为 Pydantic 模型。

    解析链：
    1. extract_json_str — 提取 JSON 字符串
    2. _load_json — JSON 反序列化（含 json_repair 兜底）
    3. model_validate — Pydantic v2 校验
    4. 宽松重试 — 移除 None 值字段后重试
    """
    json_str = extract_json_str(text)
    if json_str is None:
        raise CGParseError(
            f"无法从文本中提取 JSON。model={model_class.__name__}, "
            f"text_preview={text[:200]}"
        )

    try:
        data = _load_json(json_str)
    except CGParseError:
        raise CGParseError(
            f"JSON 解析失败。model={model_class.__name__}, "
            f"json_preview={json_str[:200]}"
        )

    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        logger.warning("Pydantic 校验有问题，尝试宽松解析。errors=%d", e.error_count())
        if isinstance(data, dict):
            cleaned = {k: v for k, v in data.items() if v is not None}
            try:
                return model_class.model_validate(cleaned)
            except ValidationError:
                pass
        raise CGParseError(
            f"模型校验失败: {e}。model={model_class.__name__}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  类型化解析函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_product_facts(text: str) -> ProductFacts:
    """
    解析 Fact Miner 的 JSON 输出为 ProductFacts。

    fallback: 若整体解析失败，返回仅含 product_name 的空结构。
    """
    try:
        return parse_model_from_text(text, ProductFacts)
    except CGParseError as e:
        logger.error("parse_product_facts 失败: %s", e)
        # fallback：尝试从文本中提取产品名称
        product_name = _extract_field_fallback(text, "product_name")
        return ProductFacts(product_name=product_name or "未解析")


def parse_compliance_rule_set(text: str) -> ComplianceRuleSet:
    """
    解析 Rule Miner 的 JSON 输出为 ComplianceRuleSet。

    fallback: 若整体解析失败，返回空规则集。
    """
    try:
        return parse_model_from_text(text, ComplianceRuleSet)
    except CGParseError as e:
        logger.error("parse_compliance_rule_set 失败: %s", e)
        return ComplianceRuleSet()


def parse_copy_strategy(text: str) -> CopyStrategy:
    """
    解析 Copy Strategist 的 JSON 输出为 CopyStrategy。

    fallback: 返回含空列表的空策略。
    """
    try:
        return parse_model_from_text(text, CopyStrategy)
    except CGParseError as e:
        logger.error("parse_copy_strategy 失败: %s", e)
        return CopyStrategy()


def parse_master_copy(text: str, variant_type: str = "universal") -> MasterCopy:
    """
    解析 Copy Writer 的 JSON 输出为 MasterCopy。

    fallback: 若 JSON 解析失败，将全文作为 body 字段返回。
    """
    try:
        copy = parse_model_from_text(text, MasterCopy)
        if not copy.variant_type:
            copy.variant_type = variant_type
        return copy
    except CGParseError as e:
        logger.error("parse_master_copy 失败: %s", e)
        return MasterCopy(variant_type=variant_type, body=text[:2000])


def parse_channel_copy(
    text: str,
    variant_type: str = "universal",
    channel_name: str = "",
) -> ChannelCopy:
    """
    解析 Channel Adapter 或 Copy Rewriter 的 JSON 输出为 ChannelCopy。

    fallback: 若 JSON 解析失败，将全文作为 copy_text 字段返回。
    """
    try:
        copy = parse_model_from_text(text, ChannelCopy)
        if not copy.variant_type:
            copy.variant_type = variant_type
        if not copy.channel_name:
            copy.channel_name = channel_name
        if not copy.char_count:
            copy.char_count = len(copy.copy_text)
        return copy
    except CGParseError as e:
        logger.error("parse_channel_copy 失败: %s", e)
        return ChannelCopy(
            variant_type=variant_type,
            channel_name=channel_name,
            copy_text=text[:2000],
            char_count=len(text),
        )


def parse_audit_report(text: str) -> AuditReport:
    """
    解析 Copy Auditor 的 JSON 输出为 AuditReport。

    fallback: 返回 overall_verdict="需人工介入" 的空报告。
    """
    try:
        return parse_model_from_text(text, AuditReport)
    except CGParseError as e:
        logger.error("parse_audit_report 失败: %s", e)
        return AuditReport(
            overall_verdict="需人工介入",
            issues=[f"审计报告解析失败: {e}"],
        )


def parse_taste_score(text: str) -> TasteScore:
    """
    解析 Taste Guardian 的 JSON 输出为 TasteScore。

    fallback: 返回 overall_pass=False 的默认评分。
    """
    try:
        return parse_model_from_text(text, TasteScore)
    except CGParseError as e:
        logger.error("parse_taste_score 失败: %s", e)
        return TasteScore(
            overall_pass=False,
            rewrite_direction=f"品味评分解析失败，需人工审核: {e}",
        )


def parse_audience_reaction(text: str) -> AudienceReaction:
    """
    解析 Simulated Audience 的 JSON 输出为 AudienceReaction。

    fallback: 返回中性分数的空反应。
    """
    try:
        return parse_model_from_text(text, AudienceReaction)
    except CGParseError as e:
        logger.error("parse_audience_reaction 失败: %s", e)
        return AudienceReaction(
            click_willingness=5.0,
            net_score=5.0,
            delivery_override="",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  内部工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_field_fallback(text: str, field_name: str) -> str:
    """从文本中尝试按字段名提取字符串值（尽力而为）。"""
    pattern = rf'"{field_name}"\s*:\s*"([^"]*)"'
    m = re.search(pattern, text)
    return m.group(1) if m else ""
