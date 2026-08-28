from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import partial
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agents import create_agent
from src.cg.parsers import (
    parse_audience_reaction,
    parse_audit_report,
    parse_channel_copy,
    parse_compliance_rule_set,
    parse_copy_strategy,
    parse_master_copy,
    parse_product_facts,
    parse_taste_score,
)
from src.cg.types import (
    AudienceReaction,
    AuditReport,
    CGState,
    ChannelCopy,
    ChannelSpec,
    MasterCopy,
    PersonaChannelStyleCard,
    PersonaStrategy,
    TasteScore,
)
from src.cg.utils import (
    check_consistency,
    extract_citations_from_tool_cache,
)

# 注：ChannelSpec / PersonaChannelStyleCard / PersonaStrategy
#     在内部函数签名和 model_validate 中使用，提供强类型校验。
#     check_consistency 在 _adapt_one_channel 后调用。
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.graph.curator_views import (
    extract_chunk_maps_from_cache,
    extract_document_metadata_from_cache,
)
from src.agents.agents import make_tool_saver_hook
from src.rag.retriever import ToolCallRecord
from src.graph.ontology import render_cg_audit_edge_checklist
from src.llms.llm import get_llm_by_type, get_llm_token_limit_by_type
from src.prompts.template import apply_prompt_template, get_system_prompt_template
from src.tools import crawl_tool, get_retriever_tool
from src.utils.context_manager import ContextManager

logger = logging.getLogger(__name__)

# ── CG 节点标签 ──────────────────────────────────────────────────────────────
_CG_FACT_TAG = "[cg_fact_miner]"
_CG_RULE_TAG = "[cg_rule_miner]"

# ── LLM 类型映射（CG 节点使用 basic，可按需调整）──────────────────────────────
_CG_LLM_TYPES: dict[str, str] = {
    "cg_fact_miner": "basic",
    "cg_rule_miner": "basic",
    "cg_copy_strategist": "basic",
    "cg_copy_writer": "basic",
    "cg_channel_adapter": "basic",
    "cg_copy_auditor": "basic",
    "cg_copy_rewriter": "basic",
    "cg_taste_guardian": "basic",
    "cg_simulated_audience": "basic",
    "cg_evidence_assembler": "basic",
}


def _current_date() -> str:
    return datetime.now().strftime("%Y年%m月%d日")


def _get_cg_llm(agent_key: str):
    """获取 CG 节点对应的 LLM，先查 CG 专属映射，再查全局映射，最后 fallback basic。"""
    llm_type = _CG_LLM_TYPES.get(agent_key) or AGENT_LLM_MAP.get(agent_key, "basic")
    return get_llm_by_type(llm_type)


def _get_cg_input(state: CGState) -> dict[str, Any]:
    """从 state 中获取 CG 输入参数，兼容 dict 和 CGInput。"""
    cg_input = state.get("cg_input", {})
    if not isinstance(cg_input, dict):
        return {}
    return cg_input


def _safe_json_dumps(obj: Any) -> str:
    """将 Pydantic 模型或 dict 序列化为 JSON 字符串。"""
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 1 — Fact Miner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def fact_miner_node(state: CGState, config: RunnableConfig) -> dict:
    """Phase 1：从业务知识库检索产品事实（ReAct Agent + Tools）。"""
    configurable = Configuration.from_runnable_config(config)
    cg_input = _get_cg_input(state)
    product_name = cg_input.get("product_name", "")
    campaign_name = cg_input.get("campaign_name", "")

    if not product_name:
        logger.warning("fact_miner_node: cg_input 缺少 product_name，跳过执行。")
        return {}

    tools = [t for t in [get_retriever_tool(state.get("resources", [])), crawl_tool] if t]
    locale = state.get("locale", "zh_CN")

    tool_returns_cache: list[ToolCallRecord] = []
    llm_limit = get_llm_token_limit_by_type(_CG_LLM_TYPES.get("cg_fact_miner", "basic"))
    hook = partial(ContextManager(llm_limit, 3).compress_messages)

    agent = create_agent(
        "cg_fact_miner",
        "cg_fact_miner",
        tools,
        "CG/fact_miner",
        make_tool_saver_hook(tool_returns_cache),
        locale=locale,
    )

    user_msg = HumanMessage(content=(
        f"## 产品名称\n{product_name}\n\n"
        f"## 营销活动名称\n{campaign_name or '（无）'}\n\n"
        f"## 当前日期\n{_current_date()}\n\n"
        f"请按照 Fact Miner 执行协议完成检索，并输出结构化 JSON。"
    ))

    try:
        out = await agent.ainvoke(
            {"messages": [user_msg]},
            config={"recursion_limit": 30},
        )
        msgs = out.get("messages", [])
        last_ai = _last_ai_content(msgs)
        product_facts_obj = parse_product_facts(last_ai)
        product_facts = product_facts_obj.model_dump()
    except Exception as e:
        logger.exception("fact_miner_node 失败: %s", e)
        product_facts = {"product_name": product_name, "_error": str(e)}

    # 提取 chunk_maps 和 citations（reducer 会自动与现有值合并）
    new_chunk_maps = extract_chunk_maps_from_cache(tool_returns_cache)
    new_metadata = extract_document_metadata_from_cache(tool_returns_cache)
    fact_citations = extract_citations_from_tool_cache(tool_returns_cache)

    return {
        "product_facts": product_facts,
        "document_chunk_maps": new_chunk_maps,
        "document_metadata": new_metadata,
        "fact_citations": fact_citations,
        "citations": fact_citations,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 1 — Rule Miner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def rule_miner_node(state: CGState, config: RunnableConfig) -> dict:
    """Phase 1：从消保审查知识库检索合规规则（ReAct Agent + Tools）。"""
    configurable = Configuration.from_runnable_config(config)
    cg_input = _get_cg_input(state)
    product_type = cg_input.get("product_type", "")
    channels = cg_input.get("channels", [])

    if not product_type:
        logger.warning("rule_miner_node: cg_input 缺少 product_type，跳过执行。")
        return {}

    channels_str = ", ".join(channels) if channels else "（未指定）"
    tools = [t for t in [get_retriever_tool(state.get("resources", [])), crawl_tool] if t]
    locale = state.get("locale", "zh_CN")

    tool_returns_cache: list[ToolCallRecord] = []
    llm_limit = get_llm_token_limit_by_type(_CG_LLM_TYPES.get("cg_rule_miner", "basic"))

    agent = create_agent(
        "cg_rule_miner",
        "cg_rule_miner",
        tools,
        "CG/rule_miner",
        make_tool_saver_hook(tool_returns_cache),
        locale=locale,
    )

    user_msg = HumanMessage(content=(
        f"## 产品类型\n{product_type}\n\n"
        f"## 渠道列表\n{channels_str}\n\n"
        f"## 当前日期\n{_current_date()}\n\n"
        f"请按照 Rule Miner 执行协议完成检索，并输出结构化 JSON。"
    ))

    try:
        out = await agent.ainvoke(
            {"messages": [user_msg]},
            config={"recursion_limit": 30},
        )
        msgs = out.get("messages", [])
        last_ai = _last_ai_content(msgs)
        compliance_rule_set_obj = parse_compliance_rule_set(last_ai)
        compliance_rule_set = compliance_rule_set_obj.model_dump()
    except Exception as e:
        logger.exception("rule_miner_node 失败: %s", e)
        compliance_rule_set = {"_error": str(e)}

    # 提取 chunk_maps 和 citations（消保知识库，reducer 自动合并）
    new_chunk_maps = extract_chunk_maps_from_cache(tool_returns_cache)
    new_metadata = extract_document_metadata_from_cache(tool_returns_cache)
    rule_citations = extract_citations_from_tool_cache(tool_returns_cache)

    return {
        "compliance_rule_set": compliance_rule_set,
        "document_chunk_maps": new_chunk_maps,
        "document_metadata": new_metadata,
        "rule_citations": rule_citations,
        "citations": rule_citations,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 2 — Copy Strategist
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def copy_strategist_node(state: CGState, config: RunnableConfig) -> dict:
    """Phase 2：文案策略规划（纯 LLM，无工具）。"""
    configurable = Configuration.from_runnable_config(config)
    cg_input = _get_cg_input(state)

    product_facts = state.get("product_facts", {})
    compliance_rule_set = state.get("compliance_rule_set", {})

    channels = cg_input.get("channels", [])
    personas = cg_input.get("personas", [])
    scene_empathy = cg_input.get("scene_empathy", True)
    relationship_temperature = cg_input.get("relationship_temperature", "warm")
    privacy_boundary = cg_input.get("privacy_boundary", "standard")
    ab_test = cg_input.get("ab_test", False)
    historical_ab_summary = cg_input.get("historical_ab_summary", "")

    # 构建经验 Skill 注入文本
    skill_text = _build_skill_injection(state.get("loaded_skills", []))

    sub_state = {
        **state,
        "product_facts": _safe_json_dumps(product_facts),
        "compliance_rules": _safe_json_dumps(compliance_rule_set),
        "channels": json.dumps(channels, ensure_ascii=False),
        "personas": json.dumps(personas, ensure_ascii=False),
        "scene_empathy": str(scene_empathy),
        "relationship_temperature": relationship_temperature,
        "privacy_boundary": privacy_boundary,
        "ab_test": str(ab_test),
        "historical_ab_summary": historical_ab_summary or "（无历史数据）",
        "skill_injection": skill_text or "（无经验库）",
        "current_date": _current_date(),
        "messages": [HumanMessage(content="请按照 Copy Strategist 流程输出文案策略 JSON。")],
    }

    llm = _get_cg_llm("cg_copy_strategist")
    try:
        resp = await llm.ainvoke(
            apply_prompt_template("CG/copy_strategist", sub_state, configurable)
        )
        content = str(resp.content or "")
        copy_strategy_obj = parse_copy_strategy(content)
        copy_strategy = copy_strategy_obj.model_dump()
        ab_plans = copy_strategy.get("ab_plans", [])
    except Exception as e:
        logger.exception("copy_strategist_node 失败: %s", e)
        copy_strategy = {"_error": str(e)}
        ab_plans = []

    return {"copy_strategy": copy_strategy, "ab_plans": ab_plans}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 3 — Copy Writer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def copy_writer_node(state: CGState, config: RunnableConfig) -> dict:
    """Phase 3：母版文案生成，逐版本(1+n)生成（纯 LLM）。支持 AB 变体。"""
    configurable = Configuration.from_runnable_config(config)
    cg_input = _get_cg_input(state)

    product_facts = state.get("product_facts", {})
    compliance_rule_set = state.get("compliance_rule_set", {})
    copy_strategy = state.get("copy_strategy", {})
    ab_test = cg_input.get("ab_test", False)
    ab_plans: list[dict] = state.get("ab_plans", [])

    personas: list[str] = cg_input.get("personas", [])
    # 生成版本列表：universal + 每个客群
    variants = ["universal"] + personas

    # 构建铁框架（hard_constraints）：L1 红线 + L2 强制披露 + L3 审慎表达 + 产品事实数值
    hard_constraints = _build_hard_constraints(product_facts, compliance_rule_set)

    copy_structure = copy_strategy.get("copy_structure", "")
    hooks_library = copy_strategy.get("hooks_library", {})
    persona_strategies_raw: list[dict] = copy_strategy.get("persona_strategies", [])
    # 将 dict 转换为 PersonaStrategy 以获得强类型校验
    persona_strategies: list[PersonaStrategy] = []
    for ps_raw in persona_strategies_raw:
        try:
            persona_strategies.append(PersonaStrategy.model_validate(ps_raw))
        except Exception:
            persona_strategies.append(PersonaStrategy(persona_name=ps_raw.get("persona_name", "")))

    master_copies: list[dict] = []
    llm = _get_cg_llm("cg_copy_writer")

    for variant_type in variants:
        # 选取当前版本对应的客群策略
        if variant_type == "universal":
            creative_strategy = {
                "persona_name": "通用客群",
                "scene_direction": "通用场景",
                "hooks_library": hooks_library,
            }
        else:
            matched_ps = next(
                (p for p in persona_strategies if p.persona_name == variant_type),
                None,
            )
            if matched_ps:
                creative_strategy = {**matched_ps.model_dump(), "hooks_library": hooks_library}
            else:
                creative_strategy = {"persona_name": variant_type, "hooks_library": hooks_library}

        sub_state = {
            **state,
            "variant_type": variant_type,
            "hard_constraints": _safe_json_dumps(hard_constraints),
            "creative_strategy": _safe_json_dumps(creative_strategy),
            "copy_structure": copy_structure,
            "current_date": _current_date(),
            "messages": [HumanMessage(content=f"请为版本「{variant_type}」生成母版文案 JSON。")],
        }

        try:
            resp = await llm.ainvoke(
                apply_prompt_template("CG/copy_writer", sub_state, configurable)
            )
            content = str(resp.content or "")
            master_obj = parse_master_copy(content, variant_type=variant_type)
            master_copies.append(master_obj.model_dump())
        except Exception as e:
            logger.exception("copy_writer_node 版本「%s」失败: %s", variant_type, e)
            master_copies.append({
                "variant_type": variant_type,
                "body": f"[生成失败] {e}",
                "_error": str(e),
            })

        # ── AB 变体生成：当 ab_test=True 且该客群有 AB 方案时，额外生成 B 组母版 ──
        if ab_test and variant_type != "universal":
            ab_plan = next(
                (p for p in ab_plans if p.get("persona_name") == variant_type), None
            )
            if ab_plan:
                # 构建 B 组的 creative_strategy：覆盖实验变量
                b_creative_strategy = dict(creative_strategy)
                variable_tested = ab_plan.get("variable_tested", "")
                group_b_value = ab_plan.get("group_b", "")
                if variable_tested == "hook_type" and group_b_value:
                    b_creative_strategy["marketing_hooks"] = [group_b_value]
                elif variable_tested == "cta_style" and group_b_value:
                    b_creative_strategy["cta_style_override"] = group_b_value
                elif variable_tested == "tone" and group_b_value:
                    b_creative_strategy["tone"] = group_b_value
                elif variable_tested and group_b_value:
                    b_creative_strategy[variable_tested] = group_b_value

                b_sub_state = {
                    **state,
                    "variant_type": variant_type,
                    "hard_constraints": _safe_json_dumps(hard_constraints),
                    "creative_strategy": _safe_json_dumps(b_creative_strategy),
                    "copy_structure": copy_structure,
                    "current_date": _current_date(),
                    "messages": [HumanMessage(content=(
                        f"请为版本「{variant_type}」生成 B 组 AB 变体母版文案 JSON。\n"
                        f"AB 实验变量：{variable_tested}，B 组取值：{group_b_value}。\n"
                        f"请确保仅在 {variable_tested} 维度与 A 组不同，其余保持一致。"
                    ))],
                }

                try:
                    b_resp = await llm.ainvoke(
                        apply_prompt_template("CG/copy_writer", b_sub_state, configurable)
                    )
                    b_content = str(b_resp.content or "")
                    b_master_obj = parse_master_copy(b_content, variant_type=variant_type)
                    b_dict = b_master_obj.model_dump()
                    b_dict["ab_group"] = "B"
                    b_dict["ab_variable"] = variable_tested
                    b_dict["ab_label"] = group_b_value
                    master_copies.append(b_dict)
                except Exception as e:
                    logger.exception(
                        "copy_writer_node AB 变体版本「%s」失败: %s", variant_type, e
                    )
                    master_copies.append({
                        "variant_type": variant_type,
                        "body": f"[AB 变体生成失败] {e}",
                        "ab_group": "B",
                        "ab_variable": ab_plan.get("variable_tested", ""),
                        "ab_label": ab_plan.get("group_b", ""),
                        "_error": str(e),
                    })

    return {"master_copies": master_copies}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 4 — Adapt-Audit-Assemble（组合节点）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def adapt_audit_assemble_node(state: CGState, config: RunnableConfig) -> dict:
    """
    Phase 4 组合节点：对每份母版×每个渠道执行完整的审改闭环。

    内部流程（逐 master × channel）：
    1. channel_adapt  — 渠道适配（LLM 调用）
    2. copy_audit     — 合规审计（LLM 调用）
    3. rewrite loop   — 需改写时进行改写，再审计（最多 max_rewrite_iterations 次）
    4. taste_guardian — 品味筛选（LLM 调用）
    5. taste rewrite  — 品味不达标时改写（最多 max_rewrite_iterations 次）
    6. simulated_audience — 模拟受众（LLM 调用）+ 可选微调
    7. final_delta_audit  — 品味改写或微调发生后的增量审计
    8. evidence_assembler — 举证报告（LLM 调用）
    """
    configurable = Configuration.from_runnable_config(config)
    cg_input = _get_cg_input(state)

    master_copies_raw: list[dict] = state.get("master_copies", [])
    copy_strategy = state.get("copy_strategy", {})
    product_facts = state.get("product_facts", {})
    compliance_rule_set = state.get("compliance_rule_set", {})

    channels: list[str] = cg_input.get("channels", [])
    channel_specs: list[dict] = copy_strategy.get("channel_specs", [])
    persona_channel_cards: list[dict] = copy_strategy.get("persona_channel_cards", [])
    max_rewrite = state.get("max_rewrite_iterations", 2)

    final_channel_copies: list[dict] = []
    final_audit_reports: list[dict] = []
    final_evidence_reports: list[str] = []
    final_taste_scores: list[dict] = []
    final_audience_reactions: list[dict] = []

    for master_raw in master_copies_raw:
        if master_raw.get("_error"):
            logger.warning(
                "跳过生成失败的母版版本: variant_type=%s",
                master_raw.get("variant_type", "?"),
            )
            continue

        master = MasterCopy.model_validate(master_raw)

        for channel_name in channels:
            channel_spec = _find_channel_spec(channel_specs, channel_name)
            style_card = _find_style_card(persona_channel_cards, master.variant_type, channel_name)

            # ── 步骤 1：渠道适配 ──────────────────────────────────────────
            channel_copy = await _adapt_one_channel(
                state, configurable, master, channel_spec, compliance_rule_set
            )

            # ── 步骤 2-3：合规审计 + 改写闭环 ─────────────────────────────
            audit_report = await _audit_one_copy(
                state, configurable, channel_copy, master, product_facts, compliance_rule_set
            )
            rewrite_count = 0
            while (
                audit_report.overall_verdict in ("需改写",)
                and rewrite_count < max_rewrite
            ):
                channel_copy = await _rewrite_one_copy(
                    state, configurable, channel_copy, audit_report, compliance_rule_set,
                    mode="compliance",
                )
                audit_report = await _audit_one_copy(
                    state, configurable, channel_copy, master, product_facts, compliance_rule_set
                )
                rewrite_count += 1

            # 过度合规时触发创意恢复改写
            if audit_report.overall_verdict == "需创意恢复":
                channel_copy = await _rewrite_one_copy(
                    state, configurable, channel_copy, audit_report, compliance_rule_set,
                    mode="creativity_restore",
                )

            # ── 步骤 4-5：品味筛选 + 改写闭环 ─────────────────────────────
            taste_score = await _taste_check(
                state, configurable, channel_copy, style_card
            )
            taste_rewrite_count = 0
            while (
                not taste_score.overall_pass
                and taste_score.weighted_total < taste_score.threshold - 1.0
                and taste_rewrite_count < max_rewrite
            ):
                channel_copy = await _rewrite_one_copy(
                    state, configurable, channel_copy, taste_score, compliance_rule_set,
                    mode="taste",
                )
                taste_score = await _taste_check(
                    state, configurable, channel_copy, style_card
                )
                taste_rewrite_count += 1

            did_taste_rewrite = taste_rewrite_count > 0

            # ── 步骤 6：模拟受众 ──────────────────────────────────────────
            audience_reaction = await _simulate_audience(
                state, configurable, channel_copy, master.variant_type, style_card
            )

            # 受众微调（仅 final_tweaks 非策略性建议时）
            non_strategic_tweaks = [
                t for t in audience_reaction.final_tweaks
                if audience_reaction.silence_reasons
                and not any(r.is_strategic for r in audience_reaction.silence_reasons)
            ]
            did_audience_tweak = False
            if non_strategic_tweaks:
                channel_copy = _apply_audience_tweaks(channel_copy, non_strategic_tweaks)
                did_audience_tweak = True

            # ── 步骤 7：增量审计（如发生了品味改写或微调）──────────────────
            if did_taste_rewrite or did_audience_tweak:
                delta_audit = await _audit_one_copy(
                    state, configurable, channel_copy, master, product_facts, compliance_rule_set
                )
                # 如果增量审计发现新的 L1 违规，回退到 taste 改写前的版本不可行（无状态），
                # 此处记录 verdict 并附加警告
                if delta_audit.overall_verdict in ("需改写", "需人工介入"):
                    logger.warning(
                        "品味改写后增量审计发现问题: verdict=%s, channel=%s, variant=%s",
                        delta_audit.overall_verdict, channel_name, master.variant_type,
                    )
                    audit_report = delta_audit

            # ── 步骤 8：举证报告 ──────────────────────────────────────────
            evidence_report = await _assemble_evidence(
                state, configurable, channel_copy, audit_report,
                cg_input.get("product_name", ""),
            )

            # 收集结果
            final_channel_copies.append(channel_copy.model_dump())
            final_audit_reports.append(audit_report.model_dump())
            final_evidence_reports.append(evidence_report)
            final_taste_scores.append(taste_score.model_dump())
            final_audience_reactions.append(audience_reaction.model_dump())

    return {
        "channel_copies": final_channel_copies,
        "audit_reports": final_audit_reports,
        "evidence_reports": final_evidence_reports,
        "taste_scores": final_taste_scores,
        "audience_reactions": final_audience_reactions,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 4 内部 LLM 调用函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _adapt_one_channel(
    state: CGState,
    configurable: Configuration,
    master: MasterCopy,
    channel_spec: ChannelSpec | dict,
    compliance_rule_set: dict,
) -> ChannelCopy:
    """步骤 1：渠道适配（Channel Adapter）。

    适配完成后自动调用 check_consistency 校验母版与渠道版一致性，
    校验问题追加到 compression_notes 字段。
    """
    spec_dict = channel_spec.model_dump() if isinstance(channel_spec, ChannelSpec) else channel_spec
    l2_rules = compliance_rule_set.get("l2_rules", [])
    channel_name = spec_dict.get("channel_name", "")
    channel_rules = compliance_rule_set.get("channel_rules", {}).get(channel_name, [])

    sub_state = {
        **state,
        "master_copy": _safe_json_dumps(master.model_dump()),
        "channel_spec": _safe_json_dumps(spec_dict),
        "l2_rules": _safe_json_dumps(l2_rules),
        "channel_rules": json.dumps(channel_rules, ensure_ascii=False),
        "current_date": _current_date(),
        "messages": [HumanMessage(content=f"请将母版适配为「{channel_name}」渠道版本。")],
    }

    llm = _get_cg_llm("cg_channel_adapter")
    try:
        resp = await llm.ainvoke(
            apply_prompt_template("CG/channel_adapter", sub_state, configurable)
        )
        channel_copy = parse_channel_copy(
            str(resp.content or ""),
            variant_type=master.variant_type,
            channel_name=channel_name,
        )
    except Exception as e:
        logger.exception("_adapt_one_channel 失败 channel=%s: %s", channel_name, e)
        return ChannelCopy(
            variant_type=master.variant_type,
            channel_name=channel_name,
            copy_text=f"[渠道适配失败] {e}",
            char_count=0,
        )

    # 一致性校验：核心数字/条件/产品名称与母版一致
    consistency_issues = check_consistency(master, channel_copy)
    if consistency_issues:
        notes = channel_copy.compression_notes or ""
        notes += " | 一致性问题: " + "; ".join(consistency_issues)
        channel_copy = channel_copy.model_copy(update={"compression_notes": notes})
        logger.warning(
            "渠道适配一致性校验发现 %d 个问题 (channel=%s, variant=%s): %s",
            len(consistency_issues), channel_name, master.variant_type,
            "; ".join(consistency_issues),
        )

    return channel_copy


async def _audit_one_copy(
    state: CGState,
    configurable: Configuration,
    channel_copy: ChannelCopy,
    master: MasterCopy,
    product_facts: dict,
    compliance_rule_set: dict,
) -> AuditReport:
    """步骤 2：合规审计（Copy Auditor）。"""
    sub_state = {
        **state,
        "copy_text": channel_copy.copy_text,
        "variant_type": channel_copy.variant_type,
        "channel_name": channel_copy.channel_name,
        "product_facts": _safe_json_dumps(product_facts),
        "compliance_rules": _safe_json_dumps(compliance_rule_set),
        "master_copy": _safe_json_dumps(master.model_dump()),
        "current_date": _current_date(),
        "cg_audit_edge_checklist": render_cg_audit_edge_checklist(),
        "messages": [HumanMessage(content="请对上述渠道版文案进行五维合规审计，输出 AuditReport JSON。")],
    }

    llm = _get_cg_llm("cg_copy_auditor")
    try:
        resp = await llm.ainvoke(
            apply_prompt_template("CG/copy_auditor", sub_state, configurable)
        )
        return parse_audit_report(str(resp.content or ""))
    except Exception as e:
        logger.exception("_audit_one_copy 失败: %s", e)
        return AuditReport(
            variant_type=channel_copy.variant_type,
            channel_name=channel_copy.channel_name,
            overall_verdict="需人工介入",
            issues=[f"审计失败: {e}"],
        )


async def _rewrite_one_copy(
    state: CGState,
    configurable: Configuration,
    channel_copy: ChannelCopy,
    audit_or_taste_report: Any,
    compliance_rule_set: dict,
    mode: str = "compliance",
) -> ChannelCopy:
    """步骤 3/5：改写（Copy Rewriter）。

    mode: "compliance" | "taste" | "creativity_restore"
    """
    sub_state = {
        **state,
        "mode": mode,
        "copy_text": channel_copy.copy_text,
        "audit_or_taste_report": _safe_json_dumps(
            audit_or_taste_report.model_dump()
            if hasattr(audit_or_taste_report, "model_dump")
            else audit_or_taste_report
        ),
        "compliance_rules": _safe_json_dumps(compliance_rule_set),
        "current_date": _current_date(),
        "messages": [HumanMessage(content=f"请以 {mode} 模式改写文案，输出 ChannelCopy JSON。")],
    }

    llm = _get_cg_llm("cg_copy_rewriter")
    try:
        resp = await llm.ainvoke(
            apply_prompt_template("CG/copy_rewriter", sub_state, configurable)
        )
        rewritten = parse_channel_copy(
            str(resp.content or ""),
            variant_type=channel_copy.variant_type,
            channel_name=channel_copy.channel_name,
        )
        # 保留原文案的 variant_type / channel_name（防止 LLM 遗漏）
        if not rewritten.variant_type:
            rewritten = rewritten.model_copy(update={"variant_type": channel_copy.variant_type})
        if not rewritten.channel_name:
            rewritten = rewritten.model_copy(update={"channel_name": channel_copy.channel_name})
        return rewritten
    except Exception as e:
        logger.exception("_rewrite_one_copy mode=%s 失败: %s", mode, e)
        return channel_copy  # 改写失败时原样返回


async def _taste_check(
    state: CGState,
    configurable: Configuration,
    channel_copy: ChannelCopy,
    style_card: PersonaChannelStyleCard | dict,
) -> TasteScore:
    """步骤 4：品味筛选（Taste Guardian）。"""
    card = style_card.model_dump() if isinstance(style_card, PersonaChannelStyleCard) else style_card

    taste_profile = card.get("taste_profile", "")
    forbidden_expressions = card.get("forbidden_phrases", [])
    positive_examples = card.get("example_good", [])
    negative_examples = card.get("example_bad", [])
    familiar_register = card.get("familiar_register", "")
    max_hooks = card.get("max_hooks", 2)
    cta_style = card.get("cta_style", "")

    sub_state = {
        **state,
        "copy_text": channel_copy.copy_text,
        "channel_name": channel_copy.channel_name,
        "variant_type": channel_copy.variant_type,
        "taste_profile": taste_profile,
        "forbidden_expressions": json.dumps(forbidden_expressions, ensure_ascii=False),
        "positive_examples": json.dumps(positive_examples, ensure_ascii=False),
        "negative_examples": json.dumps(negative_examples, ensure_ascii=False),
        "familiar_register": familiar_register,
        "max_hooks": str(max_hooks),
        "cta_style": cta_style,
        "current_date": _current_date(),
        "messages": [HumanMessage(content="请对文案进行品味评分，输出 TasteScore JSON。")],
    }

    llm = _get_cg_llm("cg_taste_guardian")
    try:
        resp = await llm.ainvoke(
            apply_prompt_template("CG/taste_guardian", sub_state, configurable)
        )
        return parse_taste_score(str(resp.content or ""))
    except Exception as e:
        logger.exception("_taste_check 失败: %s", e)
        return TasteScore(overall_pass=True, weighted_total=7.0, threshold=7.0)


async def _simulate_audience(
    state: CGState,
    configurable: Configuration,
    channel_copy: ChannelCopy,
    variant_type: str,
    style_card: PersonaChannelStyleCard | dict,
) -> AudienceReaction:
    """步骤 6：模拟受众（Simulated Audience）。"""
    card = style_card.model_dump() if isinstance(style_card, PersonaChannelStyleCard) else style_card
    persona_profile = card if card else {"persona_name": variant_type}
    channel_context_map = {
        "企业微信": "企微私聊",
        "全民生活APP": "APP 详情页",
        "短信": "短信推送",
        "APP Push": "APP Push 通知",
    }
    channel_context = channel_context_map.get(channel_copy.channel_name, channel_copy.channel_name)

    sub_state = {
        **state,
        "copy_text": channel_copy.copy_text,
        "channel_name": channel_copy.channel_name,
        "variant_type": variant_type,
        "persona_profile": _safe_json_dumps(persona_profile),
        "channel_context": channel_context,
        "current_date": _current_date(),
        "messages": [HumanMessage(content="请模拟目标客群对该文案的反应，输出 AudienceReaction JSON。")],
    }

    llm = _get_cg_llm("cg_simulated_audience")
    try:
        resp = await llm.ainvoke(
            apply_prompt_template("CG/simulated_audience", sub_state, configurable)
        )
        return parse_audience_reaction(str(resp.content or ""))
    except Exception as e:
        logger.exception("_simulate_audience 失败: %s", e)
        return AudienceReaction(
            variant_type=variant_type,
            channel_name=channel_copy.channel_name,
            click_willingness=5.0,
            net_score=5.0,
        )


async def _assemble_evidence(
    state: CGState,
    configurable: Configuration,
    channel_copy: ChannelCopy,
    audit_report: AuditReport,
    product_name: str,
) -> str:
    """步骤 8：举证报告组装（Evidence Assembler）。"""
    l2_check = audit_report.l2_disclosure_check
    consistency_check = audit_report.consistency_check
    atomic_fact_claims = [c.model_dump() for c in audit_report.atomic_fact_claims]
    compliance_evidences = [e.model_dump() for e in audit_report.compliance_evidences]

    sub_state = {
        **state,
        "channel_copy": channel_copy.copy_text,
        "variant_type": channel_copy.variant_type,
        "channel_name": channel_copy.channel_name,
        "product_name": product_name,
        "overall_verdict": audit_report.overall_verdict,
        "enriched_claims": _safe_json_dumps(atomic_fact_claims),
        "enriched_evidences": _safe_json_dumps(compliance_evidences),
        "l2_check": _safe_json_dumps(l2_check),
        "consistency_check": _safe_json_dumps(consistency_check),
        "messages": [HumanMessage(content="请组装合规举证报告，直接输出 Markdown 格式。")],
    }

    llm = _get_cg_llm("cg_evidence_assembler")
    try:
        resp = await llm.ainvoke(
            apply_prompt_template("CG/evidence_assembler", sub_state, configurable)
        )
        return str(resp.content or "（举证报告为空）")
    except Exception as e:
        logger.exception("_assemble_evidence 失败: %s", e)
        return f"[举证报告生成失败] {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _last_ai_content(messages: list) -> str:
    """从消息列表中提取最后一条 AI 消息的文本内容。"""
    for m in reversed(messages or []):
        if isinstance(m, AIMessage):
            return str(m.content or "")
        if isinstance(m, dict) and m.get("role") in ("assistant", "ai"):
            return str(m.get("content", ""))
    return ""


def _find_channel_spec(channel_specs: list[dict], channel_name: str) -> dict:
    """从 channel_specs 中查找指定渠道的规范，不存在时返回默认规范。"""
    for spec in channel_specs:
        if spec.get("channel_name") == channel_name:
            return spec
    return {"channel_name": channel_name, "max_chars": 200, "style": "中性"}


def _find_style_card(
    persona_channel_cards: list[dict],
    variant_type: str,
    channel_name: str,
) -> dict:
    """从 persona_channel_cards 中查找指定（客群, 渠道）的风格卡。"""
    for card in persona_channel_cards:
        if (
            card.get("persona_name") == variant_type
            and card.get("channel_name") == channel_name
        ):
            return card
    # fallback：仅匹配渠道
    for card in persona_channel_cards:
        if card.get("channel_name") == channel_name:
            return card
    return {"persona_name": variant_type, "channel_name": channel_name}


def _build_hard_constraints(product_facts: dict, compliance_rule_set: dict) -> dict:
    """构建 Copy Writer 的铁框架（hard_constraints）输入。"""
    l1_rules = compliance_rule_set.get("l1_rules", [])
    l2_rules = compliance_rule_set.get("l2_rules", [])
    l3_rules = compliance_rule_set.get("l3_rules", [])

    return {
        "product_facts_summary": {
            "product_name": product_facts.get("product_name", ""),
            "benefits": product_facts.get("benefits", []),
            "risk_disclosures": product_facts.get("risk_disclosures", []),
            "customer_service_hotline": product_facts.get("customer_service_hotline", ""),
        },
        "l1_rules": l1_rules,
        "l2_mandatory_disclosures": l2_rules,
        "l3_cautious_rules": l3_rules,
    }


def _apply_audience_tweaks(channel_copy: ChannelCopy, tweaks: list[str]) -> ChannelCopy:
    """将受众模拟的微调建议应用到文案文本（轻量字符串替换）。

    仅处理「把XX改为YY」格式的指令，其余格式跳过。
    """
    copy_text = channel_copy.copy_text
    for tweak in tweaks:
        # 尝试匹配「把XX改为YY」或「将XX改为YY」格式
        import re
        m = re.search(r"[把将](.+?)改为[「「]?(.+?)[」」]?$", tweak)
        if m:
            original = m.group(1).strip()
            replacement = m.group(2).strip()
            if original in copy_text:
                copy_text = copy_text.replace(original, replacement, 1)
                logger.info("受众微调：「%s」→「%s」", original, replacement)

    return channel_copy.model_copy(update={
        "copy_text": copy_text,
        "char_count": len(copy_text),
    })


def _build_skill_injection(skills: list[str]) -> str:
    """将已加载的经验 Skill 列表格式化为 Prompt 注入文本。"""
    if not skills:
        return ""
    parts = ["## 经验库（来自历史投放验证）", "以下经验经过多次实际投放验证，请在策略规划中优先参考。"]
    for i, skill_content in enumerate(skills, 1):
        parts.append(f"### 经验 {i}\n{skill_content}\n---")
    return "\n".join(parts)
