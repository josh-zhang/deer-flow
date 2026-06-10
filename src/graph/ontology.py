# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""信用卡面客知识本体（Ontology）数据模块 —— P1 施工（2026-06-10）。

单一数据源（Single Source of Truth）：
- 内容转写自《中国民生银行零售信用卡面客知识本体架构》（版本 2026），
  源文件：~/notes/my_notes/hermes/01-银行零售业务百科/knowledge-ontology/
         中国民生银行零售信用卡面客知识本体架构.md
- 本体维护方式：人工修订上述 md 后同步本模块（手册"场景 A"——零代码风控兼容，
  新业务上线只补实体/边，不改流水线代码）。

消费方：
- Ontology Mapper（原 Background Investigator 节点）：注入 render_skeleton_for_mapper()
  产出的"类+锚点词+边拓扑"骨架（不含约束全文，省 token）。
- Planner 约束注入（Constraint Injection）：render_planner_guardrail() 仅渲染
  **命中边子集**的约束解释，遵守"命中子集注入"原则。

设计纪律（务实约束）：
- 不建图数据库；本体即文本 + 查表。
- domain: credit_card（信用卡子域专属层）。跨域内容须回零售通用层
  （biz-wiki/schema.md），本模块不得越界使用。
"""

from __future__ import annotations

import re

# ── 生命周期四阶段 ──────────────────────────────────────────────

LIFECYCLE_STAGES: list[tuple[str, str]] = [
    ("Acquisition", "获客/办卡期：卡片宣传、办卡条件、新户礼"),
    ("Activation & Usage", "激活与用卡期：首刷开卡、绑卡、支付交易、查账、还款、基础权益"),
    ("Growth & Monetization", "成长与创利期：分期办理、额度提升、积分兑换、循环信用"),
    ("Servicing & Retention", "服务与留存期：换卡、挂失、资料修改、息费争议、销卡挽留"),
]

# ── 16 个核心实体类（名称, 定义, 模糊匹配锚点词, 生命周期）────────

CLASSES: list[dict] = [
    {"name": "Product_Card", "desc": "信用卡产品本身",
     "anchors": "名片、普/金/白金/钻石、联名卡、免年费、刚性年费", "stage": "Acquisition"},
    {"name": "Card_Tier", "desc": "卡片刚性物理等级（决定权益发配体系）",
     "anchors": "普卡/金卡、标准白金卡(标白)、豪华白金卡(豪白)、钻石卡、百夫长黑金卡", "stage": "贯穿全周期"},
    {"name": "Lifecycle_Stage", "desc": "客户/卡片/账户的当前状态",
     "anchors": "未激活、睡眠户、逾期、销卡态、止付、冻结、呆账", "stage": "贯穿全周期"},
    {"name": "Right_Entity", "desc": "持卡人长期恒定权益及高端服务",
     "anchors": "非凡礼遇、机场/高铁贵宾厅(龙腾)、接送机、洗牙/专家挂号、高尔夫、延误险、道路救援", "stage": "Usage / Retention"},
    {"name": "Currency_Token", "desc": "行内流转的虚拟通货/核销介质",
     "anchors": "积点(民生高端权益专属)、通用积分、联名方积分(如东航里程)、资质(次/点)", "stage": "Growth / Usage"},
    {"name": "Campaign", "desc": "限定期的营销活动",
     "anchors": "新户礼、首刷送、满减、随机抽奖、特惠商圈、瓜分积分、报名参与", "stage": "贯穿全周期"},
    {"name": "Campaign_Period", "desc": "活动时间与周期属性",
     "anchors": "自然月、账单月、活动达标期、活动领奖期、每周五、节假日", "stage": "贯穿全周期"},
    {"name": "Reward_Entity", "desc": "活动奖励/礼品的物理或虚拟形式",
     "anchors": "立减金、刷卡金、还款金、多倍积分、实物/行李箱、视频会员、接送机券", "stage": "贯穿全周期"},
    {"name": "Brand_Merchant", "desc": "参与活动的外部品牌或商户",
     "anchors": "星巴克、海底捞、中石油、山姆会员店、京东、指定商圈", "stage": "Usage"},
    {"name": "Service_Operation", "desc": "面客服务办理与账户维护",
     "anchors": "激活、密码重置、补换卡、挂失、解除挂失/止付、改预留手机号、升降额、销户、提前结清/还款、延期还款", "stage": "Activation / Servicing"},
    {"name": "Customer_Service", "desc": "面客服务与查询矩阵",
     "anchors": "咨询与投诉、办卡进度查询、额度/账单/积分查询、用卡百科、用卡须知、信息处理合作机构", "stage": "贯穿全周期"},
    {"name": "Financial_Transaction", "desc": "资金流动、交易、还款及分期",
     "anchors": "绑卡(第三方)、扫码支付、账单分期、现金分期、预借现金、按期还款、最低还款、溢缴款领回", "stage": "Usage / Growth"},
    {"name": "Installment_Product", "desc": "分期信贷产品类型",
     "anchors": "账单分期、自由分期/单笔分期、全民乐分期、现金分期、汽车分期/车位分期、商场分期", "stage": "Growth"},
    {"name": "Channel", "desc": "办卡或获取服务的自营及第三方平台",
     "anchors": "全民生活APP、微信公众号、信用卡小程序、支付宝、财付通、云闪付、网点、客服热线(IVR/人工)", "stage": "贯穿全周期"},
    {"name": "Verification_Method", "desc": "实名核身与安全鉴权要素",
     "anchors": "刷脸/人脸识别、短信验证码/OTP、交易密码、查询密码、CVV2、卡片有效期、身份证正反面、预留手机号验证、设备绑定/免密", "stage": "贯穿全周期"},
    {"name": "Audience", "desc": "面向的限制客群",
     "anchors": "新户、老客、受邀客户、资产达标(私行)", "stage": "贯穿全周期"},
    {"name": "Policy_Rule", "desc": "操作背后的核心规则与红线",
     "anchors": "年费收取规则、违约金规则、分期手续费、提前还款违约金、取现利息、宽限期/容时容差、隐私授权、催收规范、征信报送规则", "stage": "贯穿全周期"},
]

# ── ~20 条有向关系边（relation → 条目列表；同名 relation 可有多条）──

EDGES: dict[str, list[dict]] = {
    "belongs_to_tier": [{
        "src": "Product_Card", "dst": "Card_Tier",
        "constraint": "判定高端卡年费和权益的源头。严禁凭'白金卡'三字乱承诺——必须先查明是'标白''豪白''精英白'还是'百夫长白'，不同 Tier 对应完全不同的权益发配体系。",
    }],
    "issued_with": [{
        "src": "Card_Tier / Product_Card", "dst": "Currency_Token",
        "constraint": "民生高端权益核心机制：很多高端卡按卡等级每年统一发放'积点'，而非刚性分配权益次数。必须查明该卡的通货（积点/积分）发放机制。",
    }],
    "can_redeem": [{
        "src": "Currency_Token", "dst": "Right_Entity",
        "constraint": "核销边。必须查明：特定权益需扣除多少积点/积分；可否花钱加购积点；积点的有效期。",
    }],
    "redeemed_by": [{
        "src": "Right_Entity", "dst": "Service_Operation / Channel",
        "constraint": "权益核销路径必须明确（全民生活APP专区预订还是贵宾客服专线）；部分权益有提前48小时预约等硬性限制；合作商户可能调整或暂停，必须检索最新公告类规则。",
    }],
    "changes_state": [{
        "src": "Service_Operation", "dst": "Lifecycle_Stage",
        "constraint": "异常卡片处理生命线：挂失/伪冒报案导致'止付/冻结'态，'解除止付'的前置条件必须查明。消保重灾区在账户异常时的操作。",
    }],
    "authenticates_with": [{
        "src": "Service_Operation / Financial_Transaction", "dst": "Verification_Method",
        "constraint": "线上业务办理生死线：敏感操作（改预留手机号/消费分期/无卡快捷支付/密码重置）必须强制审查核身方式组合；渠道不支持该核身强度则必定引流至线下网点。",
    }],
    "relates_to_installment": [{
        "src": "Financial_Transaction", "dst": "Installment_Product",
        "constraint": "触发分期必须明确类型：'全民乐分期'独属民生体系，其额度占用、手续费率模型、提前结清规则均与'自由分期/账单分期'截然不同，严禁混为一谈。",
    }],
    "absorbs_quota_from": [{
        "src": "Installment_Product", "dst": "Policy_Rule（额度占用规则）",
        "constraint": "分期高发投诉点：必须核实占用'信用卡固定额度'还是'独立专项额度/外呼白名单额度'——直接决定客户办完分期后能否继续刷卡消费。",
    }],
    "routed_through": [
        {
            "src": "Service_Operation", "dst": "Channel",
            "constraint": "非所有业务都能线上办：高危业务（如解除反欺诈冻结）可能仅限'网点'渠道，必须查明渠道可用性列表。",
        },
        {
            "src": "Financial_Transaction（快捷支付）", "dst": "Channel",
            "constraint": "快捷支付判定：明确交易发生于微信支付/支付宝/银联通道——这决定能否享受特定积分权益或命中黑名单规则。",
        },
    ],
    "triggers": [{
        "src": "Financial_Transaction（还款）", "dst": "Policy_Rule（宽限期/容时容差）",
        "constraint": "最核心投诉区：任何涉及还款/逾期的场景，必须强制扫描该卡'容时容差金额'与'3天宽限期'是否生效，严防暴力违约金。",
    }],
    "costs": [{
        "src": "Financial_Transaction（分期/提前还款）", "dst": "Policy_Rule（息费/违约规则）",
        "constraint": "涉及分期或'提前结清'，不仅查基础手续费，必须查明'提前结清是否收取剩余期数手续费/违约金'——监管重点严查项。",
    }],
    "has_fee_rules": [{
        "src": "Product_Card", "dst": "Policy_Rule（年费收取规则）",
        "constraint": "年费投诉高发：必须核实是'刚性年费（不可免）'还是'条件豁免（刷N笔免）'，以及是否处于容时豁免期。",
    }],
    "authorizes": [{
        "src": "Customer_Service", "dst": "Policy_Rule（数据共享与隐私边界）",
        "constraint": "隐私边界：涉及材料共享、催收催告等场景，必须核实隐私政策与《信息处理合作机构》名录；名录外的数据流转视为严重合规幻觉。",
    }],
    "has_timeline": [{
        "src": "Campaign", "dst": "Campaign_Period",
        "constraint": "任何营销活动必须强制分离'参与时间''达标时间''领奖时间'——客户以为参与即得奖、规则约定次月才能领是消保投诉高发点。",
    }],
    "resolves_via": [{
        "src": "Customer_Service", "dst": "Channel（投诉受理途径）",
        "constraint": "升级投诉、息费结清争议必须关联正确受理渠道（如消保专线），而非普通客服机器人死循环。",
    }],
    "requires_action": [{
        "src": "Campaign", "dst": "Financial_Transaction / Service_Operation",
        "constraint": "达标条件解构：除'消费满额'外，达标条件还可能包含服务操作（如'需在全民生活APP报名''首次绑定微信支付'）。",
    }],
    "grants_reward": [{
        "src": "Campaign", "dst": "Reward_Entity",
        "constraint": "明确礼品类型：严防'刷卡金'与'还款金'混淆——两者在银行财务入账核销规则上大相径庭。",
    }],
    "partnered_with": [{
        "src": "Campaign / Reward_Entity", "dst": "Brand_Merchant",
        "constraint": "必须关联到对应第三方商户，并强制排查渠道限定：活动可能仅在指定商户的微信小程序内生效，客户在实体店刷卡通常不算。",
    }],
}

# ── 渲染：Mapper 骨架（类 + 锚点词 + 边拓扑，不含约束全文）─────────

# ── P3 辅助：12 类要素 ↔ 16 实体类映射（CP Analyzer 用）─────────

ELEMENT_CLASS_MAP: dict[str, list[str]] = {
    "1-产品信息":  ["Product_Card", "Card_Tier", "Audience"],
    "2-费率信息":  ["Policy_Rule"],
    "3-活动信息":  ["Campaign", "Campaign_Period"],
    "4-权益信息":  ["Right_Entity", "Currency_Token", "Reward_Entity"],
    "5-营销表述":  [],  # 无直接实体类映射，靠边链组合风险查漏
    "6-数据引用":  [],
    "7-合规标识":  [],
    "8-渠道信息":  ["Channel"],
    "9-第三方信息": ["Brand_Merchant"],
    "10-法律条款": ["Policy_Rule"],
    "11-交互要素": [],
    "12-材料形态": ["Channel"],
}

# ── P3 辅助：边→消保审查关注点映射（CP Analyzer 边链查漏用）─────

EDGE_CP_HINTS: dict[str, str] = {
    "has_timeline":          "参与/达标/领奖时间未分离披露 → 审查点27子项",
    "grants_reward":         "刷卡金/还款金/立减金混淆表述 → 审查点27",
    "partnered_with":        "活动渠道限定（仅小程序生效）未明示 → 审查点23",
    "triggers":              "容时容差/宽限期未提示 → 审查点18/20",
    "has_fee_rules":         "刚性年费 vs 条件豁免混淆 → 审查点18",
    "authorizes":            "信息共享超出《信息处理合作机构》名录 → 审查点1",
    "requires_action":       "达标条件（报名/绑卡等前置操作）未明示 → 审查点23/27",
    "belongs_to_tier":       "卡等级与权益发配错配 → 审查点12",
    "issued_with":           "积点/积分机制省略 → 审查点22/23",
    "can_redeem":            "核销成本/有效期未披露 → 审查点23",
    "redeemed_by":           "核销路径/预约限制未披露 → 审查点22/23",
    "relates_to_installment":"分期类型混淆 → 审查点8/9",
    "absorbs_quota_from":    "额度占用规则未明示 → 审查点19/23",
    "costs":                 "手续费/提前结清违约金未披露 → 审查点18/20",
    "authenticates_with":    "核身方式未说明 → 审查点6",
    "routed_through":        "渠道限定未明示 → 审查点5/23",
    "changes_state":         "账户异常态操作风险 → 审查点9",
    "resolves_via":          "投诉受理渠道缺失 → 审查点25",
}


def render_element_class_mapping() -> str:
    """渲染 12 类要素→16 实体类映射表（CP Analyzer 1A/1B 步骤用）。"""
    lines = ["| 要素类型 | 对应本体实体类 |", "|:---|:---|"]
    for etype, classes in ELEMENT_CLASS_MAP.items():
        cls_str = ", ".join(f"`{c}`" for c in classes) if classes else "（无直接映射，靠边链查漏）"
        lines.append(f"| {etype} | {cls_str} |")
    return "\n".join(lines)


def render_edge_cp_hints() -> str:
    """渲染边→消保审查关注点提示表（CP Analyzer 1C / 2B 步骤用）。"""
    lines = ["| 本体边 | 消保风险提示 |", "|:---|:---|"]
    for edge, hint in EDGE_CP_HINTS.items():
        lines.append(f"| `{edge}` | {hint} |")
    return "\n".join(lines)


# ── P3 辅助：Arbitrator 按边分桶 + 特指链 ───────────────────────

# Card_Tier 特指链：越右越特指
TIER_SPECIFICITY: list[str] = [
    "全卡通用",
    "普卡", "金卡",
    "标准白金卡", "豪华白金卡", "精英白金卡",
    "钻石卡", "百夫长黑金卡",
]


def render_arbitrator_bucketing_guide() -> str:
    """渲染 Arbitrator 的按边分桶指引与 Card_Tier 特指链。"""
    lines = [
        "## 本体辅助——按边分桶与特指链",
        "",
        "### 矛盾候选分桶规则",
        "只有挂在**同一条本体关系边**上的原子规则才构成矛盾候选。"
        "每条原子规则携带的 `ontology_edge` 标签标明其所属边；"
        "无标签或标签为\"none\"的规则归入\"通用桶\"，仅与同为通用桶的规则比对。",
        "分桶后在桶内执行标准矛盾识别流程（第二步），不做跨桶比对。",
        "",
        "### Card_Tier 特指链（优先级 2 辅助判定）",
        "当两条规则分属不同卡等级时，以下列表越右侧越特指（特指优于通用）：",
        "",
        " → ".join(TIER_SPECIFICITY),
        "",
        "例：\"白金卡规则\" 特指于 \"全卡通用规则\"；\"钻石卡规则\" 特指于 \"白金卡规则\"。"
        "此链为 `belongs_to_tier` 边的层级体现，仲裁时不再靠模型语感判断谁特指谁。",
    ]
    return "\n".join(lines)


# ── P3 辅助：CG Copy Auditor 按边对账 checklist ──────────────────

CG_EDGE_AUDIT_CHECKLIST: dict[str, str] = {
    "has_timeline":          "活动提到时间 → 必须区分参与/达标/领奖三段时间",
    "grants_reward":         "活动提到奖励 → 严禁刷卡金/还款金/立减金混淆",
    "partnered_with":        "涉及第三方品牌 → 必须标注渠道限定（仅小程序/仅线下等）",
    "requires_action":       "涉及达标条件 → 必须明示前置操作（报名/绑卡等）",
    "costs":                 "涉及分期/提前还款 → 手续费和提前结清违约金必须披露",
    "absorbs_quota_from":    "涉及分期产品 → 额度占用规则（固定额度/独立专项）必须明示",
    "relates_to_installment":"涉及分期 → 必须明确分期产品类型（全民乐/账单分期等），严禁混淆",
    "has_fee_rules":         "涉及年费 → 刚性年费/条件豁免必须区分",
    "issued_with":           "涉及积点/积分 → 发放机制/扣减标准/有效期必须说明",
    "can_redeem":            "涉及权益核销 → 核销所需积点/积分数量和可否购买必须说明",
    "redeemed_by":           "涉及权益使用 → 核销路径（APP/电话）和预约限制必须明确",
    "belongs_to_tier":       "涉及卡等级 → 权益发配因 Tier 而异，严禁跨等级承诺",
}


def render_cg_audit_edge_checklist() -> str:
    """渲染 CG Copy Auditor 的按边对账 checklist。"""
    lines = [
        "## 本体边触发检查清单（维度 1 补充）",
        "",
        "当文案涉及以下本体边对应的业务概念时，必须执行该检查项。"
        "这是封闭式对账——逐边核对，不遗漏：",
        "",
        "| 触发条件（文案涉及的概念） | 必须检查 |",
        "|:---|:---|",
    ]
    for edge, check in CG_EDGE_AUDIT_CHECKLIST.items():
        lines.append(f"| `{edge}` | {check} |")
    return "\n".join(lines)


# ── P3 便捷注入：CP Analyzer 模板变量字典 ────────────────────────

def get_cp_analyzer_template_vars() -> dict[str, str]:
    """返回 CP Analyzer 提示词所需的全部本体模板变量。

    CP 节点代码实现时，在 sub_state 中 `sub_state.update(get_cp_analyzer_template_vars())`
    即可一行注入 {{ element_class_mapping }} 和 {{ edge_cp_hints }} 两个变量。
    """
    return {
        "element_class_mapping": render_element_class_mapping(),
        "edge_cp_hints": render_edge_cp_hints(),
    }


def render_skeleton_for_mapper() -> str:
    """渲染注入 Ontology Mapper 提示词的本体骨架。

    只含：生命周期阶段、实体类（含模糊匹配锚点词）、边拓扑（不含约束解释全文）。
    约束全文留给 Planner 护栏按命中子集注入，避免 Mapper 提示词膨胀。
    """
    lines: list[str] = ["### 生命周期阶段（Lifecycle Stages）"]
    for name, desc in LIFECYCLE_STAGES:
        lines.append(f"- **{name}**：{desc}")

    lines.append("")
    lines.append("### 实体类白名单（16 个 Class，禁止超出此范围）")
    lines.append("| Class | 定义 | 模糊匹配锚点词 |")
    lines.append("|:---|:---|:---|")
    for c in CLASSES:
        lines.append(f"| `{c['name']}` | {c['desc']} | {c['anchors']} |")

    lines.append("")
    lines.append("### 关系边白名单（有向边，禁止超出此范围）")
    for rel, entries in EDGES.items():
        for e in entries:
            lines.append(f"- `{e['src']}` → `{rel}` → `{e['dst']}`")
    return "\n".join(lines)


# ── 抽取：从 LLM 输出中提取 <ontology_mapping> 块与命中边 ──────────

_MAPPING_RE = re.compile(r"<ontology_mapping>.*?</ontology_mapping>", re.DOTALL)
_RELATION_RE = re.compile(r"<Relation>(.*?)</Relation>", re.DOTALL)
_EDGE_NAME_RE = re.compile(r"->\s*([A-Za-z_]+)\s*->")


def extract_ontology_mapping(text: str) -> str:
    """从模型输出中提取 <ontology_mapping>…</ontology_mapping> 块（防模型附带闲聊）。"""
    m = _MAPPING_RE.search(text or "")
    return m.group(0).strip() if m else ""


def extract_relations(mapping_text: str) -> list[str]:
    """从映射 XML 中提取命中的关系边名（去重保序，仅保留白名单内的边）。"""
    hits: list[str] = []
    for rel_line in _RELATION_RE.findall(mapping_text or ""):
        m = _EDGE_NAME_RE.search(rel_line)
        if m:
            name = m.group(1).strip()
            if name in EDGES and name not in hits:
                hits.append(name)
    return hits


# ── 渲染：Planner 约束注入护栏（仅命中边子集）────────────────────


def render_planner_guardrail(mapping_text: str) -> str:
    """根据本体映射 XML 渲染 Planner 约束注入块。

    仅渲染命中边的约束解释（命中子集注入）。无命中边时返回空串。
    """
    hits = extract_relations(mapping_text)
    if not hits:
        return ""

    lines = [
        "## 本体合规护栏（Constraint Injection）",
        "系统已通过本体映射识别本次调查命中以下关系边。根据本体合规红线，"
        "你制定的调查计划必须逐一覆盖验证下列约束——**每条命中边至少对应一个 "
        "research Step 的检索标的**：",
        "",
    ]
    n = 0
    for rel in hits:
        for e in EDGES[rel]:
            n += 1
            lines.append(
                f"{n}. **[{rel}]** `{e['src']}` → `{e['dst']}`：{e['constraint']}"
            )
    lines.append("")
    lines.append(
        "若对上述维度界定错误或遗漏检索，将引发金融监管红线投诉。"
        "本体映射 <unmapped> 中的要素不受上述约束，但必须安排探索性检索步骤覆盖。"
    )
    return "\n".join(lines)


# ── 自检 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    skeleton = render_skeleton_for_mapper()
    assert "Product_Card" in skeleton and "relates_to_installment" in skeleton
    print(f"[ok] skeleton rendered: {len(skeleton)} chars, "
          f"{len(CLASSES)} classes, {sum(len(v) for v in EDGES.values())} edges")

    sample = """前置闲聊应被剔除。
<ontology_mapping>
  <Lifecycle_Stage>Growth & Monetization (创利期)</Lifecycle_Stage>
  <Product_Card>民生信用卡</Product_Card>
  <Financial_Transaction>分期业务办理</Financial_Transaction>
  <Installment_Product>全民乐分期</Installment_Product>
  <Policy_Rule>独立专用额度, 计息及提前还款违约金规则</Policy_Rule>
  <Relation>Financial_Transaction -> relates_to_installment -> Installment_Product</Relation>
  <Relation>Installment_Product -> absorbs_quota_from -> Policy_Rule</Relation>
  <Relation>Financial_Transaction -> costs -> Policy_Rule</Relation>
  <Relation>Fake_Class -> not_an_edge -> Nowhere</Relation>
  <unmapped>无</unmapped>
</ontology_mapping>
后置闲聊也应被剔除。"""
    mapping = extract_ontology_mapping(sample)
    assert mapping.startswith("<ontology_mapping>") and mapping.endswith("</ontology_mapping>")
    rels = extract_relations(mapping)
    assert rels == ["relates_to_installment", "absorbs_quota_from", "costs"], rels
    guardrail = render_planner_guardrail(mapping)
    assert "全民乐分期" in guardrail and "[costs]" in guardrail and "not_an_edge" not in guardrail
    print(f"[ok] mapping extracted ({len(mapping)} chars), hits={rels}")
    print(f"[ok] guardrail rendered: {len(guardrail)} chars")
    assert render_planner_guardrail("no mapping here") == ""
    print("[ok] empty guardrail on no-hit input")
    print("ALL SELF-TESTS PASSED")
