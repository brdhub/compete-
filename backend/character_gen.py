"""角色生成：调用 Anthropic Claude，开启服务端 web search 工具，
联网搜索双方名字的真实信息，再生成主题贴合的血量/技能等属性。

设计要点：
- 用一个客户端工具 `submit_characters` 约束 Claude 输出严格的 JSON 结构，
  避免从自由文本里解析 JSON 的脆弱性。
- web search 是服务端工具，Claude 会在同一轮里自动联网搜索，
  搜索完成后再调用 submit_characters 提交结果。
- 没有 API key 时自动降级到 mock，保证 demo 在无网络/无密钥下也能跑。
"""
from __future__ import annotations

import os
import json as _json
import re as _re
import random
from typing import Tuple

from models import Character, Skill


# ---------------------------------------------------------------------------
# 文本 JSON 解析兜底：DeepSeek 等推理模型在 thinking 模式下的 tool_use 经常
# 返回空 input ({} )，且 schema 描述不强制写入。遇到这种情况退回到「请
# 模型直接输出 JSON 文本」，再用容错解析器把它转成 dict。
# ---------------------------------------------------------------------------
_CODE_FENCE_RE = _re.compile(r"^```(?:json)?\s*(.*?)\s*```$", _re.DOTALL | _re.IGNORECASE)


def _coerce_json(text: str) -> dict:
    """容错解析模型文本输出里的 JSON：去 ```json``` 包裹、找最外层 {...}。"""
    text = (text or "").strip()
    m = _CODE_FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return _json.loads(text[start: end + 1])
        except _json.JSONDecodeError:
            pass
    raise ValueError(f"无法解析模型 JSON 输出：{text[:300]}")


def _deep_unstringify(v):
    """递归把"看起来是 JSON 字符串"的字段还原为 dict/list。
    DeepSeek tool_use 有时把嵌套对象序列化成字符串（character_a: '{...}'），
    pydantic validation 会因此挂掉。先归一化再校验。"""
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return _deep_unstringify(_json.loads(s))
            except _json.JSONDecodeError:
                return v
        return v
    if isinstance(v, dict):
        return {k: _deep_unstringify(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_deep_unstringify(x) for x in v]
    return v


def _create_or_json_fallback(client, *, model, system, user, tool, tool_name, schema,
                              max_tokens=4096, temperature=0.7):
    """主路径用 tool_use（Claude 最稳）；fallback 用文本 JSON（DeepSeek 兼容）。

    Why:
      - Claude 系：tool_use 一次到位
      - DeepSeek thinking 模式：会调正确的 tool 但 input={} 或字段被字符串化
        → 检测到空 input / 全部字符串化 都自动改请模型输出 JSON 文本
    """
    is_claude = model.lower().startswith(("claude", "anthropic"))
    base = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "tools": [tool],
        "messages": [{"role": "user", "content": user}],
    }
    if temperature is not None:
        base["temperature"] = temperature

    # 1) tool_use 主路径
    try:
        if is_claude:
            resp = client.messages.create(**base, tool_choice={"type": "tool", "name": tool_name})
        else:
            resp = client.messages.create(**base)
        for blk in resp.content:
            if getattr(blk, "type", None) == "tool_use" and blk.name == tool_name:
                inp = getattr(blk, "input", None) or {}
                if inp:
                    # DeepSeek 经常把嵌套对象序列化成 JSON 字符串 → 先归一化
                    return _deep_unstringify(inp)
                break  # input={} → fallback
    except Exception:
        pass

    # 2) JSON 文本 fallback
    json_user = (
        f"{user}\n\n"
        "请严格按以下 JSON 结构输出，**不要使用 Markdown 代码块包裹**，"
        "**不要输出任何额外解释**，只输出 JSON 对象本身：\n"
        + _json.dumps(schema, ensure_ascii=False, indent=2)
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": json_user}],
        **({"temperature": temperature} if temperature is not None else {}),
    )
    text_parts = [getattr(b, "text", "") for b in resp.content if getattr(b, "text", None)]
    return _deep_unstringify(_coerce_json("".join(text_parts)))

# 角色属性的 JSON Schema，作为客户端工具的入参约束。
# 注：必须**扁平化**，不能用 $ref/$defs —— 第三方 Anthropic 兼容代理
# （DeepSeek / Qwen 等）的 schema 解析器不支持 schema 引用，会让模型
# 拿到空的 input_schema → 工具被调但 input 是 {}.
_SKILL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "damage": {"type": "integer", "minimum": 0, "maximum": 60,
                   "description": "攻击技能的伤害；防御技能必须为 0"},
        "accuracy": {"type": "integer", "minimum": 10, "maximum": 100},
        "kind": {"type": "string", "enum": ["attack", "defense"],
                 "description": "技能类型：attack=攻击，defense=防御格挡"},
        "block": {"type": "integer", "minimum": 0, "maximum": 90,
                  "description": "防御技能格挡下一击的减伤百分比；攻击技能为 0"},
    },
    "required": ["name", "description", "damage", "accuracy", "kind", "block"],
}
_SINGLE_CHARACTER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "title": {"type": "string"},
        "max_hp": {"type": "integer", "minimum": 50, "maximum": 300},
        "attack": {"type": "integer", "minimum": 5, "maximum": 50},
        "defense": {"type": "integer", "minimum": 0, "maximum": 30},
        "skills": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": _SKILL_SCHEMA,
        },
        "flavor": {"type": "string"},
    },
    "required": ["name", "title", "max_hp", "attack", "defense", "skills", "flavor"],
}
_CHARACTER_SCHEMA = {
    "type": "object",
    "properties": {
        "character_a": _SINGLE_CHARACTER_SCHEMA,
        "character_b": _SINGLE_CHARACTER_SCHEMA,
        "intro": {"type": "string", "description": "一句话开场白，介绍这场对战"},
    },
    "required": ["character_a", "character_b", "intro"],
}

_SYSTEM_PROMPT = (
    "你是一个对战游戏的角色设计师。用户会给出正方和反方两个名字"
    "（可能是真实人物、虚构角色、品牌、概念等）。"
    "请根据你对这两个名字的了解（若有可用的 web search 工具，可先联网搜索补充信息），"
    "为双方各设计一个回合制对战角色。\n"
    "【统一语言风格】所有技能命名 + description 都要走「赛博中二 + 网络梗」路线，"
    "奔放、狂野、自信带点贱兮兮，像电竞解说+热血番口播的混合体。\n"
    "要求：\n"
    "1. 技能名要紧扣该名字的真实特点（职业、作品、梗、标志性事物等），起名带气势、"
    "朗朗上口；可大胆糅合「狂飙·XX」「亿点点XX」「XX·原地起飞」「破防XX」"
    "「无双XX」「秒杀XX」这类网感词，但不要烂俗到失去角色味。\n"
    "2. description 写一句生动、奔放、口语化的招式吼话（约 20~45 字），"
    "可以带「直接干」「梆梆」「闪现」「一波带走」「直接破防」「上天」「亿点点伤害」"
    "「打不动？那就再来一发」之类的网络感词，重点是画面感 + 燃。\n"
    "3. 数值与描述自洽：吹得越狠数值就要越离谱（高伤=「一发入魂」/「直接秒」，"
    "高命中=「指哪打哪」/「闪现糊脸」，豪赌技能=「赌就完了」/「不死也得脱层皮」）。\n"
    "4. 每个角色给 3~4 个技能。其中 attack 类技能 2~3 个；"
    "并且【必须恰好有 1 个 defense（防御格挡）技能】，紧扣角色特色"
    "（盾/护体/结界/钢铁之躯/无敌帧），description 也走网感奔放风"
    "（如「全体起立硬刚」「这一下我顶住」「破甲？不存在的」），"
    "damage 必须为 0、kind 填 defense、block 取 40~70（格挡下一击的减伤%）。\n"
    "5. 双方实力大致均衡，让对战有看点；数值遵守 schema 的范围限制。\n"
    "6. 必须调用 submit_characters 工具提交结构化结果，不要输出多余文本。"
)


def _web_search_enabled() -> bool:
    """是否启用 Anthropic 服务端 web_search 工具。

    默认关闭：很多第三方中转代理不支持该工具，开启会导致搜索不生效甚至报错。
    换成 Anthropic 官方接口后，在 .env 设 ENABLE_WEB_SEARCH=true 即可开启真实联网搜索。
    """
    return os.environ.get("ENABLE_WEB_SEARCH", "").strip().lower() in ("1", "true", "yes", "on")


def _build_client():
    """构造 Anthropic 客户端；无 key 时返回 None。

    支持第三方中转代理：在 .env 设置 ANTHROPIC_BASE_URL（如 https://api.openai-next.com）
    即可把请求发到代理端点。否则走 Anthropic 官方端点。
    注意：代理的 key 必须配代理的 base_url，否则官方端点会返回 403 Forbidden。
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    kwargs = {"api_key": api_key}
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


# 防御技能模板：(技能名, 描述, 格挡减伤%范围)。每个角色固定带一个。
# 走「赛博中二+网络梗」统一风格，奔放自信带点贱兮兮，与攻击技能口吻一致。
_DEFENSE_TEMPLATES = [
    ("钢铁壁垒", "全身罡气拉满硬刚到底，下一发管它多猛先给它原地干吐！", (55, 70)),
    ("不动金钟", "护体真气一波灌满金钟罩，下一击？爷今天就是不动如山。", (50, 65)),
    ("玄武护盾", "玄武之灵直接召上头，龟壳一立——破甲？不存在的事儿。", (50, 68)),
    ("圣光结界", "圣光屏障亮瞎全场，下一下的锋芒柔柔地给它卸成毛毛雨。", (45, 60)),
    ("铁布衫", "皮肉绷成钢板梆梆硬，硬抗下一波,挨打就是养生。", (45, 58)),
]


def _make_defense_skill(name: str) -> Skill:
    sk_name, sk_desc, block_range = random.choice(_DEFENSE_TEMPLATES)
    return Skill(
        name=f"{name}·{sk_name}",
        description=sk_desc,
        damage=0,
        accuracy=100,
        kind="defense",
        block=random.randint(*block_range),
    )


def _ensure_defense_skill(char: Character) -> Character:
    """安全网：保证角色恰好携带防御技能。

    Claude 偶尔不遵守"必须带防御技能"的约束，这里做兜底——
    若没有任何防御技能，则补一个主题防御技能（必要时挤掉最弱的攻击技能以不超过 4 个）。
    同时把 damage>0 却被标成 defense 的异常项纠正为 attack，保持数据自洽。
    """
    # 纠正：标成 defense 但有伤害的，按攻击处理
    for sk in char.skills:
        if sk.kind == "defense" and sk.damage > 0:
            sk.kind = "attack"
            sk.block = 0

    has_defense = any(sk.kind == "defense" for sk in char.skills)
    if not has_defense:
        if len(char.skills) >= 4:
            # 挤掉伤害最低的攻击技能，给防御技能腾位置
            weakest_idx = min(
                range(len(char.skills)), key=lambda i: char.skills[i].damage
            )
            char.skills.pop(weakest_idx)
        char.skills.append(_make_defense_skill(char.name))
    return char


def assign_skill_costs(char: Character) -> Character:
    """按伤害排序给技能分配能量消耗：防御 + 最弱攻击 = 0、次强 = 1、最强 = 2。

    规则：
      - 所有 defense 技能 cost = 0
      - 攻击技能按 damage 升序排：
          1 个 → cost=1
          2 个 → 低=0, 高=2
          3+ 个 → 最低=0, 最高=2, 中间全部=1
    任何时候修改了 skills（生成 / 习得新技能 / 强化伤害变化）后都应再跑一次。
    """
    for sk in char.skills:
        if sk.kind == "defense":
            sk.cost = 0
    atks = sorted(
        (sk for sk in char.skills if sk.kind != "defense"),
        key=lambda s: s.damage,
    )
    n = len(atks)
    if n == 0:
        return char
    if n == 1:
        atks[0].cost = 1
    elif n == 2:
        atks[0].cost = 0
        atks[1].cost = 2
    else:
        atks[0].cost = 0
        atks[-1].cost = 2
        for sk in atks[1:-1]:
            sk.cost = 1
    return char


def generate_with_claude(side_a: str, side_b: str) -> Tuple[Character, Character, str]:
    """调用 Claude（联网）生成双方角色。失败时抛异常，由上层决定是否兜底。"""
    client = _build_client()
    if client is None:
        raise RuntimeError("未配置 ANTHROPIC_API_KEY")

    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    submit_tool = {
        "name": "submit_characters",
        "description": "提交为正方和反方设计好的对战角色（结构化）。",
        "input_schema": _CHARACTER_SCHEMA,
    }

    use_web_search = _web_search_enabled()
    if use_web_search:
        # 开启联网搜索时，web_search 是服务端工具，需要 Claude 自主决定先搜后提交，
        # 因此不能强制 tool_choice 到 submit_characters，让模型自由编排两步。
        tools = [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
            submit_tool,
        ]
        user_msg = f"正方名字：{side_a}\n反方名字：{side_b}\n请先联网搜索补充信息，再设计角色并调用 submit_characters 提交。"

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": user_msg}],
        )
        payload = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "submit_characters":
                payload = block.input or {}
                break
        if not payload or "character_a" not in payload or "character_b" not in payload:
            raise RuntimeError("Claude(web_search) 未返回有效的 submit_characters 数据")
    else:
        # 不联网：走统一的 tool_use + JSON 文本兜底辅助函数。
        # 对 Claude 直接 tool_use 一次到位；对 DeepSeek/Qwen 等推理模型，
        # tool_use 失败/空 input 时自动转入"输出 JSON 文本"路径。
        user_msg = f"正方名字：{side_a}\n反方名字：{side_b}\n请根据你的了解为正反双方各设计一名对战角色。"
        payload = _create_or_json_fallback(
            client,
            model=model,
            system=_SYSTEM_PROMPT,
            user=user_msg,
            tool=submit_tool,
            tool_name="submit_characters",
            schema=_CHARACTER_SCHEMA,
            max_tokens=4096,
            temperature=None,
        )
        if "character_a" not in payload or "character_b" not in payload:
            raise RuntimeError(
                f"角色数据缺少 character_a/character_b。实际 keys={list(payload.keys())}"
            )

    char_a = Character.model_validate(payload["character_a"])
    char_b = Character.model_validate(payload["character_b"])
    _ensure_defense_skill(char_a)
    _ensure_defense_skill(char_b)
    assign_skill_costs(char_a)
    assign_skill_costs(char_b)
    intro = payload.get("intro", "")
    return char_a, char_b, intro


# ---------------------------------------------------------------------------
# mock 兜底：无 key 或调用失败时，用随机属性生成角色，保证 demo 可玩
# ---------------------------------------------------------------------------

# 每个攻击模板：(技能名, 描述, 伤害范围, 命中范围)
# 数值与描述自洽，统一走「赛博中二+网络梗」奔放风（连击=高命中中低伤、爆发=高伤等）。
_MOCK_SKILL_TEMPLATES = [
    ("闪现连鞭", "残影都跟不上节奏，糊脸三连梆梆甩鞭，根本不给反应时间。", (16, 26), (92, 100)),
    ("亿点点重锤", "全身气血灌满槽位，憋红了脸轰出一锤——这下不让你脱层皮算输。", (44, 58), (62, 78)),
    ("绝命背刺", "猫腰摸到死角直接来一记暗杀，越是关键时刻越懂得给你致命一击。", (40, 54), (66, 82)),
    ("地裂震天波", "脚下一跺直接干裂地皮，冲击波横扫八方，方圆十米全员原地起飞。", (30, 42), (80, 92)),
    ("绝境一波带走", "血越少头越铁，背水一战之下直接进入觉醒状态打你个措手不及。", (34, 48), (70, 86)),
    ("锁头瞄准", "屏息直接锁死你脑门，指哪打哪百发百中，输出永不落空也是种压迫感。", (24, 34), (96, 100)),
    ("怒涛连斩", "怒气槽爆满模式开启，唰唰唰一串斩击直接铺脸，根本停不下来。", (18, 28), (90, 99)),
    ("破甲狂袭", "怒气贯顶撕烂对面防线，无视部分防御，铜墙铁壁照样给你掰开。", (38, 50), (74, 88)),
]


def _mock_character(name: str) -> Character:
    # 攻击技能：取 2~3 个模板
    skills = []
    for sk_name, sk_desc, dmg_range, acc_range in random.sample(
        _MOCK_SKILL_TEMPLATES, k=random.randint(2, 3)
    ):
        skills.append(
            Skill(
                name=f"{name}·{sk_name}",
                description=sk_desc,
                damage=random.randint(*dmg_range),
                accuracy=random.randint(*acc_range),
            )
        )
    # 固定追加一个主题防御技能（复用全局防御模板）
    skills.append(_make_defense_skill(name))
    char = Character(
        name=name,
        title="神秘挑战者",
        max_hp=random.randint(120, 220),
        attack=random.randint(12, 28),
        defense=random.randint(0, 15),
        skills=skills,
        flavor=f"关于 {name} 的传说众说纷纭，但此刻它已踏入战场。",
    )
    assign_skill_costs(char)
    return char


def generate_mock(side_a: str, side_b: str) -> Tuple[Character, Character, str]:
    """本地随机生成双方角色（无需联网/密钥）。"""
    return (
        _mock_character(side_a),
        _mock_character(side_b),
        f"【模拟模式】{side_a} 与 {side_b} 的宿命对决一触即发！",
    )


# ---------------------------------------------------------------------------
# 试炼之塔：生成「英雄」（单个角色，主题贴合玩家名字）
# ---------------------------------------------------------------------------

_HERO_SCHEMA = {
    "type": "object",
    "properties": {
        "hero": _SINGLE_CHARACTER_SCHEMA,
        "intro": {"type": "string", "description": "一句话英雄登场介绍"},
    },
    "required": ["hero", "intro"],
}

_HERO_SYSTEM_PROMPT = (
    "你是一个 roguelike 闯关游戏的英雄设计师。玩家会给出一个名字"
    "（可能是真实人物、虚构角色、品牌、概念等），请把它设计成一名即将挑战"
    "「试炼之塔」的勇者角色。\n"
    "【统一语言风格】所有技能命名 + description 都要走「赛博中二 + 网络梗」路线，"
    "奔放、狂野、自信带点贱兮兮，像电竞解说+热血番口播的混合体。\n"
    "要求：\n"
    "1. 技能名要紧扣该名字的真实特点，起名带气势、朗朗上口；可大胆糅合"
    "「狂飙·XX」「亿点点XX」「XX·原地起飞」「破防XX」「秒杀XX」「无双XX」"
    "等网感词，但保留角色味儿。\n"
    "2. description 写一句生动、奔放、口语化的招式吼话（约 20~45 字），"
    "可带「直接干」「梆梆」「闪现」「一波带走」「破防」「上天」「亿点点伤害」"
    "之类的网络感词，重点是画面感 + 燃。\n"
    "3. 数值与描述自洽：吹得越狠数值越离谱（高伤=「一发入魂」「直接秒」，"
    "高命中=「锁头糊脸」「指哪打哪」，豪赌=「赌就完了」「不死也得脱层皮」）。\n"
    "4. 给英雄恰好 4 个技能：其中 3 个 attack 类（kind=attack）；"
    "并且【必须恰好有 1 个 defense 防御格挡技能】，紧扣角色特色（盾/护体/结界/铁壁/无敌帧），"
    "description 同走网感奔放风（如「这一下我顶住」「破甲？不存在的」），"
    "damage 必须为 0、kind 填 defense、block 取 45~70。\n"
    "5. 作为闯关起点，数值适中即可（后续会通过强化变强）："
    "建议 max_hp 130~170、attack 14~22、defense 2~10。\n"
    "6. 数值遵守 schema 范围。必须调用 submit_hero 工具提交结构化结果，不要输出多余文本。"
)


def generate_hero_with_claude(name: str) -> Tuple[Character, str]:
    """调用 Claude 生成主题贴合的英雄。失败时抛异常，由上层兜底。"""
    client = _build_client()
    if client is None:
        raise RuntimeError("未配置 ANTHROPIC_API_KEY")

    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    submit_tool = {
        "name": "submit_hero",
        "description": "提交为玩家设计好的英雄角色（结构化）。",
        "input_schema": _HERO_SCHEMA,
    }

    use_web_search = _web_search_enabled()
    if use_web_search:
        tools = [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
            submit_tool,
        ]
        user_msg = (
            f"英雄名字：{name}\n请先联网搜索补充信息，再设计这名挑战试炼之塔的勇者，"
            f"并调用 submit_hero 提交。"
        )
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=_HERO_SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": user_msg}],
        )
        payload = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "submit_hero":
                payload = block.input or {}
                break
        if not payload or "hero" not in payload:
            raise RuntimeError("Claude(web_search) 未返回有效 submit_hero 数据")
    else:
        # 同混沌出击：用 tool_use + JSON 文本兜底，兼容 DeepSeek 推理模型
        user_msg = f"英雄名字：{name}\n请为这名挑战试炼之塔的勇者设计角色资料。"
        payload = _create_or_json_fallback(
            client,
            model=model,
            system=_HERO_SYSTEM_PROMPT,
            user=user_msg,
            tool=submit_tool,
            tool_name="submit_hero",
            schema=_HERO_SCHEMA,
            max_tokens=2048,
            temperature=None,
        )
        if "hero" not in payload:
            raise RuntimeError(f"英雄数据缺少 hero 字段。实际 keys={list(payload.keys())}")

    hero = Character.model_validate(payload["hero"])
    _ensure_defense_skill(hero)
    assign_skill_costs(hero)
    intro = payload.get("intro", "")
    return hero, intro


def generate_hero_mock(name: str) -> Tuple[Character, str]:
    """本地随机生成英雄（无需联网/密钥）。数值偏起点，便于后续强化成长。"""
    hero = _mock_character(name)
    hero.title = "试炼挑战者"
    # 作为闯关起点，控制初始数值在适中区间
    hero.max_hp = random.randint(130, 170)
    hero.attack = random.randint(14, 22)
    hero.defense = random.randint(2, 10)
    hero.flavor = f"{name} 踏入试炼之塔，誓要登顶证道。"
    assign_skill_costs(hero)
    return hero, f"【模拟模式】勇者 {name} 踏入了试炼之塔的大门……"
