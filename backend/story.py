"""故事对决模式：移植自 compete! 项目。

与现有的「名字对战」（自动战斗）和「试炼之塔」（手动选招）不同，
故事对决主打**叙事性**：每一回合由 Claude 写一段 8-12 句的精彩解说，
打字机展示。三局两胜决出冠军。

接口：
  - generate_characters(name_a, name_b) → (Character, Character)
  - play_round(pro, ant, round_no, score, history) → (result, score, over, winner)

降级：无 ANTHROPIC_API_KEY 时返回 mock，与项目其它模块一致。

【为什么用 tool use 而不是直接解析文本 JSON】
Claude 偶尔会在中文 description 里夹带未转义的双引号
（例如 `座右铭是"只要还有万分之一的可能..."`），导致 json.loads 失败。
让 Claude 通过 tool 调用提交结构化参数，SDK 自动处理 JSON 编码，零解析失败。
"""
from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import anthropic


# ---------------------------------------------------------------------------
# 数据契约 + 常量
# ---------------------------------------------------------------------------

WINS_NEEDED = 2  # 三局两胜


# ---------------------------------------------------------------------------
# Claude 客户端
# ---------------------------------------------------------------------------

_client_cache: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client_cache
    if _client_cache is None:
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
        kwargs: dict[str, Any] = {"api_key": os.environ["ANTHROPIC_API_KEY"]}
        if base_url:
            kwargs["base_url"] = base_url
        _client_cache = anthropic.Anthropic(**kwargs)
    return _client_cache


def _model_name() -> str:
    return os.environ.get("CLAUDE_MODEL") or "claude-sonnet-4-20250514"


def _extract_tool_input(resp: Any, tool_name: str) -> dict[str, Any]:
    """从模型响应里取 tool_use block 的 input（已是结构化 dict）。"""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == tool_name:
            data = getattr(block, "input", None)
            if isinstance(data, dict):
                return data
    raise ValueError(f"模型未调用工具 {tool_name}")


def _model_supports_forced_tool() -> bool:
    """有些推理模型（DeepSeek v4-pro 等）在 thinking 模式下不接受
    `tool_choice={type:'tool', name:'...'}` 这种硬指定，会 400。
    这里按模型名前缀简单判定：deepseek / qwen / glm 等先关掉强制。
    Anthropic / Claude 自家模型保留强制（最稳）。
    """
    name = _model_name().lower()
    if name.startswith(("claude", "anthropic")):
        return True
    return False


def _create_with_tool(*, model: str, system: str, user: str,
                      tool: dict, tool_name: str,
                      max_tokens: int = 2500, temperature: float = 0.7) -> dict[str, Any]:
    """统一的 tool-use 调用 + 失败回退。

    步骤：
      1) 若模型支持强制 tool_choice，先尝试强制选用 tool_name
      2) 拿 400 的话自动重试一次，去掉 tool_choice 让模型自决
         （system prompt 已经强烈要求它必须调这个工具）
    """
    client = _get_client()
    base_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "tools": [tool],
        "messages": [{"role": "user", "content": user}],
    }

    if _model_supports_forced_tool():
        try:
            resp = client.messages.create(
                **base_kwargs,
                tool_choice={"type": "tool", "name": tool_name},
            )
            return _extract_tool_input(resp, tool_name)
        except Exception:
            pass  # 跌穿到下面的无 tool_choice 重试

    # 不强制 tool_choice：模型一般也会调用提供的唯一工具
    resp = client.messages.create(**base_kwargs)
    return _extract_tool_input(resp, tool_name)


# ---------------------------------------------------------------------------
# 角色生成（tool use 版本，避免内嵌引号导致的 JSON 解析失败）
# ---------------------------------------------------------------------------

_CHARACTER_SYSTEM_PROMPT = (
    "你是一位精通各类动漫、游戏、轻小说等虚拟作品的资深考据专家，"
    "同时也是一位脑洞大开、富有设计感的「技能设计师」。\n\n"
    "请按用户指定的角色名，调用提供的 submit_character 工具提交结构化资料。\n\n"
    "要求：\n"
    "1. 必须准确识别角色来自哪部作品；同名优先选取最知名版本。\n"
    "2. 概况 1~3 句话即可，点明出处、性格、简要经历。\n"
    "3. 技能 3~4 个，贴合角色原作设定，名字响亮、描述生动（每个 1~2 句）。\n"
    "4. 完全不认识时也要发挥创意基于名字脑补，并在 source 字段注明「（推测/原创角色）」。\n"
    "5. **不要在 summary / description 里使用任何引号字符**——需要强调时改用「」或省略，"
    "避免 JSON 转义出错。"
)

_CHARACTER_TOOL = {
    "name": "submit_character",
    "description": "提交一位角色的完整资料（包含名字、来源、概况、3~4 个技能）",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "角色规范名称"},
            "source": {"type": "string", "description": "角色来源作品，如《宝可梦》"},
            "summary": {"type": "string", "description": "1~3 句话概况；勿用引号字符"},
            "skills": {
                "type": "array",
                "minItems": 3,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "技能名"},
                        "description": {"type": "string", "description": "技能描述 1~2 句；勿用引号字符"},
                    },
                    "required": ["name", "description"],
                },
            },
        },
        "required": ["name", "source", "summary", "skills"],
    },
}


def _sanitize_character(raw: dict[str, Any], fallback_name: str) -> dict[str, Any]:
    """虽然 tool 已校验，仍做一层防御性清洗，统一字段类型。"""
    if not isinstance(raw, dict):
        return {"name": fallback_name, "source": "", "summary": "", "skills": []}
    name = str(raw.get("name") or fallback_name).strip()
    source = str(raw.get("source") or "").strip()
    summary = str(raw.get("summary") or "").strip()
    skills: list[dict[str, str]] = []
    for it in raw.get("skills", []) or []:
        if not isinstance(it, dict):
            continue
        sn = str(it.get("name") or "").strip()
        if not sn:
            continue
        skills.append({"name": sn, "description": str(it.get("description") or "").strip()})
    return {"name": name, "source": source, "summary": summary, "skills": skills}


def _gen_one_character(name: str, side_label: str) -> dict[str, Any]:
    user = (
        f"请为这位【{side_label}】角色生成资料，并通过 submit_character 工具提交。\n"
        f"角色名字：{name}"
    )
    raw = _create_with_tool(
        model=_model_name(),
        system=_CHARACTER_SYSTEM_PROMPT,
        user=user,
        tool=_CHARACTER_TOOL,
        tool_name="submit_character",
        temperature=0.6,
    )
    return _sanitize_character(raw, fallback_name=name)


def generate_characters_with_claude(
    name_a: str, name_b: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """并发为正反双方生成故事模式角色。"""
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(_gen_one_character, name_a, "正方")
        fb = pool.submit(_gen_one_character, name_b, "反方")
        return fa.result(), fb.result()


# ---------------------------------------------------------------------------
# Mock 兜底
# ---------------------------------------------------------------------------

_MOCK_SKILL_TEMPLATES = [
    "{name}招牌爆发技：一击爆破。",
    "{name}招牌位移技：闪现冲刺。",
    "{name}招牌奶妈技：自我修复。",
    "{name}招牌大招：场上唯一神。",
]


def _mock_character(name: str) -> dict[str, Any]:
    rng = random.Random(name)
    return {
        "name": name,
        "source": "（mock 模式）",
        "summary": f"{name} 是一位身手敏捷、招式繁多的传奇角色，传说战无不胜。",
        "skills": [
            {
                "name": f"{name}·{['流影斩','破天击','千幻舞','极光闪'][i % 4]}",
                "description": rng.choice(_MOCK_SKILL_TEMPLATES).format(name=name),
            }
            for i in range(3)
        ],
    }


def generate_characters_mock(name_a: str, name_b: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return _mock_character(name_a), _mock_character(name_b)


# ---------------------------------------------------------------------------
# 回合叙事生成（tool use 版本）
# ---------------------------------------------------------------------------

_BATTLE_SYSTEM_PROMPT = (
    "你是一位风格独特、肾上腺素永远拉满的「热血对战解说员」，"
    "负责实况解说两位虚拟角色之间的回合制对决。请通过 submit_round 工具提交本回合结果。\n\n"
    "【解说风格】\n"
    "- 电波系、略带狗血、生草有梗，让人会心一笑，「有趣」始终是第一要务。\n"
    "- 充分调用双方角色的原作设定、性格、台词梗、技能编排戏剧性的攻防与吐槽。\n"
    "- 中二台词、夸张演出、热血嘶吼都欢迎，但不要完全脱线。\n\n"
    "【激烈度】\n"
    "- 至少 3~5 个攻防来回；必须有节奏起伏与至少一次形势逆转。\n"
    "- 要写出具体的技能碰撞，画面感要强。\n"
    "- 收尾干脆有力，点明胜负手。\n\n"
    "【字符约束】**不要在 narration / highlight 里使用引号字符**——"
    "需要强调时改用「」或省略，避免 JSON 转义问题。"
)

_BATTLE_TOOL = {
    "name": "submit_round",
    "description": "提交本回合对战的解说与胜负",
    "input_schema": {
        "type": "object",
        "properties": {
            "narration": {
                "type": "string",
                "description": "本回合精彩解说正文，8~12 句，画面感强，勿用引号字符",
            },
            "winner": {
                "type": "string",
                "enum": ["protagonist", "antagonist"],
                "description": "本回合胜者：protagonist 或 antagonist",
            },
            "winner_name": {"type": "string", "description": "胜者角色名字"},
            "highlight": {
                "type": "string",
                "description": "本回合高光看点（一句话），勿用引号字符",
            },
        },
        "required": ["narration", "winner", "winner_name", "highlight"],
    },
}


def _format_char_for_battle(c: dict[str, Any]) -> str:
    skills_text = "\n".join(
        f"    - {s.get('name','未命名')}：{s.get('description','')}"
        for s in c.get("skills", [])
    )
    return (
        f"  名字：{c.get('name','')}\n"
        f"  出处：{c.get('source','')}\n"
        f"  概况：{c.get('summary','')}\n"
        f"  技能：\n{skills_text}"
    )


def _build_battle_user(
    pro: dict[str, Any], ant: dict[str, Any], round_no: int,
    score: dict[str, int], history: list[str],
) -> str:
    history_text = (
        "\n".join(f"  第{i + 1}回合：{h}" for i, h in enumerate(history))
        if history else "  （这是第一回合，暂无前情）"
    )
    return (
        f"【对战双方】\n"
        f"正方（protagonist）：\n{_format_char_for_battle(pro)}\n\n"
        f"反方（antagonist）：\n{_format_char_for_battle(ant)}\n\n"
        f"【赛制】三局两胜，当前为第 {round_no} 回合。\n"
        f"【当前比分】正方 {score.get('protagonist',0)} : {score.get('antagonist',0)} 反方\n\n"
        f"【前情回顾】\n{history_text}\n\n"
        f"请解说本回合（第 {round_no} 回合）的对决，要求 3~5 个攻防来回、"
        f"至少一次形势逆转、收尾干脆点明胜负手，然后调用 submit_round 工具提交。"
    )


_VALID_WINNERS = ("protagonist", "antagonist")


def _sanitize_round(
    raw: dict[str, Any], pro: dict[str, Any], ant: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    narration = str(raw.get("narration") or "（本回合解说生成异常，双方僵持不下……）").strip()
    winner = str(raw.get("winner") or "").strip().lower()
    if winner not in _VALID_WINNERS:
        raw_name = str(raw.get("winner_name") or "").strip()
        if raw_name and raw_name == ant.get("name"):
            winner = "antagonist"
        else:
            winner = "protagonist"
    winner_name = pro.get("name", "") if winner == "protagonist" else ant.get("name", "")
    return {
        "narration": narration,
        "winner": winner,
        "winner_name": winner_name,
        "highlight": str(raw.get("highlight") or "").strip(),
    }


def play_round_with_claude(
    pro: dict[str, Any], ant: dict[str, Any], round_no: int,
    score: dict[str, int], history: list[str],
) -> dict[str, Any]:
    user = _build_battle_user(pro, ant, round_no, score, history)
    raw = _create_with_tool(
        model=_model_name(),
        system=_BATTLE_SYSTEM_PROMPT,
        user=user,
        tool=_BATTLE_TOOL,
        tool_name="submit_round",
        temperature=0.95,
    )
    return _sanitize_round(raw, pro, ant)


def play_round_mock(
    pro: dict[str, Any], ant: dict[str, Any], round_no: int,
    score: dict[str, int], history: list[str],
) -> dict[str, Any]:
    rng = random.Random(f"{pro.get('name','')}{ant.get('name','')}{round_no}")
    winner = rng.choice(_VALID_WINNERS)
    winner_name = pro.get("name", "") if winner == "protagonist" else ant.get("name", "")
    loser_name = ant.get("name", "") if winner == "protagonist" else pro.get("name", "")
    return {
        "narration": (
            f"第 {round_no} 回合开战！{pro.get('name','')} 一招破空冲向 {ant.get('name','')}，"
            f"对方架起防御接下重击，火星四溅。两人你来我往三个回合，"
            f"招式碰撞震得场地裂开蛛网纹路。就在 {loser_name} 喘息间隙，"
            f"{winner_name} 抓住破绽，使出全力一击，干净利落地拿下本回合！"
            f"（mock 模式占位叙事，配置 ANTHROPIC_API_KEY 后即可看到 Claude 写的精彩解说）"
        ),
        "winner": winner,
        "winner_name": winner_name,
        "highlight": f"{winner_name} 抓住破绽完成关键一击！",
    }


# ---------------------------------------------------------------------------
# 比分推进 + 胜负判定（权威在后端）
# ---------------------------------------------------------------------------

def advance_score(score: dict[str, int], winner: str) -> tuple[dict[str, int], bool, str]:
    new_score = {
        "protagonist": int(score.get("protagonist", 0)) + (1 if winner == "protagonist" else 0),
        "antagonist": int(score.get("antagonist", 0)) + (1 if winner == "antagonist" else 0),
    }
    champion = ""
    if new_score["protagonist"] >= WINS_NEEDED:
        champion = "protagonist"
    elif new_score["antagonist"] >= WINS_NEEDED:
        champion = "antagonist"
    return new_score, champion != "", champion
