"""试炼之塔（打怪升级模式）核心逻辑。

职责：
- 按玩家输入的「挑战场地」用 Claude 生成三个随层数递增、各带防御技能的主题敌人；
  联网失败时降级到本地程序化生成（把场地名融入敌人名称/风味），保证始终可玩。
- 维护「三选一强化」的强化池，并把选中的强化应用到英雄身上。
- 应用强化时统一裁剪到 Character / Skill 的合法数值范围，保证英雄属性始终自洽。

设计取舍：三个敌人在开局时一次性生成（一次 Claude 调用产出三层），
避免逐层调用带来的多次延迟；强度随层数曲线递增，让「升级 → 变强 →
挑战更强敌人」的成长闭环稳定成立。
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import character_gen
from models import TOWER_FLOORS, Character, Skill, Upgrade

# ---------------------------------------------------------------------------
# 敌人本地模板：按层数递增强度（作为 mock 兜底与 Claude 失败时的后备）
# ---------------------------------------------------------------------------

# 每层一个 Boss 档案：名字、称号、风味、属性区间（随层数走强度曲线）。
# 数值区间设计：第 1 层略弱于英雄起点（可打），第 2 层势均力敌，第 3 层为压轴强敌。
# 技能描述统一走「赛博中二+网络梗」奔放风，与英雄技能口吻一致。
_BOSS_TIERS = [
    {
        "candidates": [
            ("噬铁巨鼠", "下水道霸主", "在试炼之塔的阴影里磨利了獠牙，专等愣头青送上门。"),
            ("石化守卫", "残破的傀儡", "曾经的守门石像被试炼之力唤醒，立在阶梯口梆梆挡道。"),
            ("沼泽魔藤", "缠绕的恶意", "盘踞第一层的食人魔藤，藤蔓里全是失败者的兵器残片。"),
        ],
        "hp": (85, 110),
        "attack": (10, 14),
        "defense": (1, 5),
        "skills": [
            ("利爪连抓", "低吼一声直接扑脸，爪子梆梆梆三连快得让你根本反应不过来。", (12, 18), (90, 99)),
            ("猛力撞击", "退两步加速冲撞，整个身子怼上来一发——这一下不亏血算我输。", (22, 30), (72, 85)),
            ("腐蚀吐息", "呸地喷一口酸臭气，黏糊糊糊脸上灼一波，恶心又破防。", (17, 23), (82, 92)),
        ],
        "block": (40, 55),
    },
    {
        "candidates": [
            ("炽焰魔将", "烈火的化身", "中层看守浑身燃业火，路过半步就给你烤酥。"),
            ("寒冰女妖", "霜之哀歌", "歌声直接冻血液，整层楼被她唱成了冷藏柜。"),
            ("雷暴泰坦", "怒雷的囚徒", "被铁链锁住的雷霆巨人，挣一下天上就劈一波。"),
        ],
        "hp": (165, 200),
        "attack": (20, 26),
        "defense": (7, 13),
        "skills": [
            ("元素爆轰", "掌心攒一颗失控元素核，啪地一炸——这风暴就是来收人头的。", (38, 48), (68, 80)),
            ("狂澜连斩", "化作流光梆梆梆瞬移三连斩，残影都还在原地你已经空血。", (18, 25), (92, 99)),
            ("湮灭领域", "展开吞噬万物的领域，无视部分防御直接给你来个原地起飞。", (32, 42), (76, 88)),
            ("古龙之怒", "塔层都被吼得晃，一口气倾泻出毁天灭地的一发——挨上算我输。", (44, 54), (60, 74)),
        ],
        "block": (48, 64),
    },
    {
        "candidates": [
            ("塔主·虚空君王", "试炼的终焉", "试炼之塔的话事人，掌虚空和轮回，从没见过有人在它面前登顶。"),
            ("混沌深渊领主", "万千败者之冢", "由历代挑战者的执念凝出的怪物，是这塔最绝望的压轴关。"),
            ("永夜之龙", "吞日的灾厄", "盘踞塔顶的古龙，双翼一展即永夜降临，传说它从没输过。"),
        ],
        "hp": (255, 310),
        "attack": (30, 40),
        "defense": (14, 22),
        "skills": [
            ("终焉审判", "举起象征终结的力量，一记审判劈下来，谁顶谁原地起飞。", (50, 60), (66, 78)),
            ("虚空裂隙", "撕开一道吞光的裂缝，把你卷进无尽虚无里梆梆梆绞个干净。", (40, 50), (80, 90)),
            ("不灭轮回", "被打惨了反而开觉醒，从轮回里捞一把力量反手一记更狠的。", (34, 44), (84, 94)),
            ("万象归寂", "倾尽全身威能一记终极秒杀，万物在它面前直接归零。", (54, 60), (58, 70)),
        ],
        "block": (55, 72),
    },
]

# 敌人防御技能模板：(技能名, 描述)，奔放统一风
_ENEMY_DEFENSE_TEMPLATES = [
    ("硬甲格挡", "缩起身一身硬甲梆梆绷紧，下一发?——爷今天就是不挨打。"),
    ("元素护罩", "周身骤升一圈元素罩，下一波糊脸的全给它弹回去。"),
    ("虚影闪护", "身形虚化飘忽不定，下一击的锋芒咻地卸成毛毛雨。"),
]


def _scaled_int(low: int, high: int, rng: random.Random) -> int:
    return rng.randint(low, high)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def generate_enemy(level: int, arena: str = "", rng: Optional[random.Random] = None) -> Character:
    """为指定层数本地生成一个主题化 Boss（level 取值 1~TOWER_FLOORS）。

    强度随层数递增；同一层内从候选档案中随机挑一个并做小幅浮动。
    固定带一个随层数增强的防御（格挡）技能。arena 非空时融入风味描述。
    """
    rng = rng or random.Random()
    idx = max(0, min(level - 1, len(_BOSS_TIERS) - 1))
    tier = _BOSS_TIERS[idx]

    name, title, flavor = rng.choice(tier["candidates"])
    if arena:
        flavor = f"【{arena}】{flavor}"

    # 攻击技能：取该层技能池随机 2~3 个
    pool = tier["skills"]
    k = min(len(pool), rng.randint(2, 3))
    chosen = rng.sample(pool, k=k)
    skills: List[Skill] = []
    for sk_name, sk_desc, dmg_range, acc_range in chosen:
        dmg = _clamp(_scaled_int(*dmg_range, rng), 1, 60)
        acc = _clamp(_scaled_int(*acc_range, rng), 10, 100)
        skills.append(Skill(name=sk_name, description=sk_desc, damage=dmg, accuracy=acc, kind="attack"))

    # 固定追加一个防御技能
    d_name, d_desc = rng.choice(_ENEMY_DEFENSE_TEMPLATES)
    skills.append(
        Skill(
            name=d_name,
            description=d_desc,
            damage=0,
            accuracy=100,
            kind="defense",
            block=_clamp(_scaled_int(*tier["block"], rng), 0, 90),
        )
    )

    return Character(
        name=name,
        title=title,
        max_hp=_clamp(_scaled_int(*tier["hp"], rng), 50, 300),
        attack=_clamp(_scaled_int(*tier["attack"], rng), 5, 50),
        defense=_clamp(_scaled_int(*tier["defense"], rng), 0, 30),
        skills=skills,
        flavor=flavor,
    )


def generate_enemies_mock(arena: str = "", rng: Optional[random.Random] = None) -> List[Character]:
    """本地生成三层敌人（无需联网/密钥）。"""
    rng = rng or random.Random()
    enemies = [generate_enemy(level, arena=arena, rng=rng) for level in range(1, TOWER_FLOORS + 1)]
    for e in enemies:
        character_gen.assign_skill_costs(e)
    return enemies


# ---------------------------------------------------------------------------
# Claude 按「挑战场地」生成三层敌人
# ---------------------------------------------------------------------------

_ENEMIES_SCHEMA = {
    "type": "object",
    "properties": {
        "enemies": {
            "type": "array",
            "minItems": TOWER_FLOORS,
            "maxItems": TOWER_FLOORS,
            "items": character_gen._SINGLE_CHARACTER_SCHEMA,
        },
        "intro": {"type": "string", "description": "一句话试炼登场介绍，点题挑战场地"},
    },
    "required": ["enemies", "intro"],
}

_ENEMIES_SYSTEM_PROMPT = (
    "你是一个 roguelike 闯关游戏的敌人设计师。玩家会给出一个「挑战场地」"
    "（如：熔岩深渊、霓虹赛博都市、武侠江湖、克苏鲁海沟、童话森林……）。"
    "请围绕这个场地的主题，设计三个登场顺序由弱到强的 Boss 敌人，对应试炼之塔的三层。\n"
    "【统一语言风格】所有技能命名 + description 都要走「赛博中二 + 网络梗」路线，"
    "奔放、狂野、自信带点贱兮兮，像电竞解说+热血番口播的混合体（与英雄技能口吻一致）。\n"
    "要求：\n"
    "1. 三个敌人的名字、称号、技能、风味都紧扣场地主题，画面统一、有代入感；"
    "技能名带气势，可用「狂飙·XX」「亿点点XX」「破防XX」「XX·原地起飞」等网感词。\n"
    "2. 强度随层数明显递增，建议数值区间：\n"
    "   - 第1层(热身)：max_hp 85~115、attack 10~14、defense 1~5\n"
    "   - 第2层(强敌)：max_hp 165~200、attack 20~26、defense 7~13\n"
    "   - 第3层(压轴)：max_hp 255~310、attack 30~40、defense 14~22\n"
    "3. 每个敌人给 3~4 个技能：其中 attack 类 2~3 个；"
    "并且【每个敌人必须恰好有 1 个 defense 防御格挡技能】，紧扣主题，"
    "damage 必须为 0、kind 填 defense、block 取 45~70（越高层越高），"
    "description 同走网感奔放风（如「这一发我顶住」「破甲？不存在的」）。\n"
    "4. description 是一句生动奔放的招式吼话（约 20~45 字），数值与描述自洽。\n"
    "5. 数值遵守 schema 范围。必须调用 submit_enemies 工具提交结构化结果，不要输出多余文本。"
)


def generate_enemies_with_claude(arena: str) -> Tuple[List[Character], str]:
    """调用 Claude 按场地生成三层敌人。失败时抛异常，由上层兜底。"""
    client = character_gen._build_client()
    if client is None:
        raise RuntimeError("未配置 ANTHROPIC_API_KEY")

    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    submit_tool = {
        "name": "submit_enemies",
        "description": "提交围绕挑战场地设计好的三层敌人（结构化）。",
        "input_schema": _ENEMIES_SCHEMA,
    }

    if character_gen._web_search_enabled():
        tools = [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
            submit_tool,
        ]
        user_msg = (
            f"挑战场地：{arena}\n请围绕该场地先联网搜索补充灵感，再设计三层由弱到强的敌人，"
            f"调用 submit_enemies 提交。"
        )
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_ENEMIES_SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": user_msg}],
        )
        payload = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "submit_enemies":
                payload = block.input or {}
                break
        if not payload or "enemies" not in payload:
            raise RuntimeError("Claude(web_search) 未返回有效 submit_enemies 数据")
    else:
        # 同 character_gen：用 tool_use + JSON 文本兜底，兼容 DeepSeek
        user_msg = f"挑战场地：{arena}\n请围绕该场地设计三层由弱到强的敌人。"
        payload = character_gen._create_or_json_fallback(
            client,
            model=model,
            system=_ENEMIES_SYSTEM_PROMPT,
            user=user_msg,
            tool=submit_tool,
            tool_name="submit_enemies",
            schema=_ENEMIES_SCHEMA,
            max_tokens=4096,
            temperature=None,
        )
        if "enemies" not in payload:
            raise RuntimeError(f"敌人数据缺少 enemies 字段。实际 keys={list(payload.keys())}")

    enemies_raw = payload["enemies"]
    if len(enemies_raw) != TOWER_FLOORS:
        raise RuntimeError(f"Claude 返回的敌人数量不是 {TOWER_FLOORS}")

    enemies: List[Character] = []
    for raw in enemies_raw:
        enemy = Character.model_validate(raw)
        character_gen._ensure_defense_skill(enemy)  # 安全网：保证带防御技能
        character_gen.assign_skill_costs(enemy)
        enemies.append(enemy)
    intro = payload.get("intro", "")
    return enemies, intro


def generate_enemies(arena: str) -> Tuple[List[Character], str, str]:
    """生成三层敌人。优先 Claude（按场地主题化），失败降级 mock。

    返回 (enemies, intro, source)，source 为 'claude' 或 'mock'。
    """
    arena = (arena or "").strip()
    if os.environ.get("ANTHROPIC_API_KEY") and arena:
        try:
            enemies, intro = generate_enemies_with_claude(arena)
            return enemies, intro, "claude"
        except Exception:  # noqa: BLE001 — 任何失败都兜底
            pass
    enemies = generate_enemies_mock(arena)
    intro = (
        f"挑战场地「{arena}」的三重试炼已就位，强敌依次登场！"
        if arena
        else "试炼之塔的三重考验已就位，强敌依次登场！"
    )
    return enemies, intro, "mock"


# ---------------------------------------------------------------------------
# 三选一强化：强化池 + 应用
# ---------------------------------------------------------------------------

# 可习得的「新技能」池（用于 epic 级强化）。含一个强力防御技能。
# 风格与英雄/敌人统一：网感奔放、自信带点贱兮兮。
_GRANTABLE_SKILLS = [
    Skill(name="圣裁斩", description="圣光直接凝成巨刃当头一劈，正义糊脸——挨这一发等于挨锤。", damage=46, accuracy=80, kind="attack"),
    Skill(name="影袭", description="遁进暗影闪现到你死角，啪一下背刺秒收，快到看不见残影。", damage=24, accuracy=98, kind="attack"),
    Skill(name="炎爆术", description="掌心炎核哐当一爆，烈焰直接吞了你——威力大归大，赌就完了。", damage=54, accuracy=66, kind="attack"),
    Skill(name="龙鳞铁壁", description="龙鳞铁壁直接糊一身，下一发的锋芒咻地卸干净——这一下我顶住。", damage=0, accuracy=100, kind="defense", block=75),
]


def _upgrade_pool() -> List[Upgrade]:
    """构造完整的强化候选池。每次发牌时从中随机抽取互不相同的 3 个。"""
    pool: List[Upgrade] = [
        Upgrade(
            id="vigor", name="生命涌泉", description="最大生命 +65，血条直接给你拉长，肉到没朋友。",
            icon="❤️", rarity="common", add_max_hp=65,
        ),
        Upgrade(
            id="might", name="狂暴之力", description="攻击 +9，普攻技能一起膨胀，伤害直接亿点点提升。",
            icon="💪", rarity="common", add_attack=9,
        ),
        Upgrade(
            id="bulwark", name="钢铁壁垒", description="防御 +8，减伤更高，挨打就是养生。",
            icon="🛡️", rarity="common", add_defense=8,
        ),
        Upgrade(
            id="edge", name="利刃精通", description="所有攻击技能 +11 伤害，刀刀都比之前狠。",
            icon="⚔️", rarity="common", add_skill_damage=11,
        ),
        Upgrade(
            id="focus", name="鹰眼瞄准", description="所有攻击技能 +10% 命中，从此指哪打哪。",
            icon="🎯", rarity="common", add_skill_accuracy=10,
        ),
        Upgrade(
            id="aegis", name="护盾强化", description="防御技能格挡减伤 +16%，铜墙铁壁直接焊死。",
            icon="🔰", rarity="rare", add_skill_block=16,
        ),
        Upgrade(
            id="berserk", name="嗜血狂攻", description="攻击 +8、攻击技能 +10 伤害，全力压制不给喘气。",
            icon="🩸", rarity="rare", add_attack=8, add_skill_damage=10,
        ),
        Upgrade(
            id="fortress", name="不动如山", description="生命 +55、防御 +10，固若金汤就是不让你赢。",
            icon="🏔️", rarity="rare", add_max_hp=55, add_defense=10,
        ),
        Upgrade(
            id="sharpshooter", name="精准杀意", description="攻击技能 +9 伤害、+8% 命中，刀刀直奔要害。",
            icon="🔥", rarity="rare", add_skill_damage=9, add_skill_accuracy=8,
        ),
        Upgrade(
            id="awaken", name="觉醒·全面强化",
            description="生命 +60、攻击 +9、防御 +6，全方位原地觉醒直接起飞。",
            icon="🌟", rarity="epic", add_max_hp=60, add_attack=9, add_defense=6,
        ),
    ]
    # 习得新技能（epic）：随机一个新招式
    skill = random.choice(_GRANTABLE_SKILLS)
    pool.append(
        Upgrade(
            id=f"learn_{skill.name}",
            name=f"习得·{skill.name}",
            description=f"领悟全新招式「{skill.name}」：{skill.description}",
            icon="📜",
            rarity="epic",
            grant_skill=skill,
        )
    )
    return pool


def roll_upgrades(hero: Character, count: int = 3, rng: Optional[random.Random] = None) -> List[Upgrade]:
    """从强化池中抽取 count 个互不相同的强化作为「三选一」。

    若英雄技能已满（4 个），过滤掉「习得新技能」类强化，避免给出无效选项。
    """
    rng = rng or random.Random()
    pool = _upgrade_pool()
    if len(hero.skills) >= 4:
        pool = [u for u in pool if u.grant_skill is None]
    count = min(count, len(pool))
    return rng.sample(pool, k=count)


def apply_upgrade(hero: Character, upgrade: Upgrade) -> Character:
    """把一个强化应用到英雄身上，返回强化后的新英雄。

    所有增量统一裁剪到 Character / Skill 的合法范围，保证属性自洽。
    要点：
    - add_skill_damage / add_skill_accuracy 只作用于「攻击技能」；
    - add_skill_block 只作用于「防御技能」；
    - 习得新技能时，绝不替换掉英雄唯一的防御技能（只在攻击技能里挑最弱的替换）。
    提升 max_hp 视为「同时回满到新上限」——进入下一层时本就以满血开战。
    """
    new_hero = hero.model_copy(deep=True)

    if upgrade.add_max_hp:
        new_hero.max_hp = _clamp(new_hero.max_hp + upgrade.add_max_hp, 50, 300)
    if upgrade.add_attack:
        new_hero.attack = _clamp(new_hero.attack + upgrade.add_attack, 5, 50)
    if upgrade.add_defense:
        new_hero.defense = _clamp(new_hero.defense + upgrade.add_defense, 0, 30)

    for sk in new_hero.skills:
        if sk.kind == "defense":
            if upgrade.add_skill_block:
                sk.block = _clamp(sk.block + upgrade.add_skill_block, 0, 90)
        else:
            if upgrade.add_skill_damage:
                sk.damage = _clamp(sk.damage + upgrade.add_skill_damage, 1, 60)
            if upgrade.add_skill_accuracy:
                sk.accuracy = _clamp(sk.accuracy + upgrade.add_skill_accuracy, 10, 100)

    if upgrade.grant_skill is not None:
        new_skill = upgrade.grant_skill.model_copy(deep=True)
        if not any(sk.name == new_skill.name for sk in new_hero.skills):
            if len(new_hero.skills) < 4:
                new_hero.skills.append(new_skill)
            else:
                # 技能已满：在「攻击技能」里挑伤害最低的替换，保护唯一的防御技能
                atk_idx = [i for i, s in enumerate(new_hero.skills) if s.kind != "defense"]
                if atk_idx:
                    weakest = min(atk_idx, key=lambda i: new_hero.skills[i].damage)
                    new_hero.skills[weakest] = new_skill

    # 强化可能改变了伤害（add_skill_damage / grant_skill），重新分配能量 cost
    character_gen.assign_skill_costs(new_hero)
    return new_hero
