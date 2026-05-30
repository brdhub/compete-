"""战斗引擎：纯 Python 实现的确定性回合制对战。

为什么不让 Claude 直接模拟战斗：让大模型一边记血量一边讲故事，
血量加减经常前后对不上、回合数据不可靠。这里把"讲故事"和"算数值"分开——
属性由 Claude 生成，战斗过程由真实代码计算，保证血条和每个回合的数据百分百自洽。

本模块的核心是 `resolve_action`：结算「一个角色用一个技能对另一个角色的单次行动」，
返回这次行动的 TurnEvent 与对战斗状态的影响。自动对战（名字对战）与手动选招
（试炼之塔）都复用这同一套结算逻辑，行为完全一致。

两条关键规则（本次重塑）：
1. 大幅下调闪避：攻击基本必中，只有低命中技能才偶尔被闪开。
   实际命中率 = 100 - (100 - accuracy) * DODGE_FACTOR，并设有下限。
2. 防御技能（kind=defense）：自身不造成伤害，而是进入「格挡姿态」，
   抵挡对手下一次命中的 block% 伤害；格挡在被一次命中用掉后消失。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from models import (
    INITIAL_ENERGY,
    MAX_ENERGY,
    ENERGY_PER_TURN,
    BattleResult,
    Character,
    Skill,
    TurnEvent,
)

# 单场（自动）战斗的回合上限，避免双方都很肉时无限循环
_MAX_TURNS = 80

# 闪避衰减系数：把「miss 概率」压到原来的 1/4。
# 例：命中 60 的技能，miss 概率从 40% 降到 10%；命中 90 的从 10% 降到 2.5%。
DODGE_FACTOR = 0.25
# 命中率下限：再低命中的技能也至少有这么高的实际命中率，杜绝"频繁落空"的挫败感。
_MIN_HIT = 80


def effective_hit_chance(accuracy: int) -> int:
    """把技能标称命中率换算成「实际命中率」（大幅下调闪避后的真实值）。"""
    raw = 100 - round((100 - accuracy) * DODGE_FACTOR)
    return max(_MIN_HIT, min(100, raw))


def _attack_skills(char: Character) -> List[Skill]:
    return [s for s in char.skills if s.kind != "defense"]


def _defense_skills(char: Character) -> List[Skill]:
    return [s for s in char.skills if s.kind == "defense"]


def choose_skill_ai(
    char: Character,
    hp_ratio: float,
    already_blocking: bool,
    rng: random.Random,
    energy: int = MAX_ENERGY,
) -> Skill:
    """敌方/自动方的选招 AI。

    - 攻击技能按伤害加权随机，偏向高伤但保留变数。
    - 残血且当前没有格挡时，更倾向于使用防御技能自保。
    - 能量不够付 cost 的技能直接排除（防御 cost=0 总能用作兜底）。
    """
    affordable_atk = [s for s in _attack_skills(char) if s.cost <= energy]
    dfn = [s for s in _defense_skills(char) if s.cost <= energy]

    if dfn and not already_blocking:
        defend_prob = 0.16
        if hp_ratio < 0.35:
            defend_prob = 0.55
        elif hp_ratio < 0.6:
            defend_prob = 0.28
        # 攻击技能全部用不起时，几乎一定走防御（兜底）
        if not affordable_atk:
            defend_prob = 0.95
        if rng.random() < defend_prob:
            return rng.choice(dfn)

    if affordable_atk:
        weights = [max(1, sk.damage) for sk in affordable_atk]
        return rng.choices(affordable_atk, weights=weights, k=1)[0]
    if dfn:
        return rng.choice(dfn)
    # 极端兜底：什么都用不起就硬拿最便宜的一个（理论上不会触发，因为防御 cost=0）
    return min(char.skills, key=lambda s: s.cost)


def _compute_damage(skill: Skill, attacker: Character, defender: Character, rng: random.Random) -> int:
    """伤害 = (技能基础伤害 + 攻击力的一部分) * 随机浮动，再按防御减伤。"""
    base = skill.damage + attacker.attack * 0.3
    # ±15% 随机浮动，让同样的技能每次略有差异
    fluctuation = rng.uniform(0.85, 1.15)
    raw = base * fluctuation
    # 防御按比例减伤：defense 30 时减约 23%（避免完全免疫）
    reduction = defender.defense / (defender.defense + 100)
    final = raw * (1 - reduction)
    return max(1, round(final))


def _narrate_attack(
    actor: Character, target: Character, skill: Skill, hit: bool,
    damage: int, blocked: int, hp_after: int,
) -> str:
    """为一次攻击行动生成文字解说。"""
    if not hit:
        return f"{actor.name} 使出「{skill.name}」，但 {target.name} 堪堪侧身避开，未造成伤害！"
    desc = f"{actor.name} 使出「{skill.name}」，{skill.description}"
    if blocked > 0:
        desc += f"，被 {target.name} 的格挡卸去 {blocked} 点，仍造成 {damage} 点伤害"
    else:
        desc += f"，对 {target.name} 造成 {damage} 点伤害"
    if hp_after <= 0:
        return desc + f"！{target.name} 倒下了！"
    if hp_after <= target.max_hp * 0.25:
        return desc + f"！{target.name} 已是强弩之末（剩余 {hp_after} HP）。"
    return desc + f"（剩余 {hp_after} HP）。"


def _narrate_defense(actor: Character, skill: Skill) -> str:
    """为一次防御行动生成文字解说。"""
    return (
        f"{actor.name} 摆出「{skill.name}」的架势，{skill.description}，"
        f"凝神戒备，将抵挡下一次攻击 {skill.block}% 的伤害。"
    )


@dataclass
class ActionOutcome:
    """单次行动的结算产物。"""

    event: TurnEvent
    damage: int                 # 对 defender 造成的实际伤害（已扣格挡）
    actor_block_set: int        # 防御行动时给 actor 设置的格挡%；攻击行动为 -1（表示不修改）
    defender_block_after: int   # defender 行动后剩余格挡（被命中则清零）


def resolve_action(
    actor: Character,
    defender: Character,
    skill: Skill,
    defender_block: int,
    turn: int,
    rng: random.Random,
) -> ActionOutcome:
    """结算 actor 使用 skill 对 defender 的一次行动。

    参数：
        defender_block：defender 当前持有的格挡减伤%（>0 表示会抵挡这次命中）。
    返回 ActionOutcome，调用方据此更新 hp 与双方的格挡状态。
    """
    # ---- 防御技能：进入格挡姿态，不造成伤害 ----
    if skill.kind == "defense":
        event = TurnEvent(
            turn=turn,
            actor=actor.name,
            target=actor.name,  # 防御作用于自身
            skill=skill.name,
            kind="defense",
            hit=True,
            damage=0,
            blocked=0,
            block=skill.block,
            cost=skill.cost,
            target_hp_after=-1,  # -1 表示血量未变化，前端无需更新血条
            narration=_narrate_defense(actor, skill),
        )
        return ActionOutcome(
            event=event,
            damage=0,
            actor_block_set=skill.block,
            defender_block_after=defender_block,  # 不影响对手的格挡
        )

    # ---- 攻击技能 ----
    hit = rng.randint(1, 100) <= effective_hit_chance(skill.accuracy)
    damage = 0
    blocked = 0
    defender_block_after = defender_block

    if hit:
        raw = _compute_damage(skill, actor, defender, rng)
        if defender_block > 0:
            blocked = round(raw * defender_block / 100)
            damage = max(0, raw - blocked)
            defender_block_after = 0  # 格挡被这次命中用掉
        else:
            damage = raw
    # miss 时不消耗对手的格挡（格挡只在真正挨打时用掉）

    return ActionOutcome(
        event=TurnEvent(
            turn=turn,
            actor=actor.name,
            target=defender.name,
            skill=skill.name,
            kind="attack",
            hit=hit,
            damage=damage,
            blocked=blocked,
            cost=skill.cost,
            target_hp_after=-1,  # 由调用方算出实际剩余血量后填入
            narration="",         # 由调用方在得到 hp_after 后补全
        ),
        damage=damage,
        actor_block_set=-1,
        defender_block_after=defender_block_after,
    )


def apply_action(
    actor: Character,
    defender: Character,
    skill: Skill,
    hp: dict,
    block: dict,
    turn: int,
    rng: random.Random,
) -> TurnEvent:
    """结算一次行动并就地更新 hp / block 两个字典，返回补全好的 TurnEvent。

    这是自动战斗与手动选招（试炼之塔）共用的唯一结算入口，保证两种模式行为一致。
        hp[name]    ：各角色当前血量
        block[name] ：各角色待生效的格挡减伤%
    """
    outcome = resolve_action(actor, defender, skill, block[defender.name], turn, rng)
    ev = outcome.event

    if skill.kind == "defense":
        # 进入格挡姿态：给自己设置格挡，不改血量
        block[actor.name] = outcome.actor_block_set
    else:
        block[defender.name] = outcome.defender_block_after
        hp[defender.name] = max(0, hp[defender.name] - outcome.damage)
        ev.target_hp_after = hp[defender.name]
        ev.narration = _narrate_attack(
            actor, defender, skill, ev.hit, ev.damage, ev.blocked, hp[defender.name]
        )
    return ev


def simulate_battle(
    char_a: Character,
    char_b: Character,
    intro: str,
    source: str,
    seed: Optional[int] = None,
) -> BattleResult:
    """模拟一整场（自动）对战并返回完整结果。用于「名字对战」模式。

    先手由攻击力决定（高者先动），相同则正方先动。
    双方均由 choose_skill_ai 自动选招，复用 apply_action 结算，支持防御/格挡。
    能量：双方开局 INITIAL_ENERGY；每完成一轮（双方各出一招）后各回 ENERGY_PER_TURN，
    上限 MAX_ENERGY；AI 不会选超出能量预算的技能。
    """
    rng = random.Random(seed)

    hp = {char_a.name: char_a.max_hp, char_b.name: char_b.max_hp}
    block = {char_a.name: 0, char_b.name: 0}
    energy = {char_a.name: INITIAL_ENERGY, char_b.name: INITIAL_ENERGY}
    events: List[TurnEvent] = []

    # 决定出手顺序
    if char_b.attack > char_a.attack:
        order = [char_b, char_a]
    else:
        order = [char_a, char_b]

    turn = 0
    winner = "平局"
    rounds = 0
    while rounds < _MAX_TURNS:
        rounds += 1
        for attacker in order:
            defender = char_b if attacker is char_a else char_a
            if hp[attacker.name] <= 0 or hp[defender.name] <= 0:
                break

            turn += 1
            hp_ratio = hp[attacker.name] / attacker.max_hp
            skill = choose_skill_ai(
                attacker,
                hp_ratio,
                already_blocking=block[attacker.name] > 0,
                rng=rng,
                energy=energy[attacker.name],
            )

            ev = apply_action(attacker, defender, skill, hp, block, turn, rng)
            energy[attacker.name] = max(0, energy[attacker.name] - skill.cost)
            ev.actor_energy_after = energy[attacker.name]
            events.append(ev)

            if skill.kind != "defense" and hp[defender.name] <= 0:
                winner = attacker.name
                break

        if hp[char_a.name] <= 0 or hp[char_b.name] <= 0:
            break
        # 一轮结束后，双方各回 +ENERGY_PER_TURN（上限 MAX_ENERGY）
        for name in energy:
            energy[name] = min(MAX_ENERGY, energy[name] + ENERGY_PER_TURN)

    # 到达回合上限仍未分胜负：按剩余血量比例判定
    if winner == "平局":
        ratio_a = hp[char_a.name] / char_a.max_hp
        ratio_b = hp[char_b.name] / char_b.max_hp
        if ratio_a > ratio_b:
            winner = char_a.name
        elif ratio_b > ratio_a:
            winner = char_b.name

    return BattleResult(
        character_a=char_a,
        character_b=char_b,
        events=events,
        winner=winner,
        source=source,  # type: ignore[arg-type]
        intro=intro,
    )
