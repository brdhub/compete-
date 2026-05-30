"""数据模型：定义角色、技能、对战请求与响应的结构。

这些 Pydantic 模型既用于 FastAPI 的请求/响应校验，
也作为 Claude 生成角色时强制约束的 JSON 结构（通过工具调用 schema）。
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Skill(BaseModel):
    """一个技能。伤害与角色血量在同一量级，便于战斗引擎做平衡。

    技能分两类（kind）：
    - attack ：攻击技能，对敌人造成 damage 伤害。
    - defense：防御技能，自身不造成伤害，而是进入「格挡」姿态——
               抵挡对手下一次攻击 block% 的伤害，格挡用掉即消失。
    """

    name: str = Field(..., description="技能名称")
    description: str = Field(..., description="技能的简短描述")
    damage: int = Field(..., ge=0, le=60, description="基础伤害值（防御技能为 0）")
    # 命中率用 0~100 的整数表示，避免浮点带来的歧义
    accuracy: int = Field(default=90, ge=10, le=100, description="命中率(百分比)")
    kind: Literal["attack", "defense"] = Field(
        default="attack", description="技能类型：attack=攻击，defense=防御格挡"
    )
    block: int = Field(
        default=0, ge=0, le=90, description="防御技能格挡下一击的减伤百分比(0~90)"
    )
    # 能量消耗：防御 + 最弱攻击 = 0、次强 = 1、最强 = 2。由后端按伤害排序自动分配。
    cost: int = Field(default=0, ge=0, le=2, description="技能能量消耗(0/1/2)")


class Character(BaseModel):
    """对战中的一个角色。"""

    name: str = Field(..., description="角色名字")
    title: str = Field(default="", description="称号/简短身份描述")
    max_hp: int = Field(..., ge=50, le=300, description="最大血量")
    attack: int = Field(..., ge=5, le=50, description="基础攻击力(用于普通攻击)")
    defense: int = Field(default=0, ge=0, le=30, description="防御力，按比例减伤")
    skills: List[Skill] = Field(..., min_length=1, max_length=4, description="技能列表")
    flavor: str = Field(default="", description="一段背景风味描述")


class BattleRequest(BaseModel):
    """前端发起对战的请求体。"""

    side_a: str = Field(..., min_length=1, max_length=50, description="正方名字")
    side_b: str = Field(..., min_length=1, max_length=50, description="反方名字")


class TurnEvent(BaseModel):
    """一个回合中发生的单次行动记录。"""

    turn: int = Field(..., description="第几回合")
    actor: str = Field(..., description="行动者名字")
    target: str = Field(..., description="目标名字")
    skill: str = Field(..., description="使用的技能名")
    kind: Literal["attack", "defense"] = Field(
        default="attack", description="行动类型：attack=攻击，defense=进入格挡姿态"
    )
    hit: bool = Field(..., description="是否命中（防御行动恒为 True）")
    damage: int = Field(..., description="实际造成的伤害（已扣除格挡减免后）")
    blocked: int = Field(default=0, description="本次被对手格挡减免掉的伤害量")
    block: int = Field(default=0, description="防御行动时进入的格挡减伤%（仅 kind=defense 有意义）")
    cost: int = Field(default=0, description="本次行动消耗的能量")
    actor_energy_after: int = Field(default=-1, description="本次行动后行动者剩余能量；-1 表示未跟踪")
    target_hp_after: int = Field(..., description="目标行动后剩余血量")
    narration: str = Field(..., description="这次行动的文字解说")


class BattleResult(BaseModel):
    """完整对战结果，返回给前端。"""

    character_a: Character
    character_b: Character
    events: List[TurnEvent]
    winner: str = Field(..., description="胜方名字，平局为 '平局'")
    source: Literal["claude", "mock"] = Field(
        ..., description="角色属性来源：claude=联网生成，mock=本地兜底"
    )
    intro: str = Field(default="", description="开场白")


# ===========================================================================
# 试炼之塔（打怪升级模式）相关模型
# ===========================================================================
# 玩法：玩家输入「勇者之名」+「挑战场地」→ 生成英雄 + 三个主题敌人。
# 战斗为「手动选招」的回合制：每回合玩家点一个技能，后端结算我方行动 + 敌方 AI 行动，
# 返回本回合发生的事件与更新后的战斗状态。通过一层可获得「三选一」强化。
# 三层全部通关即胜利，途中任意一层落败即试炼终止。
#
# 后端保持无状态：英雄、敌人、战斗状态都由前端持有并在每次请求中带上，
# 强化的应用、属性边界裁剪、单回合结算都在后端完成，保证数据始终自洽。

TOWER_FLOORS = 3  # 试炼之塔的总层数

# ===========================================================================
# 能量系统：双方各持一条 0~MAX_ENERGY 的能量条
# - 开局能量 = INITIAL_ENERGY（3）
# - 每个完整回合（双方各出一招）结束后，双方各回 ENERGY_PER_TURN（1），上限 MAX_ENERGY
# - 技能 cost：防御 + 最弱攻击 = 0、次强攻击 = 1、最强攻击 = 2（按伤害排序自动分配）
# ===========================================================================
INITIAL_ENERGY = 3
MAX_ENERGY = 5
ENERGY_PER_TURN = 1


class Upgrade(BaseModel):
    """一次「三选一」强化选项。各 add_* 字段描述对英雄属性的增量改动。"""

    id: str = Field(..., description="强化唯一标识")
    name: str = Field(..., description="强化名称（酷炫招牌名）")
    description: str = Field(..., description="强化效果的一句话说明")
    icon: str = Field(default="✨", description="展示用的图标/emoji")
    rarity: Literal["common", "rare", "epic"] = Field(
        default="common", description="稀有度，用于前端配色"
    )
    # 以下为对英雄属性的增量；apply 时统一裁剪到 Character 的合法范围
    add_max_hp: int = Field(default=0, description="最大生命增量")
    add_attack: int = Field(default=0, description="攻击力增量")
    add_defense: int = Field(default=0, description="防御力增量")
    add_skill_damage: int = Field(default=0, description="对所有攻击技能的伤害增量")
    add_skill_accuracy: int = Field(default=0, description="对所有技能的命中增量")
    add_skill_block: int = Field(default=0, description="对防御技能的格挡减伤增量")
    grant_skill: Optional[Skill] = Field(default=None, description="习得的全新技能（可选）")


class AdventureStartRequest(BaseModel):
    """开始一段试炼：玩家提供勇者之名与挑战场地。"""

    name: str = Field(..., min_length=1, max_length=50, description="英雄名字")
    arena: str = Field(
        default="", max_length=50, description="挑战场地（据此主题化生成三个敌人）"
    )


class AdventureStartResponse(BaseModel):
    """开局结果：英雄 + 三个随层数递增的主题敌人。"""

    hero: Character
    enemies: List[Character] = Field(..., description="三层敌人，按层数从弱到强")
    arena: str = Field(default="", description="实际使用的挑战场地")
    intro: str = Field(default="", description="试炼登场介绍")
    source: Literal["claude", "mock"] = Field(..., description="生成来源")


class BattleState(BaseModel):
    """一场对战的可变状态，由前端持有并逐回合回传，保证后端无状态。"""

    hero_hp: int = Field(..., ge=0, description="英雄当前血量")
    enemy_hp: int = Field(..., ge=0, description="敌人当前血量")
    # 防御「格挡姿态」：>0 表示下一次被击将按该百分比减伤，触发后清零
    hero_block: int = Field(default=0, ge=0, le=90, description="英雄待生效的格挡减伤%")
    enemy_block: int = Field(default=0, ge=0, le=90, description="敌人待生效的格挡减伤%")
    # 能量：开局 INITIAL_ENERGY，使用技能消耗 cost，每回合双方各 +1（上限 MAX_ENERGY）
    hero_energy: int = Field(default=INITIAL_ENERGY, ge=0, le=MAX_ENERGY, description="英雄当前能量")
    enemy_energy: int = Field(default=INITIAL_ENERGY, ge=0, le=MAX_ENERGY, description="敌人当前能量")
    turn: int = Field(default=0, description="已进行的行动次数（用于解说编号）")


class TurnRequest(BaseModel):
    """手动选招：在当前战斗状态下，英雄使用第 skill_index 个技能。"""

    hero: Character
    enemy: Character
    state: BattleState
    skill_index: int = Field(..., ge=0, description="英雄本回合使用的技能下标")


class TurnResponse(BaseModel):
    """一个回合（我方行动 + 敌方行动）的结算结果。"""

    events: List[TurnEvent] = Field(..., description="本回合发生的行动序列")
    state: BattleState = Field(..., description="结算后的战斗状态")
    battle_over: bool = Field(..., description="本场战斗是否结束")
    winner: str = Field(default="", description="battle_over 时的胜方名字")


class RollUpgradesRequest(BaseModel):
    """通关一层后，为英雄抽取三选一强化。"""

    hero: Character


class RollUpgradesResponse(BaseModel):
    upgrades: List[Upgrade] = Field(..., description="三选一强化选项")


class ApplyUpgradeRequest(BaseModel):
    """应用一个强化到英雄身上。"""

    hero: Character
    upgrade: Upgrade


class ApplyUpgradeResponse(BaseModel):
    """应用强化后的英雄。"""

    hero: Character
