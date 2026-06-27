"""FastAPI 服务入口。

职责：
- 提供 POST /api/battle 接口：接收正反方名字 → 生成角色 → 模拟战斗 → 返回结果
- 提供试炼之塔（打怪升级）一组接口：开局生成英雄+三敌人、逐回合手动选招结算、抽取/应用强化
- 托管 frontend/ 下的静态前端（手机网页直接访问根路径即可）
- 通过环境变量自动决定使用 Claude 联网生成还是 mock 兜底

启动：
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import character_gen
import adventure
import story
import match_history
from battle_engine import apply_action, choose_skill_ai, simulate_battle
from models import (
    AdventureStartRequest,
    AdventureStartResponse,
    ApplyUpgradeRequest,
    ApplyUpgradeResponse,
    BattleRequest,
    BattleResult,
    BattleState,
    ENERGY_PER_TURN,
    MAX_ENERGY,
    RollUpgradesRequest,
    RollUpgradesResponse,
    TurnRequest,
    TurnResponse,
)

# 加载项目根目录下的 .env（若存在）
# override=True：让 .env 总是覆盖 OS 环境变量；否则 shell 里已有的同名变量
# （比如 Claude Code 自己注入的 ANTHROPIC_BASE_URL）会盖掉项目 .env 的设置，
# 导致请求路由到错误端点 + 用对的 key 触发 401。
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("battle")

app = FastAPI(title="名字对战 Demo", version="0.1.0")

# 允许跨域，方便前端在不同端口/设备上调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端目录
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/api/health")
def health() -> dict:
    """健康检查，并告知当前是否具备联网生成能力。"""
    return {
        "status": "ok",
        "claude_enabled": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.post("/api/battle", response_model=BattleResult)
def create_battle(req: BattleRequest) -> BattleResult:
    """生成角色并模拟一整场对战。"""
    side_a = req.side_a.strip()
    side_b = req.side_b.strip()
    if not side_a or not side_b:
        raise HTTPException(status_code=400, detail="双方名字都不能为空")

    # 优先用 Claude 联网生成；任何失败都降级到 mock，保证 demo 可用
    source = "mock"
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            char_a, char_b, intro = character_gen.generate_with_claude(side_a, side_b)
            source = "claude"
        else:
            char_a, char_b, intro = character_gen.generate_mock(side_a, side_b)
    except Exception as exc:  # noqa: BLE001 — 任何异常都兜底，避免前端报错
        logger.warning("Claude 生成失败，降级到 mock：%s", exc)
        char_a, char_b, intro = character_gen.generate_mock(side_a, side_b)
        source = "mock"

    result = simulate_battle(char_a, char_b, intro=intro, source=source)
    return result


# ===========================================================================
# 试炼之塔（打怪升级模式）接口
# ===========================================================================
# 后端保持无状态：英雄、敌人、战斗状态都由前端持有并回传。
#   1) POST /api/adventure/start    生成英雄 + 按场地生成三层敌人
#   2) POST /api/adventure/turn     手动选招：结算我方行动 + 敌方 AI 行动
#   3) POST /api/adventure/upgrades 通关一层后抽取三选一强化
#   4) POST /api/adventure/upgrade  应用选中的强化到英雄


@app.post("/api/adventure/start", response_model=AdventureStartResponse)
def adventure_start(req: AdventureStartRequest) -> AdventureStartResponse:
    """根据「勇者之名」生成英雄，并按「挑战场地」生成三层主题敌人。"""
    name = req.name.strip()
    arena = (req.arena or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="勇者之名不能为空")

    # 英雄：优先 Claude 主题化生成，失败降级 mock
    hero_source = "mock"
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            hero, hero_intro = character_gen.generate_hero_with_claude(name)
            hero_source = "claude"
        else:
            hero, hero_intro = character_gen.generate_hero_mock(name)
    except Exception as exc:  # noqa: BLE001 — 任何异常都兜底
        logger.warning("英雄生成失败，降级到 mock：%s", exc)
        hero, hero_intro = character_gen.generate_hero_mock(name)
        hero_source = "mock"

    # 三层敌人：优先 Claude 按场地生成，失败降级 mock（adventure 内部已兜底）
    enemies, enemies_intro, enemies_source = adventure.generate_enemies(arena)

    # 整体来源：英雄与敌人都来自 Claude 才算 claude，否则视作 mock（提示更诚实）
    source = "claude" if (hero_source == "claude" and enemies_source == "claude") else "mock"
    intro = enemies_intro or hero_intro

    return AdventureStartResponse(
        hero=hero,
        enemies=enemies,
        arena=arena,
        intro=intro,
        source=source,  # type: ignore[arg-type]
    )


@app.post("/api/adventure/turn", response_model=TurnResponse)
def adventure_turn(req: TurnRequest) -> TurnResponse:
    """手动选招的单回合结算：英雄先用所选技能行动，未分胜负则敌方 AI 回应。

    能量规则：
      - 玩家选的技能 cost 必须 <= 当前英雄能量，否则返 400
      - 敌方 AI 只会选 cost <= 当前敌人能量 的技能
      - 双方行动后能量各扣 skill.cost
      - 一回合结束后，双方各回 ENERGY_PER_TURN 能量（上限 MAX_ENERGY），作为下回合可用资源
    """
    hero = req.hero
    enemy = req.enemy
    st = req.state

    if not 0 <= req.skill_index < len(hero.skills):
        raise HTTPException(status_code=400, detail="技能下标越界")

    hero_skill = hero.skills[req.skill_index]
    if hero_skill.cost > st.hero_energy:
        raise HTTPException(status_code=400, detail="能量不足，换个技能试试")

    rng = random.Random()
    hp = {hero.name: st.hero_hp, enemy.name: st.enemy_hp}
    block = {hero.name: st.hero_block, enemy.name: st.enemy_block}
    energy = {hero.name: st.hero_energy, enemy.name: st.enemy_energy}
    events = []
    turn = st.turn

    # ---- 英雄行动（玩家选招）----
    turn += 1
    ev = apply_action(hero, enemy, hero_skill, hp, block, turn, rng)
    energy[hero.name] = max(0, energy[hero.name] - hero_skill.cost)
    ev.actor_energy_after = energy[hero.name]
    events.append(ev)

    battle_over = False
    winner = ""
    if hp[enemy.name] <= 0:
        battle_over = True
        winner = hero.name
    else:
        # ---- 敌方行动（AI 选招，遵守能量预算）----
        turn += 1
        enemy_skill = choose_skill_ai(
            enemy,
            hp_ratio=hp[enemy.name] / enemy.max_hp,
            already_blocking=block[enemy.name] > 0,
            rng=rng,
            energy=energy[enemy.name],
        )
        ev2 = apply_action(enemy, hero, enemy_skill, hp, block, turn, rng)
        energy[enemy.name] = max(0, energy[enemy.name] - enemy_skill.cost)
        ev2.actor_energy_after = energy[enemy.name]
        events.append(ev2)
        if hp[hero.name] <= 0:
            battle_over = True
            winner = enemy.name

    # 回合结算后：双方各回 +ENERGY_PER_TURN 能量（战斗结束就不再回了）
    if not battle_over:
        for name in energy:
            energy[name] = min(MAX_ENERGY, energy[name] + ENERGY_PER_TURN)

    new_state = BattleState(
        hero_hp=hp[hero.name],
        enemy_hp=hp[enemy.name],
        hero_block=block[hero.name],
        enemy_block=block[enemy.name],
        hero_energy=energy[hero.name],
        enemy_energy=energy[enemy.name],
        turn=turn,
    )
    return TurnResponse(
        events=events, state=new_state, battle_over=battle_over, winner=winner
    )


@app.post("/api/adventure/upgrades", response_model=RollUpgradesResponse)
def adventure_roll_upgrades(req: RollUpgradesRequest) -> RollUpgradesResponse:
    """通关一层后，为英雄抽取三选一强化。"""
    upgrades = adventure.roll_upgrades(req.hero, count=3)
    return RollUpgradesResponse(upgrades=upgrades)


@app.post("/api/adventure/upgrade", response_model=ApplyUpgradeResponse)
def adventure_upgrade(req: ApplyUpgradeRequest) -> ApplyUpgradeResponse:
    """把玩家选中的强化应用到英雄身上，返回强化后的英雄。"""
    new_hero = adventure.apply_upgrade(req.hero, req.upgrade)
    return ApplyUpgradeResponse(hero=new_hero)


# ===========================================================================
# 故事对决（compete! 项目移植）：三局两胜的叙事性战斗
# ===========================================================================
# 与 /api/battle 的「自动战斗 + 数值结算」不同，故事模式：
#   1) /api/story/characters  生成两位叙事用角色（简介 + 技能描述，无数值）
#   2) /api/story/round       每回合 Claude 生成 8-12 句精彩解说，前端打字机展示
#   3) 整场由前端按 score 推进，三局两胜，与 compete! 一致

from pydantic import BaseModel as _BM  # type: ignore  # 局部 import 避免顶部噪声
from typing import Any as _Any


class StoryCharsRequest(_BM):
    name_a: str
    name_b: str


class StoryRoundRequest(_BM):
    protagonist: dict
    antagonist: dict
    round_number: int
    score: dict  # {"protagonist": int, "antagonist": int}
    history: list[str] = []


class SaveMatchRequest(_BM):
    mode: str = "story"
    protagonist: dict
    antagonist: dict
    rounds: list[dict] = []
    final_score: dict = {"protagonist": 0, "antagonist": 0}
    champion: str = ""
    champion_name: str = ""


@app.post("/api/story/characters")
def story_generate_characters(req: StoryCharsRequest) -> dict:
    """为故事对决生成正反双方角色（简介 + 技能描述，无数值）。"""
    name_a = req.name_a.strip()
    name_b = req.name_b.strip()
    if not name_a or not name_b:
        raise HTTPException(status_code=400, detail="双方名字都不能为空")
    source = "mock"
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            pro, ant = story.generate_characters_with_claude(name_a, name_b)
            source = "claude"
        else:
            pro, ant = story.generate_characters_mock(name_a, name_b)
    except Exception as exc:  # noqa: BLE001
        logger.warning("故事角色生成失败，降级 mock：%s", exc)
        pro, ant = story.generate_characters_mock(name_a, name_b)
        source = "mock"
    return {"protagonist": pro, "antagonist": ant, "source": source}


@app.post("/api/story/round")
def story_play_round(req: StoryRoundRequest) -> dict:
    """进行一回合故事对决：Claude 生成解说 + 后端权威推进比分。"""
    source = "mock"
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            result = story.play_round_with_claude(
                req.protagonist, req.antagonist, req.round_number, req.score, req.history,
            )
            source = "claude"
        else:
            result = story.play_round_mock(
                req.protagonist, req.antagonist, req.round_number, req.score, req.history,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("故事回合生成失败，降级 mock：%s", exc)
        result = story.play_round_mock(
            req.protagonist, req.antagonist, req.round_number, req.score, req.history,
        )
        source = "mock"

    new_score, over, champion = story.advance_score(req.score, result["winner"])
    champion_name = ""
    if champion == "protagonist":
        champion_name = req.protagonist.get("name", "")
    elif champion == "antagonist":
        champion_name = req.antagonist.get("name", "")
    return {
        "result": result,
        "score": new_score,
        "match_over": over,
        "champion": champion,
        "champion_name": champion_name,
        "source": source,
    }


# ===========================================================================
# 比赛存档 + 排行榜（compete! 项目移植；三种模式都可入库）
# ===========================================================================

@app.post("/api/matches")
def save_match_endpoint(req: SaveMatchRequest) -> dict:
    """保存一场比赛存档（任一模式都可调用）。"""
    match_id = match_history.save_match(req.model_dump())
    return {"id": match_id}


@app.get("/api/matches")
def list_matches_endpoint() -> dict:
    return {"matches": match_history.list_matches()}


@app.get("/api/matches/{match_id}")
def get_match_endpoint(match_id: str) -> dict:
    rec = match_history.get_match(match_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="未找到该比赛记录")
    return rec


@app.delete("/api/matches/{match_id}")
def delete_match_endpoint(match_id: str) -> dict:
    """删除一场比赛存档；用于前端"删除历史"时同步清掉后端存档，
    让排行榜（基于扫描所有存档聚合）自动反映胜率变化。"""
    ok = match_history.delete_match(match_id)
    if not ok:
        raise HTTPException(status_code=404, detail="未找到该比赛记录")
    return {"deleted": True, "id": match_id}


@app.get("/api/leaderboard")
def get_leaderboard() -> dict:
    """返回 4 类排行榜（按胜率/胜场分别排序）。"""
    return match_history.get_leaderboards()


@app.get("/api/characters/{name}/profile")
def get_character_profile(name: str) -> dict:
    data = match_history.get_character_profile(name)
    if data is None:
        raise HTTPException(status_code=404, detail="该角色暂无对局记录")
    return data


# ---- 静态前端托管（放在最后，避免覆盖 /api 路由）----
if _FRONTEND_DIR.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_FRONTEND_DIR / "index.html"))

    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
