"""比赛存档 + 排行榜聚合（移植自 compete! 项目）。

存档：每场比赛存为 backend/data/matches/{id}.json，零依赖、易迁移。
所有三种模式（名字对战、试炼之塔、故事对决）的胜负结果都可入库，
排行榜按角色名累加各方面胜负数据，给出 4 类榜单。

存档结构（统一）：
{
  "id": "短 hex",
  "created_at": "ISO 8601",
  "mode": "battle|tower|story",
  "protagonist": {"name": "...", "source": "...", "summary": "..."},
  "antagonist":  {"name": "...", "source": "...", "summary": "..."},
  "rounds": [{"round_number":1,"winner":"protagonist","winner_name":"...","narration":"..."}],
  "final_score": {"protagonist":2, "antagonist":1},
  "champion": "protagonist|antagonist",
  "champion_name": "..."
}

「大场次」= 一整场胜负；「小场次」= 单回合胜负。
"""
from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent / "data" / "matches"
_LOCK = threading.Lock()

MIN_MATCHES_FOR_RATE = 1  # 至少打过 1 场即上胜率榜
MIN_ROUNDS_FOR_RATE = 1


def _ensure_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _path(match_id: str) -> Path:
    return _DATA_DIR / f"{match_id}.json"


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------

def save_match(payload: dict[str, Any]) -> str:
    """保存一场比赛，返回新 id。payload 字段见模块顶部 docstring。"""
    match_id = uuid.uuid4().hex[:12]
    created_at = datetime.now().isoformat(timespec="seconds")

    record = {
        "id": match_id,
        "created_at": created_at,
        "mode": str(payload.get("mode") or "story"),
        "protagonist": payload.get("protagonist") or {},
        "antagonist": payload.get("antagonist") or {},
        "rounds": payload.get("rounds") or [],
        "final_score": payload.get("final_score") or {"protagonist": 0, "antagonist": 0},
        "champion": str(payload.get("champion") or ""),
        "champion_name": str(payload.get("champion_name") or ""),
    }

    with _LOCK:
        _ensure_dir()
        _path(match_id).write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return match_id


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def _load_all() -> list[dict[str, Any]]:
    with _LOCK:
        files = list(_DATA_DIR.glob("*.json")) if _DATA_DIR.exists() else []
    out: list[dict[str, Any]] = []
    for p in files:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def list_matches() -> list[dict[str, Any]]:
    """返回所有比赛概要（按时间倒序）。"""
    records = _load_all()
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return [
        {
            "id": r.get("id", ""),
            "created_at": r.get("created_at", ""),
            "mode": r.get("mode", ""),
            "protagonist_name": (r.get("protagonist") or {}).get("name", ""),
            "antagonist_name": (r.get("antagonist") or {}).get("name", ""),
            "champion_name": r.get("champion_name", ""),
            "final_score": r.get("final_score", {"protagonist": 0, "antagonist": 0}),
        }
        for r in records
    ]


def get_match(match_id: str) -> dict[str, Any] | None:
    with _LOCK:
        p = _path(match_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def delete_match(match_id: str) -> bool:
    """删除一场比赛存档。返回是否删除成功。

    用于「在前端历史档案里删除某条记录时，同步把后端的存档也删掉」，
    这样排行榜（基于扫描存档目录聚合）会自动反映出胜率变化。
    """
    with _LOCK:
        p = _path(match_id)
        if not p.exists():
            return False
        try:
            p.unlink()
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# 排行榜聚合
# ---------------------------------------------------------------------------

def _empty_stat() -> dict[str, Any]:
    return {
        "name": "",
        "source": "",
        "summary": "",
        "matches_played": 0,
        "matches_won": 0,
        "rounds_played": 0,
        "rounds_won": 0,
    }


def _accumulate(stats: dict[str, dict[str, Any]], rec: dict[str, Any], side: str) -> None:
    role = (rec.get(side) or {})
    name = (role.get("name") or "").strip()
    if not name:
        return
    s = stats[name]
    s["name"] = name
    if role.get("source"):
        s["source"] = role["source"]
    if role.get("summary"):
        s["summary"] = role["summary"]
    s["matches_played"] += 1
    if rec.get("champion") == side:
        s["matches_won"] += 1
    for r in rec.get("rounds") or []:
        s["rounds_played"] += 1
        if r.get("winner") == side:
            s["rounds_won"] += 1


def _aggregate() -> list[dict[str, Any]]:
    records = _load_all()
    records.sort(key=lambda r: r.get("created_at", ""))  # 让 source/summary 取到最新
    stats: dict[str, dict[str, Any]] = defaultdict(_empty_stat)
    for rec in records:
        _accumulate(stats, rec, "protagonist")
        _accumulate(stats, rec, "antagonist")
    return list(stats.values())


def _with_rates(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in stats:
        item = dict(s)
        mp = s["matches_played"]
        rp = s["rounds_played"]
        item["match_rate"] = (s["matches_won"] / mp) if mp else 0.0
        item["round_rate"] = (s["rounds_won"] / rp) if rp else 0.0
        out.append(item)
    return out


def get_leaderboards() -> dict[str, list[dict[str, Any]]]:
    """返回 4 个榜单：match_rate / round_rate / match_wins / round_wins。"""
    all_stats = _with_rates(_aggregate())
    match_rate = sorted(
        (s for s in all_stats if s["matches_played"] >= MIN_MATCHES_FOR_RATE),
        key=lambda s: (-s["match_rate"], -s["matches_won"], -s["matches_played"]),
    )
    round_rate = sorted(
        (s for s in all_stats if s["rounds_played"] >= MIN_ROUNDS_FOR_RATE),
        key=lambda s: (-s["round_rate"], -s["rounds_won"], -s["rounds_played"]),
    )
    match_wins = sorted(all_stats, key=lambda s: (-s["matches_won"], -s["match_rate"]))
    round_wins = sorted(all_stats, key=lambda s: (-s["rounds_won"], -s["round_rate"]))
    return {
        "match_rate": match_rate,
        "round_rate": round_rate,
        "match_wins": match_wins,
        "round_wins": round_wins,
    }


def get_character_profile(name: str) -> dict[str, Any] | None:
    """返回某角色的简介 + 全部历史对局（对手视角）。无记录返回 None。"""
    name = (name or "").strip()
    if not name:
        return None
    records = _load_all()
    if not records:
        return None
    records.sort(key=lambda r: r.get("created_at", ""))

    matches: list[dict[str, Any]] = []
    source = summary = ""
    mp = mw = rp = rw = 0

    for rec in records:
        pro = rec.get("protagonist") or {}
        ant = rec.get("antagonist") or {}
        is_pro = (pro.get("name") or "") == name
        is_ant = (ant.get("name") or "") == name
        if not is_pro and not is_ant:
            continue
        side = "protagonist" if is_pro else "antagonist"
        opp_side = "antagonist" if is_pro else "protagonist"
        me = pro if is_pro else ant
        opp = ant if is_pro else pro
        if me.get("source"):
            source = me["source"]
        if me.get("summary"):
            summary = me["summary"]
        final_score = rec.get("final_score") or {}
        won = rec.get("champion") == side
        mp += 1
        if won:
            mw += 1
        rcnt = rwon = 0
        for r in rec.get("rounds") or []:
            rp += 1
            rcnt += 1
            if r.get("winner") == side:
                rw += 1
                rwon += 1
        matches.append({
            "id": rec.get("id", ""),
            "created_at": rec.get("created_at", ""),
            "mode": rec.get("mode", ""),
            "opponent_name": opp.get("name", ""),
            "side": side,
            "won": won,
            "my_score": int(final_score.get(side, 0)),
            "opponent_score": int(final_score.get(opp_side, 0)),
            "rounds": rcnt,
            "rounds_won_in_match": rwon,
        })

    if mp == 0:
        return None
    matches.sort(key=lambda m: m["created_at"], reverse=True)
    return {
        "name": name,
        "source": source,
        "summary": summary,
        "matches_played": mp,
        "matches_won": mw,
        "rounds_played": rp,
        "rounds_won": rw,
        "match_rate": (mw / mp) if mp else 0.0,
        "round_rate": (rw / rp) if rp else 0.0,
        "matches": matches,
    }
