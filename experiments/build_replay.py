#!/usr/bin/env python
"""Build a self-contained rank-evidence replay page (viz/rank_replay.html + embedded data).

Selections (one of):
  --per-rank K        K players per rank, one random game each (10 x K boards; default 5 -> 50)
  --player PID        all scored games of one player (up to 20)
Evidence comes from the per-move Human SL log p already on disk:
  --source pilot      results/pilot (folds 0 and 1, 200 players each)
  --source scores     results/scores/fold{F} (the full-fold run; finished parts only)
A page holds at most 100 boards (measured lab01 GPU P95: 50 -> 28 ms, 100 -> 54 ms,
200 -> 105 ms); more needs --force or separate windows (scripts/launch_replay.sh --windows).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.replay import (DEFAULT_BOARDS, build_payload, check_board_count, class_bands, load_move_logp,  # noqa: E402
                          select_games)
from aicup.scoring import load_games, read_scored_games  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pilot", "scores"], default="pilot")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--per-rank", type=int, default=DEFAULT_BOARDS // 10)
    ap.add_argument("--player", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="allow more than 100 boards on one page")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.source == "pilot":
        directory = ROOT / "results" / "pilot"
        games = pd.concat([pd.read_csv(p, usecols=["game_key", "player_id", "rank"])
                           for p in sorted(directory.glob(f"games_fold{a.fold}_shard*.csv"))], ignore_index=True)
    else:
        directory = ROOT / "results" / "scores"
        games = read_scored_games(directory / f"fold{a.fold}")[["game_key", "player_id", "rank"]]
    if a.player:
        rows = games[games["player_id"].str.startswith(a.player)].sort_values("game_key")
        if rows["player_id"].nunique() != 1:
            raise SystemExit(f"--player {a.player!r} matches {rows['player_id'].nunique()} players")
        what = f"玩家 {rows['player_id'].iloc[0][:8]}（{rows['rank'].iloc[0]}）的 {len(rows)} 局"
    else:
        rows = select_games(games, a.per_rank, a.seed)
        what = f"每級 {a.per_rank} 位玩家各 1 局"
    check_board_count(len(rows), a.force)

    pilot_json = ROOT / "results" / "pilot_humansl.json"
    bands = class_bands(json.loads(pilot_json.read_text())["curves_fold0"]["best_profile_counts_by_class"]) if pilot_json.exists() else {}
    keys = rows["game_key"].tolist()
    payload = build_payload(rows, load_games(keys), load_move_logp(directory, a.fold, a.source, keys), bands,
                            title=f"棋力證據回放 · fold {a.fold} · {what}", source=f"{a.source} fold {a.fold}")
    missing = sum(g["evidence"] is None for g in payload["games"])
    template = (ROOT / "viz" / "rank_replay.html").read_text(encoding="utf-8")
    html = template.replace("/*PAYLOAD*/", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    out = Path(a.out) if a.out else ROOT / "results" / "replay" / (
        f"replay_{a.source}_fold{a.fold}_" + (f"player_{a.player}" if a.player else f"{len(rows)}boards") + ".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(rows)} boards, {missing} without evidence, {len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
