"""Section 4M M25 (2026-11-06; user: 「強網加搜尋」): the strong net of M23 WITH tree search, through the official KataGo v1.18.2
analysis engine (tools/katago_v1182/bin/katago, external binary, sha256 8c39d9ac...; net .bin.gz sha256 9d7a6afe..., the same network
as M23's model.ckpt; both to be disclosed). Same fold-1 class-stratified sample as M23 (60 players per class, 20 games, rng 20261105).
Every position of both sides, maxVisits V (config tools/katago_v1182/analysis_m25.cfg, values for the side to move); per position:
root win probability and score lead after search, whether the played move is the search's best, whether it is in the top 3.
Per game: quality_score.game_quality's point / win-rate loss statistics (t_* own, o_* opponent) from the searched values, plus
t_s_agree, t_s_top3, o_s_agree, o_s_top3. Output: results/ai_search/fold1/shard{i}.csv (read by ai_search_eval.py).
Rules: HD0 japanese komi 6.5, HD >= 1 komi 0.5 (Q13: HD1 讓先 has no komi; half point avoids draws); HD >= 2 handicap stones placed
(aicup.sgf.setup_and_moves gives the actual colours).
  APPIMAGE_EXTRACT_AND_RUN=1 bash scripts/nano4/shards.sh 8 python -I logs/nano4/ai_search_pilot.py --fold 1 --per-class 60 --visits 100 --out results/ai_search
  --benchmark-games 40: only that many games of shard 0, for a throughput check
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.scoring import fold_players, load_games, player_games  # noqa: E402
from aicup.sgf import handicap, setup_and_moves, to_gtp  # noqa: E402
import quality_score as qs  # noqa: E402

KDIR = ROOT / "tools/katago_v1182"
BIN, CFG = KDIR / "bin/katago", KDIR / "analysis_m25.cfg"
APPRUN = KDIR / "bin/squashfs-root/AppRun"   # the AppImage extracted once per job (8 shards extracting into one /tmp dir collide)
MODEL = KDIR / "dl/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
NV = ROOT / ".venv/lib/python3.12/site-packages/nvidia"
GCC_LIB = "/work/envstack/apps/gcc/13.2/lib64"   # the gcc/13.2 module: the release binary needs GLIBCXX_3.4.30 (system libstdc++ stops at 3.4.29)


def query(key, sgf, visits):
    stones, moves = setup_and_moves(sgf); hd = handicap(sgf) or 0
    mv = [[c, "pass" if x is None else to_gtp(x, y)] for c, x, y in moves]
    q = {"id": key, "moves": mv, "initialStones": [["B", to_gtp(x, y)] for x, y in stones], "rules": "japanese", "komi": 6.5 if hd == 0 else 0.5,
         "boardXSize": 19, "boardYSize": 19, "analyzeTurns": list(range(len(mv) + 1)), "maxVisits": visits}
    if stones:
        q["initialPlayer"] = mv[0][0] if mv else "W"
    return q, mv


def game_row(key, info, mv, resp, target, moves_out=None):
    """resp: {turn: response}; position t is before move t (side to move = mv[t][0]); the last position has no move.
    moves_out (M29 --save-moves): a list that receives one record per answered position: t, side (0 B / 1 W), win, lead (side to move),
    best (the played move is the search's best), top3."""
    T = len(mv); tb, vb, tw, vw = [], [], [], []; agree = {"B": [], "W": []}; top3 = {"B": [], "W": []}
    for t in range(T + 1):
        r = resp.get(t)
        if r is None or "rootInfo" not in r:
            continue
        side = r["rootInfo"].get("currentPlayer") or (mv[t][0] if t < T else ("W" if mv and mv[-1][0] == "B" else "B"))
        val = [float(r["rootInfo"]["winrate"]), float(r["rootInfo"]["scoreLead"])]
        (tb if side == "B" else tw).append(t); (vb if side == "B" else vw).append(val)
        b_, t3_ = np.nan, np.nan
        if t < T and r.get("moveInfos"):
            infos = sorted(r["moveInfos"], key=lambda m: m.get("order", 0)); played = mv[t][1]
            b_, t3_ = float(infos[0]["move"] == played), float(played in [m["move"] for m in infos[:3]])
            agree[side].append(b_); top3[side].append(t3_)
        if moves_out is not None:
            moves_out.append((t, 0 if side == "B" else 1, val[0], val[1], b_, t3_))
    row = {"game_key": key, "player_id": info[1], "rank": info[0], "color": target, "turns_answered": len(resp), "turns_expected": T + 1}
    row.update(qs.game_quality(np.array(tb), np.array(vb).reshape(-1, 2), np.array(tw), np.array(vw).reshape(-1, 2), target))
    own, opp = (target, "W" if target == "B" else "B")
    for name, s in (("t", own), ("o", opp)):
        row[f"{name}_s_agree"] = float(np.mean(agree[s])) if agree[s] else np.nan
        row[f"{name}_s_top3"] = float(np.mean(top3[s])) if top3[s] else np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1); ap.add_argument("--per-class", type=int, default=60); ap.add_argument("--out", required=True)
    ap.add_argument("--visits", type=int, default=100); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--benchmark-games", type=int, default=0)
    ap.add_argument("--save-moves", default=None, help="M29: also save every answered position (t, side, win, lead, best, top3) per game to <dir>/fold<f>/shard<i>.npz")
    ap.add_argument("--config", default=str(CFG), help="analysis config (threads / batch size only; the search settings are the M25 ones)")
    a = ap.parse_args(); t0 = time.time()
    all_games, folds = player_games(), pd.read_csv(ROOT / "splits/player_folds.csv")
    players = fold_players(all_games, folds, a.fold, max_games=20); players = players[players["game_keys"].map(len) == 20]
    rng = np.random.default_rng(20261105)   # the M23 sample, same order
    pick = pd.concat([grp.iloc[rng.permutation(len(grp))[:a.per_class]] for _, grp in players.groupby("rank")])
    games = [(p.rank, p.player_id, k) for p in pick.itertuples() for k in p.game_keys][a.shard::a.nshards]
    if a.benchmark_games:
        games = games[:a.benchmark_games]
    info = {g[2]: g for g in games}; data = load_games([g[2] for g in games])
    env = dict(os.environ, APPIMAGE_EXTRACT_AND_RUN="1", LD_LIBRARY_PATH=":".join([GCC_LIB] + [str(NV / d / "lib") for d in ("cuda_runtime", "cublas", "cudnn", "cuda_nvrtc", "nvjitlink")]) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    (ROOT / "logs/nano4/katago_m25").mkdir(parents=True, exist_ok=True)
    exe = APPRUN if APPRUN.exists() else BIN
    if exe == APPRUN:
        env.pop("APPIMAGE_EXTRACT_AND_RUN", None)
    print(f"engine {exe}", flush=True)
    proc = subprocess.Popen([str(exe), "analysis", "-config", a.config, "-model", str(MODEL)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    qs_moves, n_pos = {}, 0
    for _, _, key in games:
        q, mv = query(key, data[key][0], a.visits); qs_moves[key] = mv; n_pos += len(mv) + 1

    def feed():
        try:
            for _, _, key in games:
                q, _ = query(key, data[key][0], a.visits); proc.stdin.write(json.dumps(q) + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            print("katago closed its input (engine failed to start?)", flush=True)

    threading.Thread(target=feed, daemon=True).start()
    threading.Thread(target=lambda: [sys.stderr.write("katago: " + l) for l in proc.stderr], daemon=True).start()
    resp = {key: {} for _, _, key in games}; done, recs, got = set(), [], 0
    mrec, mkeys = [], []
    out_dir = Path(a.out) / f"fold{a.fold}"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(games)} games, {n_pos} positions, {a.visits} visits each ({time.time() - t0:.0f} s)", flush=True)
    for line in proc.stdout:
        r = json.loads(line)
        if "error" in r or "warning" in r:
            print("katago message:", line.strip()[:300], flush=True)
            if "error" in r and r.get("id") in resp:
                done.add(r["id"])
            continue
        key = r["id"]; resp[key][r["turnNumber"]] = r; got += 1
        if len(resp[key]) == len(qs_moves[key]) + 1:
            done.add(key); rk, pid, _ = info[key]
            mo = [] if a.save_moves else None
            recs.append(game_row(key, info[key], qs_moves[key], resp.pop(key), data[key][1], mo))
            if mo is not None:
                mrec.extend((len(mkeys),) + x for x in mo); mkeys.append((key, info[key][1], info[key][0], data[key][1]))
            if len(done) % 100 == 0:
                el = time.time() - t0; print(f"{len(done)}/{len(games)} games, {got} positions, {got / max(el, 1e-9):.0f} positions/s, {got * a.visits / max(el, 1e-9):.0f} visits/s", flush=True)
        if len(done) == len(games):
            break
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.terminate()
    if not recs:
        raise SystemExit(f"katago returned no results (exit {proc.poll()}); see the katago: lines above")
    name = f"bench{a.benchmark_games}_v{a.visits}_{Path(a.config).stem}" if a.benchmark_games else f"shard{a.shard}"
    pd.DataFrame(recs).to_csv(out_dir / f"{name}.csv", index=False)
    if a.save_moves:
        md = Path(a.save_moves) / f"fold{a.fold}"; md.mkdir(parents=True, exist_ok=True)
        M = np.array(mrec, dtype=np.float64).reshape(-1, 7)
        np.savez_compressed(md / f"{name}.npz", game=M[:, 0].astype(np.int32), t=M[:, 1].astype(np.int16), side=M[:, 2].astype(np.int8), win=M[:, 3].astype(np.float32),
                            lead=M[:, 4].astype(np.float32), best=M[:, 5].astype(np.float32), top3=M[:, 6].astype(np.float32))
        pd.DataFrame(mkeys, columns=["game_key", "player_id", "rank", "color"]).to_csv(md / f"{name}_games.csv", index=False)
    el = time.time() - t0
    print(f"saved {len(recs)} games ({got} positions) in {el / 60:.1f} min: {got / max(el, 1e-9):.0f} positions/s, {got * a.visits / max(el, 1e-9):.0f} visits/s", flush=True)


if __name__ == "__main__":
    main()
