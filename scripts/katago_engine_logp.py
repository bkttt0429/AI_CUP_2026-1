#!/usr/bin/env python
"""Reference log p of a player's moves from the KataGo analysis engine (Human SL model).

One query per target move: the position *before* the move with the full move history,
maxVisits=1, includePolicy=true and the humanSLProfile under test. With the human model
loaded as `-model`, the engine's `policy` is the raw human policy: -1 at illegal points,
legal mass summing to 1. Output layout matches aicup.humansl.target_move_logps.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.sgf import parse_moves, to_gtp  # noqa: E402

KATAGO = ROOT / "scripts" / "katago.sh"
MODEL = ROOT / "models" / "katago" / "b18c384nbt-humanv0.bin.gz"
CFG = ROOT / "tools" / "katago" / "analysis_example.cfg"
# analysis_example.cfg already sets numSearchThreadsPerAnalysisThread, so override that
# key (not numSearchThreads) and disable the random input symmetry.
OVERRIDES = "numAnalysisThreads=1,numSearchThreadsPerAnalysisThread=1,nnRandomize=false"
PASS_POS = 19 * 19


def _gtp(x, y):
    return "pass" if x is None else to_gtp(x, y)


def _query(qid, moves, profile):
    return {
        "id": qid,
        "moves": [[c, _gtp(x, y)] for c, x, y in moves],
        "initialPlayer": "B",
        "rules": "chinese",
        "komi": 7.5,
        "boardXSize": 19,
        "boardYSize": 19,
        "maxVisits": 1,
        "includePolicy": True,
        "overrideSettings": {"humanSLProfile": profile, "ignorePreRootHistory": False},
    }


def engine_target_logps(moves, target_color, profile, katago=KATAGO, model=MODEL, cfg=CFG, timeout=1800,
                        only_t=None):
    targets = {f"q{t}": (t, x, y) for t, (c, x, y) in enumerate(moves)
               if c == target_color and (only_t is None or t in only_t)}
    if not targets:
        return []
    stdin = "".join(json.dumps(_query(qid, moves[:t], profile)) + "\n" for qid, (t, _, _) in targets.items())
    proc = subprocess.run(
        [str(katago), "analysis", "-model", str(model), "-config", str(cfg), "-override-config", OVERRIDES],
        input=stdin, capture_output=True, text=True, timeout=timeout,
    )
    responses = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        o = json.loads(line)
        if "error" in o:
            raise RuntimeError(f"engine error for {o.get('id')}: {o['error']}")
        responses[o["id"]] = o
    missing = sorted(set(targets) - set(responses))
    if missing:
        raise RuntimeError(f"no engine response for {missing}; stderr tail:\n{proc.stderr[-2000:]}")
    out = []
    for qid, (t, x, y) in targets.items():
        p = np.asarray(responses[qid]["policy"], dtype=np.float64)
        legal = p >= 0
        logp = np.full(p.shape, -np.inf)
        logp[legal] = np.log(np.maximum(p[legal], 1e-300))
        pos = PASS_POS if x is None else y * 19 + x
        out.append({"t": t, "gtp": _gtp(x, y), "logp": float(logp[pos]), "policy_logp": logp})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sgf", required=True)
    ap.add_argument("--color", required=True, choices=["B", "W"])
    ap.add_argument("--profile", default="rank_3d")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = engine_target_logps(parse_moves(a.sgf), a.color, a.profile)
    for r in res:
        r["policy_logp"] = r["policy_logp"].tolist()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"profile": a.profile, "color": a.color, "moves": res}, indent=1))
    print(f"wrote {a.out}: {len(res)} target moves, logp {[round(r['logp'], 3) for r in res]}")


if __name__ == "__main__":
    main()
