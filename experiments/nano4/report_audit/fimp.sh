#!/usr/bin/env bash
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
J=""; for v in 1 2 3 4; do j=$(sbatch --parsable $EX -c 12 -t 03:00:00 -J fimp$v scripts/nano4/run.sbatch python -I logs/nano4/feature_importance.py --val-fold $v --out results/r_variants/importance_val$v.json); J="$J $j"; done
echo "== $(date '+%F %T') jobs$J"
for j in $J; do until state $j | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 30; done; echo "== $j $(state $j)"; done
python3 -I - <<'PY'
import json, numpy as np
reps = [json.load(open(f"results/r_variants/importance_val{v}.json")) for v in (1, 2, 3, 4)]
print("base scores", [round(r["base"], 4) for r in reps])
rows = []
for kind in ("token_groups", "r_blocks", "r_columns"):
    for k in reps[0][kind]:
        vals = [r[kind][k] for r in reps]; rows.append((float(np.mean(vals)), kind, k, vals))
for kind in ("token_groups", "r_blocks"):
    print(f"== {kind}")
    for mval, kd, k, vals in sorted([r for r in rows if r[1] == kind], reverse=True):
        print(f"  {k:24s} {mval:+.4f}  folds " + " ".join(f"{x:+.4f}" for x in vals))
print("== top 15 single items (token groups + R columns)")
for mval, kd, k, vals in sorted([r for r in rows if r[1] != "r_blocks"], reverse=True)[:15]:
    print(f"  {k:24s} {mval:+.4f}  ({kd})  folds " + " ".join(f"{x:+.4f}" for x in vals))
PY
echo "== $(date '+%F %T') done"
