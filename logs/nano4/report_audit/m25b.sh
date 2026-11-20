#!/usr/bin/env bash
# M25 after tuning: pick the fastest analysis config from the 1-GPU tuning run (712082), then the 8-GPU pilot (100 visits) + eval.
set -euo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EXC='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
submit() { for i in $(seq 1 300); do j=$(sbatch --parsable "$@" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
waitj() { until state "$1" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 30; done; state "$1"; }
T=712082; echo "tune $(waitj $T)"; grep -E "saved|H200|katago: KataGo" logs/nano4/m25tune-$T.out
best=analysis_m25; bestv=7859   # the first test (64 threads, batch 128): 7,859 visits/s
for c in analysis_m25_t128 analysis_m25_t256; do
  v=$(python3 - "$c" <<'PY'
import glob, re, sys
c = sys.argv[1]; s = open(sorted(glob.glob("logs/nano4/m25tune-*.out"))[-1]).read().split("--config tools/katago_v1182/")
for part in s:
    if part.startswith(c + ".cfg"):
        m = re.findall(r"saved \d+ games .*? (\d+) visits/s", part); print(m[-1] if m else 0); break
else:
    print(0)
PY
)
  echo "$c: $v visits/s"; if [ "${v:-0}" -gt "$bestv" ]; then best=$c; bestv=$v; fi
done
echo "chosen config $best ($bestv visits/s on 1 GPU)"
J2=$(submit $EXC -J m25 --gres=gpu:8 -c 96 -t 06:00:00 scripts/nano4/run.sbatch env APPIMAGE_EXTRACT_AND_RUN=1 bash scripts/nano4/shards.sh 8 python -I logs/nano4/ai_search_pilot.py --fold 1 --per-class 60 --visits 100 --config tools/katago_v1182/$best.cfg --out results/ai_search)
echo "pilot $J2"; S2=$(waitj $J2); echo "pilot $S2"; grep -h "saved" logs/nano4/shard*-$J2.log | head -8
J3=$(submit $EXC -J m25eval -c 12 -t 00:30:00 scripts/nano4/run.sbatch python -I logs/nano4/ai_search_eval.py --root results/ai_search/fold1 --nosearch results/ai_quality/fold1 --out results/r_variants/ai_search_pilot.json)
echo "eval $J3"; S3=$(waitj $J3); echo "eval $S3"; grep -vE "Warning|warnings.warn|^\+" logs/nano4/m25eval-$J3.out | tail -25
