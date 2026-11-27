#!/usr/bin/env bash
# M25: raise the per-user thread limit to the hard limit; 8 engines x 128 positions (~256 threads each) if the hard limit allows >= 4096,
# else 8 x 96; extract the AppImage once; the 8-GPU pilot at 100 visits; then eval.
set -euo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EXC='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
submit() { for i in $(seq 1 300); do j=$(sbatch --parsable "$@" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
waitj() { until state "$1" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 30; done; state "$1"; }
J2=$(submit $EXC -J m25 --gres=gpu:8 -c 96 -t 06:00:00 scripts/nano4/run.sbatch bash -c 'H=$(ulimit -Hu); echo "nproc soft $(ulimit -u) hard $H"; [ "$H" = unlimited ] && H=1000000; ulimit -u $H || true; echo "nproc now $(ulimit -u)"; C=analysis_m25_t96; [ "$(ulimit -u)" = unlimited ] || [ "$(ulimit -u)" -ge 4096 ] && C=analysis_m25_t128; echo "config $C"; cd tools/katago_v1182/bin && rm -rf squashfs-root && ./katago --appimage-extract > /dev/null && cd ../../.. && bash scripts/nano4/shards.sh 8 python -I logs/nano4/ai_search_pilot.py --fold 1 --per-class 60 --visits 100 --config tools/katago_v1182/$C.cfg --out results/ai_search')
echo "pilot $J2"; S2=$(waitj $J2); echo "pilot $S2"; grep -h "saved" logs/nano4/shard*-$J2.log | head -8
[ "$S2" = COMPLETED ] || exit 1
J3=$(submit $EXC -J m25eval -c 12 -t 00:30:00 scripts/nano4/run.sbatch python -I logs/nano4/ai_search_eval.py --root results/ai_search/fold1 --nosearch results/ai_quality/fold1 --out results/r_variants/ai_search_pilot.json)
echo "eval $J3"; S3=$(waitj $J3); echo "eval $S3"; grep -vE "Warning|warnings.warn|^\+" logs/nano4/m25eval-$J3.out | tail -25
