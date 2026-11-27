#!/usr/bin/env bash
# M36: wait for heads A/B -> score 254,672 games (5 nodes x 8 shards) -> R-level screen (continue only if CI lower > 0) -> 24 members
# (pc recipe + --r-extra) -> protocol check -> single-arm compare + 6v6 vs the pc members.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 300); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
H=$(cat $A/m36_heads.jobs | awk '{print $2","$4}')
echo "== $(date '+%F %T') M36 step 1: heads $H"; wait_jobs "$H"
for j in ${H//,/ }; do echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; grep -E "saved head|done in|training games" logs/nano4/m36h*-$j.out | cut -c1-160; done
ok_all "$H" && ls results/r_variants/attn_heads/attn_A.pt results/r_variants/attn_heads/attn_B.pt || { echo "heads failed: stop"; exit 1; }
echo "== $(date '+%F %T') M36 step 2: scoring"
NN=5; NSH=$((NN * 8)); S=""
for n in $(seq 0 $((NN - 1))); do S="$S,$(submit m36sc$n "--gres=gpu:8 -c 96 -t 04:00:00" bash $A/m36_score_node.sh $n $NSH)"; done
echo "score jobs ${S#,}"; wait_jobs "$S"; for j in ${S//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
C=$(submit m36cnt "-c 12 -t 00:20:00" python -I -c "
import glob, numpy as np, pandas as pd
want = set(pd.read_csv('results/r_variants/attn_games.csv', dtype={'game_key': str})['game_key'])
got = set(); [got.update(np.load(f)['keys'].tolist()) for f in glob.glob('results/attn_scores/part_*.npz')]
print(f'scored {len(got & want)} of {len(want)} games; extra {len(got - want)}'); raise SystemExit(0 if want <= got else 1)")
wait_jobs "$C"; grep scored logs/nano4/m36cnt-$C.out; [ "$(state $C)" = "COMPLETED" ] || { echo "scoring incomplete: stop"; exit 1; }
echo "== $(date '+%F %T') M36 step 3: R-level screen"
R=$(submit m36r "-c 12 -t 02:00:00" python -I logs/nano4/r_variants.py --variants opphsl attn20 --seeds 0 1 2 --pairs opphsl:base attn20:opphsl --out results/r_variants/m36_attn20.json)
wait_jobs "$R"; grep -E " vs |^attn20 |^opphsl " logs/nano4/m36r-$R.out | cut -c1-240
python3 -I -c "
import json, sys; p = json.load(open('results/r_variants/m36_attn20.json'))['pairs']['attn20 vs opphsl']
print('screen: attn20 vs opphsl', round(p['diff'], 4), [round(x, 4) for x in p['ci95']]); sys.exit(0 if p['ci95'][0] > 0 else 1)" || { echo "SCREEN: CI lower <= 0 -> stop (pre-registered)"; exit 0; }
echo "== $(date '+%F %T') M36 step 4: 24 members"
rm -f $A/m36_cmds_*
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  echo "$TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1 --r-extra results/attn_scores --save-dir results/models/ct_cv_m36 --out results/trunk_seq/es" >> $A/m36_cmds_all.txt
done; done; done
split -l 8 -d $A/m36_cmds_all.txt $A/m36_cmds_
P=""; for f in $A/m36_cmds_0*; do P="$P,$(submit m36p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; done
echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
K=$(submit m36chk "-c 12 -t 00:20:00" python -I logs/nano4/m36_protocol_check.py); wait_jobs "$K"
grep -v "^==\|^+" logs/nano4/m36chk-$K.out | tail -26
[ "$(state $K)" = "COMPLETED" ] || { echo "protocol check FAILED: comparisons not run"; exit 1; }
echo "== $(date '+%F %T') M36 step 4: compare"
K2=$(submit m36cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C-pc-f0-x40-oh T-pc-f0-x40-oh C-pc-f0-x40-oh-ra T-pc-f0-x40-oh-ra --seeds 0 1 2 --pairs C-pc-f0-x40-oh-ra:C-pc-f0-x40-oh T-pc-f0-x40-oh-ra:T-pc-f0-x40-oh --tag m36_ --out results/trunk_seq/es)
E=$(submit m36ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-pc-f0-x40-oh T-pc-f0-x40-oh --new C-pc-f0-x40-oh-ra T-pc-f0-x40-oh-ra --seeds 0 1 2 --out results/trunk_seq/es/ens_m36.json)
wait_jobs "$K2,$E"
echo "== compare $K2 $(state $K2)"; grep -v "^==\|^+" logs/nano4/m36cmp-$K2.out | tail -8
echo "== ensemble 6v6 $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m36ens-$E.out | tail -8
echo "== $(date '+%F %T') M36 done"
