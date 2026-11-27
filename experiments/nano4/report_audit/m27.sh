#!/usr/bin/env bash
# M27: full-system Chinese vs Japanese rules for the Human SL inputs. Stage 1: 7 x 8-GPU scoring jobs (56 shards) under AICUP_RULES=jp;
# stage 2: score arrays + R caches under AICUP_HSL=jp; stage 3: 24 members (3 x 8-GPU packs); stage 4: compare + 6v6 ensemble vs ct7's members.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 300); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
echo "== $(date '+%F %T') M27 stage 1: scoring"
S=""; for k in 1 2 3 4 5 6 7; do S="$S,$(submit m27s$k "--gres=gpu:8 -c 96 -t 04:00:00" bash $A/pack8.sh $A/m27_score_$k.txt)"; done
echo "scoring jobs ${S#,}"; wait_jobs "$S"
for j in ${S//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
bad=0; for d in scores_hd_jp:16 scores_opp_hd_jp:16 scores_hd_x40_jp:12 scores_opp_hd_x40_jp:12; do n=${d#*:}; dir=results/${d%:*}
  for f in 0 1 2 3 4; do c=$(ls $dir/fold$f/done_s* 2>/dev/null | wc -l); [ "$c" = "$n" ] || { echo "INCOMPLETE $dir/fold$f: $c of $n shards"; bad=1; }; done; done
[ $bad = 0 ] || { echo "scoring incomplete: stop"; exit 1; }
echo "== $(date '+%F %T') M27 stage 2: score arrays + R caches"
B="$(submit m27b0 "-c 12 -t 02:00:00" env AICUP_HSL=jp $TR build --opp-hsl --build-folds 0 1 2 3 4 --cache results/trunk_seq)"
B="$B,$(submit m27b40 "-c 12 -t 02:00:00" env AICUP_HSL=jp $TR build --opp-hsl --vec-dir results/trunk_vec_x40 --extra-games 40 --build-folds 0 1 2 3 4 --cache results/trunk_seq)"
for v in 1 2 3 4; do B="$B,$(submit m27r$v "-c 12 -t 02:00:00" env AICUP_HSL=jp $TR rfeat --val-fold $v --with-fold0 --extra-games 40)"; done
echo "build jobs $B"; wait_jobs "$B"; ok_all "$B" || { echo "a build failed: stop"; for j in ${B//,/ }; do echo "$j $(state $j)"; done; exit 1; }
ls -la results/trunk_seq/scores*_oh_jp_fold*.npy results/trunk_seq/r_val*_f0_x40_jp.npz | awk '{print $5, $9}'
echo "== $(date '+%F %T') M27 stage 3: 24 members"
rm -f $A/m27_cmds_*
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  echo "AICUP_HSL=jp $TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --save-dir results/models/ct_cv_jp --out results/trunk_seq/es" >> $A/m27_cmds_all.txt
done; done; done
split -l 8 -d $A/m27_cmds_all.txt $A/m27_cmds_
P=""; for f in $A/m27_cmds_0*; do P="$P,$(submit m27p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $A/pack8.sh $f)"; done
echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; grep -h "slot .* done\|FAILED" logs/nano4/m27p*-$j.out | cut -c1-140; done
ok_all "$P" || echo "a pack failed (continuing to compare what exists)"
echo "== $(date '+%F %T') M27 stage 4: compare"
K=$(submit m27cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C-f0-x40-oh T-f0-x40-oh C-f0-x40-oh-jp T-f0-x40-oh-jp --seeds 0 1 2 --pairs C-f0-x40-oh-jp:C-f0-x40-oh T-f0-x40-oh-jp:T-f0-x40-oh --tag m27_ --out results/trunk_seq/es)
E=$(submit m27ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-f0-x40-oh T-f0-x40-oh --new C-f0-x40-oh-jp T-f0-x40-oh-jp --seeds 0 1 2 --out results/trunk_seq/es/ens_m27.json)
wait_jobs "$K,$E"
echo "== compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/m27cmp-$K.out | tail -20
echo "== ensemble 6v6 $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m27ens-$E.out | tail -20
echo "== $(date '+%F %T') M27 done"
