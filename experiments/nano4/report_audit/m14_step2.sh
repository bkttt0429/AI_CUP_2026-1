#!/usr/bin/env bash
# M14 step 2 (2026-11-05, pre-registered in section 4M): after the opponent-HSL score arrays (ohb, ohb40) and the ohr R caches exist,
# 48 CV members (C/T x seeds 0-2 x folds 1-4 x variants oh / ohr, x40 training data), then compare + ens_compare 6-vs-6 against the
# x40 members (and ohr vs oh). Production models and ct6 are a separate step, after the gate. args: <build ids csv> <cache ids csv>
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=19
AUD=/home/u2507115/.cache/report_audit
TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() {
  local j=""
  while [ -z "$j" ]; do
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 45; done
    j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 90; }
  done
  echo "$j"
}
echo "== $(date '+%F %T') M14 step 2: waiting for builds $1 and caches $2"
wait_jobs "$1,$2"
for j in ${1//,/ } ${2//,/ }; do echo "$j $(state $j)"; done
ok_all "$1,$2" || { echo "a build or cache job did not complete: abort"; exit 1; }
ls -la results/trunk_seq/scores_oh_fold*.npy results/trunk_seq/scores_x40_oh_fold*.npy results/trunk_seq/r_val*_x40_ohr.npz results/trunk_seq/r_all_f0_x40_ohr.npz | cut -c30-
rm -f $AUD/m14_cmds_*.txt
for var in "--opp-hsl" "--opp-hsl-r"; do for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  SD=""; [ "$v$s" = "10" ] && SD="--save-dir results/models/ct_check"  # fold-1 seed-0 weights for the mock consistency check
  echo "$TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --extra-games 40 $var $SD --out results/trunk_seq/es" >> $AUD/m14_cmds_all.txt
done; done; done; done
split -l 8 -d $AUD/m14_cmds_all.txt $AUD/m14_cmds_
ids=""; for f in $AUD/m14_cmds_0*; do ids="$ids,$(submit ohp$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 05:00:00" bash $AUD/pack8.sh $f)"; done
echo "packed jobs ${ids#,}" | tee $AUD/m14_pack_jobs.txt
wait_jobs "$ids"
for j in ${ids//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; grep -h "slot .* done\|FAILED" logs/nano4/ohp*-$j.out | cut -c1-140; done
K=$(submit ohcmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C-x40 T-x40 C-x40-oh T-x40-oh C-x40-ohr T-x40-ohr --seeds 0 1 2 --pairs C-x40-oh:C-x40 T-x40-oh:T-x40 C-x40-ohr:C-x40 T-x40-ohr:T-x40 C-x40-ohr:C-x40-oh T-x40-ohr:T-x40-oh --tag m14_ --out results/trunk_seq/es)
E1=$(submit ohens1 "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-x40 T-x40 --new C-x40-oh T-x40-oh --seeds 0 1 2 --out results/trunk_seq/es/ens_m14_oh.json)
E2=$(submit ohens2 "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-x40 T-x40 --new C-x40-ohr T-x40-ohr --seeds 0 1 2 --out results/trunk_seq/es/ens_m14_ohr.json)
E3=$(submit ohens3 "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C-x40-oh T-x40-oh --new C-x40-ohr T-x40-ohr --seeds 0 1 2 --out results/trunk_seq/es/ens_m14_ohr_vs_oh.json)
wait_jobs "$K,$E1,$E2,$E3"
echo "== M14 compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/ohcmp-$K.out
echo "== M14 ensemble oh vs x40 $E1 $(state $E1)"; grep -v "^==\|^+" logs/nano4/ohens1-$E1.out
echo "== M14 ensemble ohr vs x40 $E2 $(state $E2)"; grep -v "^==\|^+" logs/nano4/ohens2-$E2.out
echo "== M14 ensemble ohr vs oh $E3 $(state $E3)"; grep -v "^==\|^+" logs/nano4/ohens3-$E3.out
echo "== $(date '+%F %T') M14 step 2 done"
