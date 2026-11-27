#!/usr/bin/env bash
# Section 4M M8 finish (2026-11-04 20:55): the pipeline driver got duplicated by a session restart and both copies were stopped after
# the caches; 9 of the 48 trainings are queued. This script submits the missing trainings packed 8 per 8-GPU node (user: use the GPUs),
# waits for everything, then runs the pre-registered compares, the variant rule, submission_rank_ct3.csv and the mock check.
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
stops() { for j in ${1//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(grep -h 'stopped at step' logs/nano4/*-$j.out 2>/dev/null | sed 's/.*stopped/stopped/' | cut -c1-90)"; done; }
echo "== $(date '+%F %T') M8 finish start"
queued=$(squeue -u $USER -h -o "%j")
rm -f $AUD/m8_cmds_*.txt; missing=0; waitnames=""
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  f=results/trunk_seq/es/val$v/${arm}-x40_s$s.npz; name=x40$arm$v$s
  if [ -f "$f" ]; then :; elif echo "$queued" | grep -qx "$name"; then waitnames="$waitnames $name"; else echo "$TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --extra-games --out results/trunk_seq/es" >> $AUD/m8_cmds_all.txt; missing=$((missing + 1)); fi
done; done; done
for arm in C T; do for s in 0 1 2 3 4 5; do
  for variant in base x40; do
    if [ "$variant" = x40 ]; then f=results/models/ct/all_${arm}-f0-x40_s$s.pt; name=allf0x$arm$s; flag="--with-fold0 --extra-games"; else f=results/models/ct/all_${arm}-f0_s$s.pt; name=allf0$arm$s; flag="--with-fold0"; fi
    if [ -f "$f" ]; then :; elif echo "$queued" | grep -qx "$name"; then waitnames="$waitnames $name"; else echo "$TR train --arm $arm --val-fold all --seed $s --inner-frac 0.1 $flag --out results/trunk_seq" >> $AUD/m8_cmds_all.txt; missing=$((missing + 1)); fi
  done
done; done
echo "missing $missing trainings -> packed jobs; waiting for queued:$waitnames"
ids=""
if [ "$missing" -gt 0 ]; then
  split -l 8 -d $AUD/m8_cmds_all.txt $AUD/m8_cmds_
  for f in $AUD/m8_cmds_0*; do ids="$ids,$(submit m8pack$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 04:00:00" bash $AUD/pack8.sh $f)"; done
  echo "packed jobs ${ids#,}" | tee $AUD/m8_pack_jobs.txt
fi
qids=$(squeue -u $USER -h -o "%i %j" | awk -v names=" $waitnames " 'index(names, " "$2" ") > 0 {print $1}' | paste -sd, -)
echo "queued ids $qids"
wait_jobs "$qids,$ids"
stops "$qids"
for j in ${ids//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; grep -h "slot .* done\|FAILED" logs/nano4/m8pack*-$j.out | cut -c1-140; done
n_cv=$(ls results/trunk_seq/es/val*/*-x40_s*.npz 2>/dev/null | wc -l); n_pr=$(ls results/models/ct/all_*-f0*_s*.pt 2>/dev/null | wc -l)
echo "members: CV $n_cv / 24, production $n_pr / 24"
[ "$n_cv" -eq 24 ] && [ "$n_pr" -eq 24 ] || { echo "members missing: abort before compare"; exit 1; }
prod=""
source $AUD/m8_tail.sh
