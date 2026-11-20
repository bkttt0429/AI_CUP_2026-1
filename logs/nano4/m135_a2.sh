#!/usr/bin/env bash
# M135 part A, stage 2 only (after the recovery jobs): the external set's score array, index, game tables, questions and checks.
#   bash logs/nano4/m135_a2.sh
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,069,155,177]'
TR="python -I scripts/trunk_seq_check.py"; E=/work/u2507115/ext_rank
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
main() {
  if [ -f $E/scores_index_fold9.npz ] && [ -f $E/r_ext_p0.npz ]; then echo "stage 2 is ALREADY DONE: built and checked by job 776928 (20:21-20:29) from the session instance on 25a-lgn02, which took over after the recovery jobs ended; log logs/nano4/m135_a2.log.first. The per-move score files are removed and the members are trained by logs/nano4/m135_b.sh (lock $E/.m135_b.lock): nothing to do here, and do not start part B again."; return 0; fi
  echo "== $(date '+%F %T') M135 stage 2: score array, index, game tables, questions, checks"; df -h /work/u2507115 | tail -n 1; du -sh $E
  B=""; for i in $(seq 1 120); do B=$(sbatch --parsable $EX -p 8gpus,16gpus -c 12 -t 02:00:00 -J m135b scripts/nano4/run.sbatch bash -c "$TR build --ext $E && python -I logs/nano4/m135_smoke_check.py $E" 2>/dev/null) && [[ "$B" =~ ^[0-9]+$ ]] && break; sleep 20; done
  echo "build job $B"; until state "$B" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 20; done
  echo "build job $B $(state $B) $(sacct -n -X -j $B -o Elapsed | head -1)"; grep -v "^==\|^+" logs/nano4/m135b-$B.out | tail -26 | cut -c1-260
  du -sh $E $E/*; df -h /work/u2507115 | tail -n 1
  echo "== $(date '+%F %T') stage 2 done"
}
main "$@"; exit $?
