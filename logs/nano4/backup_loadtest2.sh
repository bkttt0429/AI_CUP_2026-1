#!/usr/bin/env bash
# Backup load test 2 (2026-11-18): the remaining backup items (release/, results/l1b, the four scores_* folders,
# trunk_seq/smoke, the intermediates of closed experiments) are moved aside, the chain "identifier fit -> paired pool
# comparison -> registered pool test + file build (must reproduce ct121 byte for byte) -> probe decoder -> ledger build ->
# fold tables" is run, and everything is MOVED BACK (also when a step fails). Nothing is deleted.
#   bash logs/nano4/backup_loadtest2.sh [wait]     wait: start only after both seeds of M221 ended, never after 14:44
A=results/_offload2_20261118; M=$A/MOVED.txt
restore() { local p; [ -f $M ] || return 0; tac $M | while read -r p; do [ -e "$A/$p" ] && mv "$A/$p" "$p"; done; echo "== $(date '+%F %T') moved back; left in the aside folder: $(find $A -mindepth 1 ! -name MOVED.txt ! -type d | wc -l) file(s)"; }
main() {
  local p rc=0 m n=0
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  if [ "${1:-}" = wait ]; then
    until [ "$(grep -l 'training ended' logs/nano4/m221s30-791945.out logs/nano4/m221s31-791945.out 2>/dev/null | wc -l)" = 2 ]; do [ "$(date +%H%M)" -ge 1444 ] && { echo "== $(date '+%T') too late for this job: load test not started"; return 4; }; sleep 20; done
    [ "$(date +%H%M)" -ge 1444 ] && { echo "== too late: not started"; return 4; }
  fi
  [ -e $A ] && { echo "$A exists already - stopped"; return 1; }
  mkdir -p $A && : > $M; trap restore EXIT
  { for p in release results/l1b results/scores_hd results/scores_opp_hd results/scores_hd_x40 results/scores_opp_hd_x40 results/trunk_seq/smoke results/m200_tta results/m161r_tta; do echo "$p"; done; ls results/m200/out_*.npz results/m161r/out_*.npz; find results/m161 results/m161q -mindepth 2 -type f -name '*.npz'; } | while read -r p; do
    [ -e "$p" ] && [ ! -L "$p" ] || { echo "skipped (missing or a link): $p"; continue; }; mkdir -p "$A/$(dirname "$p")" && mv "$p" "$A/$p" && echo "$p" >> $M; done
  echo "== $(date '+%F %T') on $(hostname): $(wc -l < $M) item(s) moved aside"
  rm -f submission_rank_ctloadtest.csv results/r_variants/*ctloadtest* results/r_variants/loadtest2_pool.json results/r_variants/m163_pool_loadtest.json results/r_variants/*LT1*
  python -I logs/nano4/m164_ident.py --prior fit --tag LT1:S7t,S11u > /dev/shm/lt2_ident.$$ 2>&1 && echo "== identifier fit ran: $(grep -c 'written' /dev/shm/lt2_ident.$$) 'written' line(s)" || { echo "== m164_ident FAILED without the items"; tail -5 /dev/shm/lt2_ident.$$; rc=1; }
  python -I logs/nano4/m165_pool.py --sets SQ4 SZ3 --json loadtest2_pool.json | grep "minus\|^seeds" || { echo "== m165_pool FAILED without the items"; rc=1; }
  python -I logs/nano4/m163_pool.py --new SZ3 --also SZ1 SZ2 SZ4 SZ5 --control SPALm --name ctloadtest --json m163_pool_loadtest.json | grep "REGISTERED\|written" || { echo "== m163_pool FAILED without the items"; rc=1; }
  m=$(md5sum submission_rank_ctloadtest.csv 2>/dev/null | cut -c1-12)
  if [ "$m" = 8b071e1bb307 ]; then echo "== the rebuilt file equals today's ct121 byte for byte (md5 $m)"; else echo "== the rebuilt file DIFFERS from ct121: '$m'"; rc=1; fi
  python -I logs/nano4/m206_decode.py upload_probe/probe_C2B.csv 0.8417364 --ref submission_rank_ct116.csv --ref-score 0.8394955 | tail -2 || rc=1
  python -I - <<'PY' || rc=1
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led
import ens_compare as ec
z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); prior = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}
qs, cand, obs = led.build({**led.KNOWN_SCORES, **json.load(open(led.LEDGER))}, prior); print(f"ledger build: {len(obs)} uploads, {len(qs)} constrained Public questions")
for v in ec.FOLDS:
    y, n, pid, q = ec.load_fold(v); print(f"fold {v}: {len(y)} questions")
PY
  rm -f submission_rank_ctloadtest.csv results/r_variants/*ctloadtest* results/r_variants/loadtest2_pool.json results/r_variants/m163_pool_loadtest.json results/r_variants/*LT1* /dev/shm/lt2_ident.$$
  echo "== $(date '+%F %T') load test 2 finished with rc=$rc"
  return $rc
}
main "$@"; rc=$?; restore; trap - EXIT; rmdir -p --ignore-fail-on-non-empty $(find $A -type d 2>/dev/null | sort -r) 2>/dev/null; [ -f $M ] && [ "$(find $A -mindepth 1 ! -name MOVED.txt ! -type d 2>/dev/null | wc -l)" = 0 ] && rm -f $M && rmdir $A 2>/dev/null; exit $rc
