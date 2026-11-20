#!/usr/bin/env bash
# Backup load test (2026-11-18): the 48 big base-model arrays of results/trunk_seq (scores_*.npy, r_*.npz; the user holds a
# copy) are moved aside, their sha256 is checked against the list given to the user, and the candidate-building chain is rerun:
#   paired pool comparison (m165_pool.py), the registered pool test + file build (m163_pool.py, must reproduce tonight's ct119
#   byte for byte), the probe decoder, the ledger build and the fold tables.
# The files STAY in the aside folder afterwards (deleted or moved back by a separate, explicit step).
#   bash logs/nano4/backup_loadtest.sh
main() {
  local T=results/trunk_seq A=results/trunk_seq/_offload_20261118 n rc=0 m want
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  mkdir -p $A || return 1
  n=$(ls $T/scores_*.npy $T/r_*.npz 2>/dev/null | wc -l); echo "== $(date '+%F %T') on $(hostname): $n of the 48 files are in $T, $(ls $A | wc -l) in the aside folder"
  if [ "$n" != 0 ]; then mv $T/scores_*.npy $T/r_*.npz $A/ || return 1; fi
  [ "$(ls $A | wc -l)" = 48 ] || { echo "the aside folder does not hold exactly 48 files - stopped"; return 1; }
  if ( cd $A && sha256sum --quiet -c ../../../reports/backup_20261118/sha256_trunk_seq_big.txt ); then echo "== sha256: all 48 files match reports/backup_20261118/sha256_trunk_seq_big.txt"; else echo "== sha256 MISMATCH against the list"; rc=1; fi
  rm -f submission_rank_ctloadtest.csv results/r_variants/*ctloadtest* results/r_variants/loadtest_pool.json results/r_variants/m163_pool_loadtest.json
  python -I logs/nano4/m165_pool.py --sets SQ4 SY3 --json loadtest_pool.json || { echo "== m165_pool FAILED without the files"; rc=1; }
  python -I logs/nano4/m163_pool.py --new SY3 --also SY1 SY2 SY4 SY5 --control SPALm --name ctloadtest --json m163_pool_loadtest.json || { echo "== m163_pool FAILED without the files"; rc=1; }
  m=$(md5sum submission_rank_ctloadtest.csv 2>/dev/null | cut -c1-12); want=$(md5sum submission_rank_ct119.csv | cut -c1-12)
  if [ "$m" = "$want" ]; then echo "== the rebuilt candidate file equals tonight's ct119 byte for byte (md5 $m)"; else echo "== the rebuilt candidate file DIFFERS: $m against $want"; rc=1; fi
  python -I logs/nano4/m206_decode.py upload_probe/probe_C2B.csv 0.8417364 --ref submission_rank_ct116.csv --ref-score 0.8394955 || rc=1
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
    y, n, pid, q = ec.load_fold(v); P = np.load(ec.ES / f"val{v}/{ec.STD[0] if hasattr(ec, 'STD') else 'C-pc-f0-x40-oh'}_s0.npz")["proba"]; print(f"fold {v}: {len(y)} questions, base posterior {P.shape}")
PY
  rm -f submission_rank_ctloadtest.csv results/r_variants/*ctloadtest* results/r_variants/loadtest_pool.json results/r_variants/m163_pool_loadtest.json
  echo "== $(date '+%F %T') load test finished with rc=$rc; the 48 files remain in $A"
  return $rc
}
main "$@"; exit $?
