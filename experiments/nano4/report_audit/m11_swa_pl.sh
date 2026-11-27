#!/usr/bin/env bash
# Remaining experiments (2026-11-05, user: 都跑): SWA window averaging (-swa5) and CV pseudo-labelling of the validation questions
# (-pl) for C / T on folds 1-4, seeds 0-2; compare against C / T and 6-vs-6 ensemble checks. Packed 8 per 8-GPU node.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=19
AUD=/home/u2507115/.cache/report_audit
TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
submit() {
  local j=""
  while [ -z "$j" ]; do
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 45; done
    j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 90; }
  done
  echo "$j"
}
echo "== $(date '+%F %T') M11 start (swa5, pseudo)"
T=$(cat $AUD/x80test_job.txt | awk '{print $2}'); wait_jobs "$T"; echo "tests $T $(state $T)"
[ "$(state $T)" = "COMPLETED" ] || { tail -30 logs/nano4/x80test-$T.out; echo "tests failed: abort"; exit 1; }
# soft labels per fold = the 6-member C+T (seeds 0-2) validation probabilities (out-of-fold; no labels used)
S=$(submit softlab "-c 12 -t 00:20:00" python -I -c "
import numpy as np, json
from pathlib import Path
Path('results/pseudo').mkdir(exist_ok=True)
for v in (1,2,3,4):
    P = np.mean([np.load(f'results/trunk_seq/es/val{v}/{a}_s{s}.npz')['proba'] for a in ('C','T') for s in (0,1,2)], axis=0)
    m = json.load(open(f'results/trunk_seq/r_val{v}.json'))['val']
    assert len(m['rank']) == len(P)
    np.savez(f'results/pseudo/val{v}_ct6.npz', proba=P.astype(np.float32), n_games=np.array(m['n_games']))
    print(v, P.shape, 'mean max p', float(P.max(axis=1).mean()))
")
wait_jobs "$S"; echo "soft labels $S $(state $S)"; grep -v "^+" logs/nano4/softlab-$S.out | tail -4
[ "$(state $S)" = "COMPLETED" ] || { echo "soft labels failed: abort"; exit 1; }
rm -f $AUD/m11_cmds_*.txt
for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
  echo "$TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --swa 5 --out results/trunk_seq/es" >> $AUD/m11_cmds_all.txt
  echo "$TR train --arm $arm --val-fold $v --seed $s --inner-frac 0.1 --pseudo results/pseudo/val${v}_ct6.npz --pseudo-weight 1.0 --out results/trunk_seq/es" >> $AUD/m11_cmds_all.txt
done; done; done
split -l 8 -d $AUD/m11_cmds_all.txt $AUD/m11_cmds_
ids=""; for f in $AUD/m11_cmds_0*; do ids="$ids,$(submit m11p$(basename $f | sed 's/.*_//') "--gres=gpu:8 -c 96 -t 04:00:00" bash $AUD/pack8.sh $f)"; done
echo "packed jobs ${ids#,}" | tee $AUD/m11_jobs.txt
wait_jobs "$ids"
for j in ${ids//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; grep -h "slot .* done\|FAILED" logs/nano4/m11p*-$j.out | cut -c1-140; done
K=$(submit m11cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C T C-swa5 T-swa5 C-pl T-pl --seeds 0 1 2 --pairs C-swa5:C T-swa5:T C-pl:C T-pl:T --tag m11_ --out results/trunk_seq/es)
E1=$(submit m11ens1 "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C T --new C-swa5 T-swa5 --seeds 0 1 2 --out results/trunk_seq/es/ens_m11_swa.json)
E2=$(submit m11ens2 "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C T --new C-pl T-pl --seeds 0 1 2 --out results/trunk_seq/es/ens_m11_pl.json)
wait_jobs "$K,$E1,$E2"
echo "== M11 compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/m11cmp-$K.out
echo "== M11 swa 6v6 $E1 $(state $E1)"; grep -v "^==\|^+" logs/nano4/m11ens1-$E1.out
echo "== M11 pseudo 6v6 $E2 $(state $E2)"; grep -v "^==\|^+" logs/nano4/m11ens2-$E2.out
echo "== $(date '+%F %T') M11 done"
