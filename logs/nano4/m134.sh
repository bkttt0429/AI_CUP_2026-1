#!/usr/bin/env bash
# M134: design changes of the base members (game-level auxiliary loss, additive per-game log-likelihood channel, wider blocks),
# seeds 0-2: smoke, 72 members in nine 8-GPU packs (alternating 8gpus / 16gpus), then the comparisons against the current members.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,069,155,177]'
A=/home/u2507115/.cache/report_audit; TR="python -I scripts/trunk_seq_check.py"
REC="--inner-frac 0.1 --with-fold0 --extra-games 40 --opp-hsl --player-contrast 0.1"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 45; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() { local j=""; for i in $(seq 1 240); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 45; done; return 1; }
main() {
  echo "== $(date '+%F %T') M134 stage 0: smoke"
  mkdir -p $A/m134smoke
  SM="--limit 2000 --epochs 1 --evals-per-epoch 2 --out $A/m134smoke"
  S=$(submit m134smk "-c 12 -t 00:40:00" bash -c "$TR train --arm C --val-fold 1 --seed 0 $REC --game-aux 0.3 $SM && $TR train --arm T --val-fold 1 --seed 0 $REC --pooling sumll --game-aux 0.3 $SM && $TR train --arm C --val-fold 1 --seed 0 $REC --width-mult 2 --dropout 0.3 $SM")
  echo "smoke job $S"; wait_jobs "$S"; echo "smoke $(state $S)"; grep -E "stopped at step|Error|error|Traceback" logs/nano4/m134smk-$S.out | tail -6 | cut -c1-200; ls $A/m134smoke/val1 2>/dev/null | tr '\n' ' '; echo
  ok_all "$S" || { echo "smoke failed: stop"; return 1; }
  echo "== $(date '+%F %T') M134 stage 1: 72 members"
  rm -f $A/m134_cmds_*
  for aug in "--game-aux 0.3" "--pooling sumll --game-aux 0.3" "--width-mult 2 --dropout 0.3"; do for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do
    echo "$TR train --arm $arm --val-fold $v --seed $s $REC $aug --save-dir results/models/ct_cv_m134 --out results/trunk_seq/es" >> $A/m134_cmds_all.txt
  done; done; done; done
  split -l 8 -d $A/m134_cmds_all.txt $A/m134_cmds_
  P=""; k=0
  for f in $A/m134_cmds_0*; do
    if [ $((k % 3)) -eq 2 ]; then part="-p 16gpus"; else part="-p 8gpus"; fi
    P="$P,$(submit m134p$k "$part --gres=gpu:8 -c 96 -t 02:00:00" bash $A/pack8.sh $f)"; k=$((k + 1))
  done
  echo "pack jobs ${P#,}"; wait_jobs "$P"; for j in ${P//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(sacct -n -X -j $j -o Elapsed | head -1)"; done
  ok_all "$P" || echo "a pack failed: the comparisons below show what is missing"
  echo "== $(date '+%F %T') M134 stage 2: comparisons"
  J=""
  for t in "pc-f0-x40-oh-ga30" "sumll-pc-f0-x40-oh-ga30" "pc-f0-x40-oh-w2-do30"; do
    n=$(echo $t | sed 's/pc-f0-x40-oh//; s/^-//; s/-$//; s/--/-/')
    J="$J,$(submit m134c "-c 12 -t 01:00:00" bash -c "$TR compare --val-folds 1 2 3 4 --arms C-pc-f0-x40-oh T-pc-f0-x40-oh C-$t T-$t --seeds 0 1 2 --pairs C-$t:C-pc-f0-x40-oh T-$t:T-pc-f0-x40-oh --tag m134_${n}_ --out results/trunk_seq/es; python -I logs/nano4/ens_compare.py --old C-pc-f0-x40-oh T-pc-f0-x40-oh --new C-$t T-$t --seeds 0 1 2 --out results/trunk_seq/es/ens_m134_$n.json; python -I logs/nano4/ens_compare.py --old C-pc-f0-x40-oh T-pc-f0-x40-oh --new C-pc-f0-x40-oh T-pc-f0-x40-oh C-$t T-$t --seeds 0 1 2 --out results/trunk_seq/es/ens_m134_${n}_added.json")"
  done
  wait_jobs "$J"
  for j in ${J//,/ }; do [ -z "$j" ] && continue; echo "== comparison job $j $(state $j)"; grep -v "^==\|^+" logs/nano4/m134c-$j.out | tail -22 | cut -c1-260; done
  echo "== $(date '+%F %T') M134 done"
}
main "$@"; exit $?
