#!/usr/bin/env bash
# Section 4M M9 (2026-11-04): second L1b heads split for ensemble diversity. After the per-fold caches (702594-702597):
# heads4_A on folds 1+3, heads4_B on folds 2+4, each through the heads3 lineage (stage 1 frozen trunk -> stage 2 trunk lr 1e-5 ->
# stage 3 + disc_weight 1.0), seed 4, torchrun on 8 GPUs; then the OOF scoring passes (target + opponent, base 20 games and the
# x40 games) into results/l1b/scores4{,_B}{,_opp}_hd{,_x40}, eight jobs in parallel, and the 4-heads test re-shard. m9_ct.sh follows.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
AUD=/home/u2507115/.cache/report_audit
K=results/data/x40_keys.csv
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit8() {  # submit8 <name> <hours> <command...>: one 8-GPU node
  local j=""
  while [ -z "$j" ]; do j=$(sbatch --parsable $EX --gres=gpu:8 -c 96 -t "$2:00:00" -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 120; }; done
  echo "$j"
}
TR8="torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py --batch 128 --seed 4"
echo "== $(date '+%F %T') M9 heads start; waiting for caches 702594,702595,702596,702597"
wait_jobs "702594,702595,702596,702597"
for j in 702594 702595 702596 702597; do echo "$j $(state $j)"; done
ok_all "702594,702595,702596,702597" || { echo "a cache build failed: abort"; exit 1; }
du -sh results/l1b/cache_f*; ls results/l1b/cache_f1 | head -3
# stage 1 -> 2 -> 3 for each head, the two heads on two nodes
chain() {  # chain <A|B> <fold a> <fold b>
  local h=$1 ca=results/l1b/cache_f$2 cb=results/l1b/cache_f$3
  S1=$(submit8 h4s1$h 4 $TR8 --cache $ca $cb --out results/l1b/heads4s1_$h.pt)
  echo "heads4 $h stage 1 $S1"; wait_jobs "$S1"; echo "stage 1 $h $(state $S1) $(grep -h 'inner\|best' logs/nano4/h4s1$h-$S1.out | tail -1 | cut -c1-160)"
  [ "$(state $S1)" = "COMPLETED" ] || { tail -20 logs/nano4/h4s1$h-$S1.out; return 1; }
  S2=$(submit8 h4s2$h 5 $TR8 --cache $ca $cb --init-heads results/l1b/heads4s1_$h.pt --trunk-lr 1e-5 --epochs 3 --out results/l1b/heads4s2_$h.pt)
  echo "heads4 $h stage 2 $S2"; wait_jobs "$S2"; echo "stage 2 $h $(state $S2) $(grep -h 'inner\|best' logs/nano4/h4s2$h-$S2.out | tail -1 | cut -c1-160)"
  [ "$(state $S2)" = "COMPLETED" ] || { tail -20 logs/nano4/h4s2$h-$S2.out; return 1; }
  S3=$(submit8 h4s3$h 4 $TR8 --cache $ca $cb --init-heads results/l1b/heads4s2_$h.pt --trunk-lr 1e-5 --disc-weight 1.0 --epochs 2 --out results/l1b/heads4_$h.pt)
  echo "heads4 $h stage 3 $S3"; wait_jobs "$S3"; echo "stage 3 $h $(state $S3) $(grep -h 'inner\|best' logs/nano4/h4s3$h-$S3.out | tail -1 | cut -c1-160)"
  [ "$(state $S3)" = "COMPLETED" ] || { tail -20 logs/nano4/h4s3$h-$S3.out; return 1; }
  python3 -c "import json;d=json.load(open('results/l1b/heads4_$h.json'));print('heads4_$h', {k:d.get(k) for k in ('train_folds','best_epoch','inner_nll_init','inner_nll_best','trunk_trained','stopped_early')})"
}
chain A 1 3 > $AUD/m9_chainA.log 2>&1 &
chain B 2 4 > $AUD/m9_chainB.log 2>&1 &
wait
cat $AUD/m9_chainA.log $AUD/m9_chainB.log
[ -f results/l1b/heads4_A.pt ] && [ -f results/l1b/heads4_B.pt ] || { echo "heads4 missing: abort scoring"; exit 1; }
# OOF scoring: A (folds 1, 3) scores 0, 2, 4; B (folds 2, 4) scores 0, 1, 3; target + opponent; base selection and the x40 games --
# eight 8-GPU jobs in parallel (user: 能跑快一點嗎), plus the 4-heads re-shard of the official test as soon as the heads exist.
SC="python -I scripts/l1b_score.py --workers 10"
ids=""
for spec in "A:0 2 4" "B:0 1 3"; do
  h=${spec%%:*}; folds=${spec#*:}; suf=""; [ "$h" = "B" ] && suf="_B"
  ids="$ids,$(submit8 sc4$h 4 bash scripts/nano4/shards.sh 8 $SC --heads results/l1b/heads4_$h.pt --folds $folds --out results/l1b/scores4${suf}_hd)"
  ids="$ids,$(submit8 sc4${h}o 4 bash scripts/nano4/shards.sh 8 $SC --heads results/l1b/heads4_$h.pt --folds $folds --opponent --out results/l1b/scores4${suf}_opp_hd)"
  ids="$ids,$(submit8 sc4${h}x 4 bash scripts/nano4/shards.sh 8 $SC --heads results/l1b/heads4_$h.pt --folds $folds --max-games 40 --game-keys $K --out results/l1b/scores4${suf}_hd_x40)"
  ids="$ids,$(submit8 sc4${h}xo 4 bash scripts/nano4/shards.sh 8 $SC --heads results/l1b/heads4_$h.pt --folds $folds --max-games 40 --game-keys $K --opponent --out results/l1b/scores4${suf}_opp_hd_x40)"
done
R=$(submit8 sub4 3 bash scripts/nano4/shards.sh 8 python -I scripts/predict_submission.py --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv --out submission_rank4.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt results/l1b/heads4_A.pt results/l1b/heads4_B.pt --workers 10)
echo "scoring jobs ${ids#,}; reshard $R" | tee $AUD/m9_scoring_jobs.txt
wait_jobs "$ids,$R"
for j in ${ids//,/ } $R; do [ -z "$j" ] && continue; echo "$j $(state $j)"; done
ok_all "$ids" || { echo "a scoring pass failed: abort (parts resume per game)"; exit 1; }
echo "reshard $R $(state $R)"
du -sh results/l1b/scores4*; df -h /home/u2507115 | tail -1
echo "== $(date '+%F %T') M9 heads + scoring done"
