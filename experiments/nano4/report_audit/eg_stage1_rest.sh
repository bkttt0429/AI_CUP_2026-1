#!/bin/bash
# Section 4I Stage 1, after the smoke passed (job 691804): wait for the fold-2 extraction -> build -> four MLP jobs -> eval.
cd $HOME/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,039,062,155]'
EXTRACT_JOBS="${1:-691867}"
submit() { until j=$(sbatch --parsable "$@" 2>/dev/null); do sleep 120; done; echo "$j"; }
# one id at a time: squeue -j with a list that contains a finished id errors out and looks "empty" while others still run
wait_jobs() { for id in ${1//,/ }; do until [ -z "$(squeue -h -j $id 2>/dev/null)" ]; do sleep 60; done; done; }
wait_jobs $EXTRACT_JOBS
for f in 1 2 3 4; do [ -s results/endgame/games_fold$f.csv ] || { echo "extraction incomplete: fold $f missing"; exit 1; }; done
wc -l results/endgame/games_fold*.csv
j=$(submit $EX -c 12 -t 00:40:00 -J egbuild scripts/nano4/run.sbatch python -I logs/nano4/endgame_stage1.py build)
echo "build job $j"
wait_jobs $j
grep -q "build done" logs/nano4/egbuild-$j.out || { echo "build failed (job $j)"; grep -v -E '^(\+|== 2026)' logs/nano4/egbuild-$j.out | tail -30; exit 1; }
grep -v -E '^(\+|== 2026)' logs/nano4/egbuild-$j.out
ids=""
for v in 1 2 3 4; do
  ids="$ids,$(submit $EX -c 12 -t 01:30:00 -J egmlp$v scripts/nano4/run.sbatch python -I logs/nano4/endgame_stage1.py mlp --val-fold $v)"
done
echo "mlp jobs ${ids#,}"
wait_jobs $ids
echo "mlp: $(sacct -j ${ids#,} -X --format=JobName%8,State -n | tr -s ' ' | tr '\n' ';')"
for v in 1 2 3 4; do [ -s results/endgame/mlp_val$v.npz ] || { echo "mlp fold $v missing"; exit 1; }; done
j=$(submit $EX -c 12 -t 00:30:00 -J egeval scripts/nano4/run.sbatch python -I logs/nano4/endgame_stage1.py eval)
echo "eval job $j"
wait_jobs $j
grep -v -E '^(\+|== 2026)' logs/nano4/egeval-$j.out
