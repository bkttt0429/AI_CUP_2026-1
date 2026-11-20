#!/bin/bash
# Section 4I Stage 1 driver: smoke on fold 1 -> wait for the extraction jobs -> build O columns -> four MLP jobs -> eval.
cd $HOME/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,039,062,155]'
EXTRACT_JOBS="${1:-690895,691770,691771}"
submit() { until j=$(sbatch --parsable "$@" 2>/dev/null); do sleep 120; done; echo "$j"; }
# one id at a time: squeue -j with a list that contains a finished id errors out and looks "empty" while others still run
wait_jobs() { for id in ${1//,/ }; do until [ -z "$(squeue -h -j $id 2>/dev/null)" ]; do sleep 60; done; done; }
j=$(submit $EX -c 12 -t 00:30:00 -J egsmoke scripts/nano4/run.sbatch python -I logs/nano4/endgame_stage1.py smoke)
echo "smoke job $j"
wait_jobs $j
grep -q "SMOKE OK" logs/nano4/egsmoke-$j.out || { echo "smoke failed (job $j)"; grep -v -E '^(\+|== 2026)' logs/nano4/egsmoke-$j.out | tail -30; exit 1; }
grep -v -E '^(\+|== 2026)' logs/nano4/egsmoke-$j.out | grep -E "quality|smoke"
echo "smoke ok ($j); waiting for extraction jobs $EXTRACT_JOBS"
wait_jobs $EXTRACT_JOBS
ls results/endgame/games_fold1.csv results/endgame/games_fold2.csv results/endgame/games_fold3.csv results/endgame/games_fold4.csv || { echo "extraction incomplete"; exit 1; }
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
j=$(submit $EX -c 12 -t 00:30:00 -J egeval scripts/nano4/run.sbatch python -I logs/nano4/endgame_stage1.py eval)
echo "eval job $j"
wait_jobs $j
grep -v -E '^(\+|== 2026)' logs/nano4/egeval-$j.out
