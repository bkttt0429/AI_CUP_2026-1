#!/usr/bin/env bash
# Submitted once the scheduler accepts jobs again (when_up.sh): M17 eval, duplicate scan, production held-out check, learned n-curve, metadata pilot + eval.
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
AUD=/home/u2507115/.cache/report_audit
sub() { local j; j=$(sbatch --parsable $EX "$@" 2>&1 | tail -1); echo "$j"; }
E=$(sub -J tgteval -c 12 -t 00:30:00 scripts/nano4/run.sbatch python -I logs/nano4/oppview_tgt_eval.py --root results/oppview_tgt/fold1 --fold 1 --out results/r_variants/tgtview_pilot.json)
D=$(sub -J dupscan -c 12 -t 00:40:00 scripts/nano4/run.sbatch python -I logs/nano4/dup_scan.py --out results/r_variants/dup_scan.json)
H=$(sub -J prodheld -c 12 -t 02:00:00 scripts/nano4/run.sbatch python -I logs/nano4/prod_heldout_check.py --out results/r_variants/prod_heldout.json)
N=$(sub -J ncurve -c 12 -t 01:30:00 scripts/nano4/run.sbatch python -I logs/nano4/n_curve_oh.py --out results/r_variants/n_curve_learned.json)
M=$(sub -J metapilot -p 8gpus --gres=gpu:8 -c 96 -t 03:00:00 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I logs/nano4/meta_pilot_score.py --fold 1 --per-class 60 --out results/meta_pilot)
ME=$(sub -J metaeval -c 12 -t 00:30:00 --dependency=afterok:$M scripts/nano4/run.sbatch python -I logs/nano4/meta_pilot_eval.py --root results/meta_pilot/fold1 --fold 1 --out results/r_variants/meta_pilot.json)
B=$(sub -J biaspred -c 12 -t 00:40:00 scripts/nano4/run.sbatch python -I logs/nano4/bias_pred.py --out results/r_variants/bias_pred.json)
R=$(sub -J resproxy -c 12 -t 00:40:00 scripts/nano4/run.sbatch python -I logs/nano4/result_proxy.py --out results/r_variants/result_proxy.json)
S=$(sub -J bandspec -c 12 -t 01:00:00 scripts/nano4/run.sbatch python -I logs/nano4/band_specialist.py --out results/r_variants/band_specialist.json)
echo "tgteval $E dupscan $D prodheld $H ncurve $N metapilot $M metaeval $ME biaspred $B resproxy $R bandspec $S" | tee $AUD/queue_1105_jobs.txt
