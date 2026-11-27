#!/usr/bin/env bash
# PR #17 chain (stage 1 on 8 GPUs, --batch 32 = global 256): submits job by job under the per-user 20-job
# submit cap (login node: squeue/sbatch only). Job ids go to logs/nano4/pr17_jobs.txt.
cd ~/BK_AICUP_2026-1
X='--exclude=25a-hgpn[001,003,004,007,039,062]'; R=scripts/nano4/run.sbatch; L=logs/nano4/pr17_jobs.txt
C='--cache results/l1b/cache_A results/l1b/cache_B results/l1b/cache_0'; G8='-p 8gpus --gres=gpu:8 -c 96'
TR="torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py $C"
sb(){ while [ "$(squeue -h -u $USER | wc -l)" -ge 20 ]; do sleep 60; done; sbatch --parsable $X "$@"; }
declare -A S1 S2 S3; SC=""
for k in 0 1 2 3 4; do S1[$k]=$(sb $G8 -t 06:00:00 -J x5s1f$k $R $TR --exclude-folds $k --out results/l1b/heads5s1_f$k.pt --batch 32) || exit 1; echo "s1 f$k=${S1[$k]}" >> $L; done
for k in 0 1 2 3 4; do S2[$k]=$(sb --dependency=afterok:${S1[$k]} $G8 -t 06:00:00 -J x5s2f$k $R $TR --exclude-folds $k --out results/l1b/heads5s2_f$k.pt --trunk-lr 1e-5 --init-heads results/l1b/heads5s1_f$k.pt --batch 128) || exit 1; echo "s2 f$k=${S2[$k]}" >> $L; done
for k in 0 1 2 3 4; do S3[$k]=$(sb --dependency=afterok:${S2[$k]} $G8 -t 06:00:00 -J x5f$k $R $TR --exclude-folds $k --out results/l1b/heads5_f$k.pt --trunk-lr 1e-5 --init-heads results/l1b/heads5s2_f$k.pt --disc-weight 1 --epochs 2 --batch 128) || exit 1; echo "s3 f$k=${S3[$k]}" >> $L; done
for k in 0 1 2 3 4; do
  T=$(sb --dependency=afterok:${S3[$k]} $G8 -t 02:00:00 -J x5sc$k $R bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f$k.pt --folds $k --out results/l1b/scores5 --workers 10) || exit 1
  O=$(sb --dependency=afterok:${S3[$k]} $G8 -t 02:00:00 -J x5so$k $R bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f$k.pt --folds $k --out results/l1b/scores5_opp --opponent --workers 10) || exit 1
  echo "sc f$k=$T so f$k=$O" >> $L; SC="$SC:$T:$O"
done
XF=$(sb --dependency=afterok$SC -t 01:00:00 -J xfit $R python -I scripts/xfit_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --xfit-l1b results/l1b/scores5 --xfit-opp results/l1b/scores5_opp --out results/l1b/xfit_check.json) || exit 1
echo "xfit=$XF" >> $L; echo FEEDER_DONE
