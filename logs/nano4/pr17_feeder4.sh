#!/usr/bin/env bash
# f1 chain now (stage 1 hit the old NCCL timeout); scoring for f0/f2/f3/f4 after their stage 3; then xfit.
cd ~/BK_AICUP_2026-1
X='--exclude=25a-hgpn[001,003,004,007,039,062]'; R=scripts/nano4/run.sbatch; L=logs/nano4/pr17_jobs.txt
C='--cache results/l1b/cache_A results/l1b/cache_B results/l1b/cache_0'; G8='-p 8gpus --gres=gpu:8 -c 96'
TR="torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py $C"
sb(){ while [ "$(squeue -h -u $USER | wc -l)" -ge 20 ]; do sleep 60; done; sbatch --parsable $X "$@"; }
score(){ local k=$1 s3=$2 T O
  T=$(sb --dependency=afterok:$s3 $G8 -t 02:00:00 -J x5sc$k $R bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f$k.pt --folds $k --out results/l1b/scores5 --workers 10) || return 1
  O=$(sb --dependency=afterok:$s3 $G8 -t 02:00:00 -J x5so$k $R bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f$k.pt --folds $k --out results/l1b/scores5_opp --opponent --workers 10) || return 1
  echo "sc f$k=$T so f$k=$O (after s3 $s3)" >> $L; echo ":$T:$O"; }
rm -f results/l1b/heads5s1_f1.pt.ckpt
S1=$(sb $G8 -t 06:00:00 -J x5s1f1 $R $TR --exclude-folds 1 --out results/l1b/heads5s1_f1.pt --batch 32) || exit 1
S2=$(sb --dependency=afterok:$S1 $G8 -t 06:00:00 -J x5s2f1 $R $TR --exclude-folds 1 --out results/l1b/heads5s2_f1.pt --trunk-lr 1e-5 --init-heads results/l1b/heads5s1_f1.pt --batch 128) || exit 1
S3=$(sb --dependency=afterok:$S2 $G8 -t 06:00:00 -J x5f1 $R $TR --exclude-folds 1 --out results/l1b/heads5_f1.pt --trunk-lr 1e-5 --init-heads results/l1b/heads5s2_f1.pt --disc-weight 1 --epochs 2 --batch 128) || exit 1
echo "RETRY f1: s1=$S1 s2=$S2 s3=$S3" >> $L
SC=""
for ks in 0:639904 2:640011 3:639907 4:639908 1:$S3; do SC="$SC$(score ${ks%%:*} ${ks##*:})" || exit 1; done
XF=$(sb --dependency=afterok$SC -t 01:00:00 -J xfit $R python -I scripts/xfit_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --xfit-l1b results/l1b/scores5 --xfit-opp results/l1b/scores5_opp --out results/l1b/xfit_check.json) || exit 1
echo "xfit=$XF ($SC)" >> $L; echo FEEDER_DONE
