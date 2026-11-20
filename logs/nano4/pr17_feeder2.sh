#!/usr/bin/env bash
# Resubmits the f2 chain (stage 1 failed on the NCCL timeout, fixed in 95d4382) and then xfit, under the 20-job cap.
cd ~/BK_AICUP_2026-1
X='--exclude=25a-hgpn[001,003,004,007,039,062]'; R=scripts/nano4/run.sbatch; L=logs/nano4/pr17_jobs.txt
C='--cache results/l1b/cache_A results/l1b/cache_B results/l1b/cache_0'; G8='-p 8gpus --gres=gpu:8 -c 96'
TR="torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py $C"
sb(){ while [ "$(squeue -h -u $USER | wc -l)" -ge 20 ]; do sleep 60; done; sbatch --parsable $X "$@"; }
k=2
S1=$(sb $G8 -t 06:00:00 -J x5s1f$k $R $TR --exclude-folds $k --out results/l1b/heads5s1_f$k.pt --batch 32) || exit 1; echo "RETRY s1 f$k=$S1" >> $L
S2=$(sb --dependency=afterok:$S1 $G8 -t 06:00:00 -J x5s2f$k $R $TR --exclude-folds $k --out results/l1b/heads5s2_f$k.pt --trunk-lr 1e-5 --init-heads results/l1b/heads5s1_f$k.pt --batch 128) || exit 1; echo "RETRY s2 f$k=$S2" >> $L
S3=$(sb --dependency=afterok:$S2 $G8 -t 06:00:00 -J x5f$k $R $TR --exclude-folds $k --out results/l1b/heads5_f$k.pt --trunk-lr 1e-5 --init-heads results/l1b/heads5s2_f$k.pt --disc-weight 1 --epochs 2 --batch 128) || exit 1; echo "RETRY s3 f$k=$S3" >> $L
T=$(sb --dependency=afterok:$S3 $G8 -t 02:00:00 -J x5sc$k $R bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f$k.pt --folds $k --out results/l1b/scores5 --workers 10) || exit 1
O=$(sb --dependency=afterok:$S3 $G8 -t 02:00:00 -J x5so$k $R bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f$k.pt --folds $k --out results/l1b/scores5_opp --opponent --workers 10) || exit 1
echo "RETRY sc f$k=$T so f$k=$O" >> $L
SC=":639909:639910:639911:639912:639967:639969:639970:639971:$T:$O"
XF=$(sb --dependency=afterok$SC -t 01:00:00 -J xfit $R python -I scripts/xfit_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --xfit-l1b results/l1b/scores5 --xfit-opp results/l1b/scores5_opp --out results/l1b/xfit_check.json) || exit 1
echo "xfit=$XF" >> $L; echo FEEDER_DONE
