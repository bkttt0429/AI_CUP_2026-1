#!/usr/bin/env bash
# Redo the f1 chain (stage 1 hit the old 10-min NCCL timeout); redo f4 too if its stage 1 (639898, old code) fails;
# then xfit on the final scoring jobs. Login node: squeue/sbatch/sacct only.
cd ~/BK_AICUP_2026-1
X='--exclude=25a-hgpn[001,003,004,007,039,062]'; R=scripts/nano4/run.sbatch; L=logs/nano4/pr17_jobs.txt
C='--cache results/l1b/cache_A results/l1b/cache_B results/l1b/cache_0'; G8='-p 8gpus --gres=gpu:8 -c 96'
TR="torchrun --standalone --nproc_per_node 8 scripts/l1b_train.py $C"
sb(){ while [ "$(squeue -h -u $USER | wc -l)" -ge 20 ]; do sleep 60; done; sbatch --parsable $X "$@"; }
chain(){ local k=$1 S1 S2 S3 T O
  rm -f results/l1b/heads5s1_f$k.pt.ckpt
  S1=$(sb $G8 -t 06:00:00 -J x5s1f$k $R $TR --exclude-folds $k --out results/l1b/heads5s1_f$k.pt --batch 32) || return 1
  S2=$(sb --dependency=afterok:$S1 $G8 -t 06:00:00 -J x5s2f$k $R $TR --exclude-folds $k --out results/l1b/heads5s2_f$k.pt --trunk-lr 1e-5 --init-heads results/l1b/heads5s1_f$k.pt --batch 128) || return 1
  S3=$(sb --dependency=afterok:$S2 $G8 -t 06:00:00 -J x5f$k $R $TR --exclude-folds $k --out results/l1b/heads5_f$k.pt --trunk-lr 1e-5 --init-heads results/l1b/heads5s2_f$k.pt --disc-weight 1 --epochs 2 --batch 128) || return 1
  T=$(sb --dependency=afterok:$S3 $G8 -t 02:00:00 -J x5sc$k $R bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f$k.pt --folds $k --out results/l1b/scores5 --workers 10) || return 1
  O=$(sb --dependency=afterok:$S3 $G8 -t 02:00:00 -J x5so$k $R bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f$k.pt --folds $k --out results/l1b/scores5_opp --opponent --workers 10) || return 1
  echo "RETRY f$k: s1=$S1 s2=$S2 s3=$S3 sc=$T so=$O" >> $L; echo ":$T:$O"; }
SC1=$(chain 1) || exit 1
while squeue -h -j 639898 2>/dev/null | grep -q .; do sleep 60; done
st=$(sacct -n -X -j 639898 -o State | awk '{print $1}')
if [ "$st" = COMPLETED ]; then SC4=":639970:639971"; else echo "f4 stage 1 639898 $st -> redo" >> $L; SC4=$(chain 4) || exit 1; fi
SC=":639909:639910:640012:640013:639967:639969$SC1$SC4"
XF=$(sb --dependency=afterok$SC -t 01:00:00 -J xfit $R python -I scripts/xfit_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --xfit-l1b results/l1b/scores5 --xfit-opp results/l1b/scores5_opp --out results/l1b/xfit_check.json) || exit 1
echo "xfit=$XF ($SC)" >> $L; echo FEEDER_DONE
