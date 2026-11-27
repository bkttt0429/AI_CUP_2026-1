#!/usr/bin/env bash
# Rest of PR #17 after feeder4: so f3, sc/so f4, sc/so f1, xfit. Started with setsid: stop with kill -- -<pgid>.
cd ~/BK_AICUP_2026-1
X='--exclude=25a-hgpn[001,003,004,007,039,062]'; R=scripts/nano4/run.sbatch; L=logs/nano4/pr17_jobs.txt; G8='-p 8gpus --gres=gpu:8 -c 96'
sb(){ local id; while true; do while [ "$(squeue -h -u $USER | wc -l)" -ge 20 ]; do sleep 60; done; id=$(sbatch --parsable $X "$@" 2>/dev/null) && { echo $id; return 0; }; sleep 60; done; }
sc(){ sb --dependency=afterok:$2 $G8 -t 02:00:00 -J x5sc$1 $R bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f$1.pt --folds $1 --out results/l1b/scores5 --workers 10; }
so(){ sb --dependency=afterok:$2 $G8 -t 02:00:00 -J x5so$1 $R bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads5_f$1.pt --folds $1 --out results/l1b/scores5_opp --opponent --workers 10; }
O3=$(so 3 639907); echo "sc f3=640045 so f3=$O3 (after s3 639907)" >> $L
T4=$(sc 4 639908); O4=$(so 4 639908); echo "sc f4=$T4 so f4=$O4 (after s3 639908)" >> $L
T1=$(sc 1 640039); O1=$(so 1 640039); echo "sc f1=$T1 so f1=$O1 (after s3 640039)" >> $L
SC=":640040:640041:640042:640043:640045:$O3:$T4:$O4:$T1:$O1"
XF=$(sb --dependency=afterok$SC -t 01:00:00 -J xfit $R python -I scripts/xfit_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --xfit-l1b results/l1b/scores5 --xfit-opp results/l1b/scores5_opp --out results/l1b/xfit_check.json)
echo "xfit=$XF ($SC)" >> $L; echo FEEDER_DONE
