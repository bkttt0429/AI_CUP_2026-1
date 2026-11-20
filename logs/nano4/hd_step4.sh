set -e
python -I scripts/patch_scored.py --base results/scores --patch results/scores_hdpatch --out results/scores_hd --folds 0 1 2 3 4 --summary results/data/patch_scores_hd.json
python -I scripts/patch_scored.py --base results/l1b/scores3 --patch results/l1b/scores3_hdpatch --out results/l1b/scores3_hd --folds 0 3 4 --summary results/data/patch_scores3_hd.json
python -I scripts/patch_scored.py --base results/l1b/scores3_opp --patch results/l1b/scores3_opp_hdpatch --out results/l1b/scores3_opp_hd --folds 0 3 4 --summary results/data/patch_scores3_opp_hd.json
python -I scripts/patch_scored.py --base results/l1b/scores3_B --patch results/l1b/scores3_B_hdpatch --out results/l1b/scores3_B_hd --folds 0 1 2 --summary results/data/patch_scores3_B_hd.json
python -I scripts/patch_scored.py --base results/l1b/scores3_B_opp --patch results/l1b/scores3_B_opp_hdpatch --out results/l1b/scores3_B_opp_hd --folds 0 1 2 --summary results/data/patch_scores3_B_opp_hd.json
