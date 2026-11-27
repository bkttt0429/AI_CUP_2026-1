K=$(submit m8cmp "-c 12 -t 01:00:00" $TR compare --val-folds 1 2 3 4 --arms C T C-x40 T-x40 --seeds 0 1 2 --pairs C-x40:C T-x40:T --tag m8_ --out results/trunk_seq/es)
E=$(submit m8ens "-c 12 -t 01:00:00" python -I logs/nano4/ens_compare.py --old C T --new C-x40 T-x40 --seeds 0 1 2 --out results/trunk_seq/es/ens_m8.json)
wait_jobs "$K,$E"
echo "== M8 compare $K $(state $K)"; grep -v "^==\|^+" logs/nano4/m8cmp-$K.out
echo "== M8 ensemble $E $(state $E)"; grep -v "^==\|^+" logs/nano4/m8ens-$E.out
VAR=$(python3 - <<'PY'
import json
d = json.load(open("results/trunk_seq/es/ens_m8.json"))["pairs"]["6_new vs 6_old"]
print("x40" if d["diff"] > 0 and d["ci95"][0] > -0.002 else "nox40")
PY
)
echo "== production variant by the pre-registered rule: $VAR"
wait_jobs "$prod"; stops "$prod"
if [ "$VAR" = "x40" ]; then M="results/models/ct/all_C-f0-x40_s*.pt results/models/ct/all_T-f0-x40_s*.pt"; else M="results/models/ct/all_C-f0_s*.pt results/models/ct/all_T-f0_s*.pt"; fi
echo "models: $(ls $M | wc -l)"
P=$(submit ctpred3 "-t 01:00:00" python -I scripts/trunk_predict.py predict --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv --shards submission_rank.csv --nshards 8 --vec results/ct_submission/vec --models $M --fallback-submission submission_rank_ct.csv --out submission_rank_ct3.csv)
IN="data/mock_tests/fold1_100/rank_prediction_test_public.csv data/mock_tests/fold1_100/rank_prediction_test_private.csv"
OUT=results/mock_fold1_100
Q=$(submit mockct3 "-t 01:00:00" bash -c "python -I scripts/trunk_predict.py predict --inputs $IN --expect-rows 100 --expect-public 50 --expect-private 50 --shards $OUT/submission_rank.csv --nshards 4 --vec $OUT/vec --models $M --l1b-names heads3_B --fallback-submission $OUT/submission_rank.csv --out $OUT/submission_rank_ct3.csv && python -I scripts/trunk_predict.py offline --mock-dir data/mock_tests/fold1_100 --fold 1 --models $M --out $OUT/offline_ct3_proba.csv && python -I scripts/trunk_predict.py compare --a $OUT/submission_rank_ct3_proba.csv --b $OUT/offline_ct3_proba.csv --answers data/mock_tests/fold1_100/answers.csv --report $OUT/ct3_check.json")
wait_jobs "$P,$Q"
echo "== ctpred3 $P $(state $P)"; grep -v "^+" logs/nano4/ctpred3-$P.out | tail -4 | cut -c1-700
echo "== mock ct3 $Q $(state $Q)"; grep -v "^+" logs/nano4/mockct3-$Q.out | grep -i "wrote\|max\|decision\|score\|differ" | cut -c1-400
md5sum submission_rank_ct3.csv
echo "== $(date '+%F %T') M8 pipeline done"
