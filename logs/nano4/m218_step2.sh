#!/usr/bin/env bash
# Section 4M M218 step 2 (2026-11-17; registered in the design doc before any reading of m218_prod.py) - as m216_step2.sh /
# m200_step2.sh, for ONE arm of the networks trained with external accounts as strangers. Run on the node that holds the
# arm's checkpoints in /dev/shm/m218/<arm> (or after they were copied to /work/u2507115/m218_ckpt/<arm>):
#   the per-network output files (m218_prod.py --save-out; no training, the finished checkpoints are read) ->
#   orientation-averaged queries -> var_<var> (prototypes) and var_<var>c (learned centres) ->
#   identifier compositions SY1..SY3 (prototype sources only; the selection list) and SY4..SY5 (with the centres; report only)
#   -> label-free choice -> recognised-share report -> paired pool comparison against SQ4 on both seed sets -> registered pool test.
#   bash logs/nano4/m218_step2.sh <arm> <ext share> <outlier weight> <ckpt dir> [seed=30] [var=S11t] [name=ct119]
main() {
  local arm="$1" share="$2" oe="$3" ckd="$4" seed="${5:-30}" var="${6:-S11t}" name="${7:-ct119}" c="results/m84_cache" sel=(SY1 SY2 SY3) rep=(SY4 SY5) specs best t also=()
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 M218_ARM="$arm" M218_CK="$ckd"; local g5; g5=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}" | cut -d, -f1-5)
  [ "$(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc 0-9)" -ge 2000 ] || { echo "/home has less than 2.0 GB free: step 2 not started (it writes about 0.35 GB there for a while)"; return 3; }
  ls "$ckd"/ckpt_*_s"$seed".pt > /dev/null || { echo "no checkpoints in $ckd"; return 1; }
  CUDA_VISIBLE_DEVICES=$g5 python -I logs/nano4/m218_prod.py --jobs 1:$seed 2:$seed 3:$seed 4:$seed all:$seed --ext-share "$share" --oe "$oe" --out "$arm" --ck-dir "$ckd" --no-periodic --save-out || return 1
  python -I logs/nano4/m218_tta.py --seeds "$seed" --name "$var" || return 1
  [ -f "$c/var_S7t.npz" ] && [ -f "$c/var_S6at.npz" ] || { echo "var_S7t / var_S6at missing"; return 1; }
  specs=("SY1:$var" "SY2:S7t,$var" "SY3:S6at,S7t,$var" "SY4:S7t,S7tc,$var,${var}c" "SY5:S6at,S7t,S7tc,$var,${var}c")
  echo "== identifiers: ${specs[*]}"
  python -I logs/nano4/m164_ident.py --prior fit $(printf -- '--tag %s ' "${specs[@]}") || return 1
  python -I logs/nano4/m164_select.py "${sel[@]}" "${rep[@]}" > /dev/null || true
  best=$(python -I logs/nano4/m164_select.py "${sel[@]}") || return 1
  echo "== chosen identifier (label-free rule, prototype compositions only): $best"
  for t in "${sel[@]}" "${rep[@]}"; do [ "$t" != "$best" ] && also+=("$t"); done
  python -I logs/nano4/m120_diag_mass.py SPALm~m15 SQ4~m15 $(printf -- '%s~m15 ' "${sel[@]}" "${rep[@]}") || echo "(the recognition-share report failed; the test goes on)"
  python -I logs/nano4/m165_pool.py --sets SQ4 "$best" "${also[@]}" --json "m218_pool.json" --dump "m218_dump.npz" || echo "(the paired comparison failed; the test goes on)"
  python -I logs/nano4/m163_pool.py --new "$best" --also "${also[@]}" --control SPALm --name "$name"; local rc=$?
  rm -f results/m218/"$arm"/out_*_s"$seed".npz; echo "== the per-network output files of this run were removed (regenerable from the checkpoints with --save-out)"
  return $rc
}
main "$@"; exit $?
