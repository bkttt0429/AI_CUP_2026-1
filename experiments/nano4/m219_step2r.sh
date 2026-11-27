#!/usr/bin/env bash
# Section 4M M219 step 2 (2026-11-17; registered in the design doc before any reading of m219_prod.py) - as m218_step2.sh, for
# ONE arm of the networks with the window input, both seeds. Run on a node that sees the arm's ten slim checkpoints in <ckpt dir>:
#   per-network output files (m219_prod.py --save-out; no training, the finished checkpoints are read) ->
#   orientation-averaged queries -> var_<var> (prototypes) and var_<var>c (learned centres) ->
#   identifiers SW1 = block + S6at + new + new centres (S7t / S7tc replaced; primary), SW2 = SQ4's sources + new + new centres,
#   SW3 = block + new + new centres -> label-free choice -> recognised-share report -> paired pool comparison against SQ4 on
#   both seed sets. REPORT ONLY (the arm did not pass its gate): the registered pool test and the file build are left out.
#   bash logs/nano4/m219_step2r.sh <arm> <second window 0|1> <ckpt dir> [var=S12t] [name=ct120]
main() {
  local arm="$1" prev="$2" ckd="$3" var="${4:-S12t}" name="${5:-ct120}" c="results/m84_cache" sel=(SW1 SW2 SW3) specs best t also=() s g5
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 M219_ARM="$arm" M219_CK="$ckd"; g5=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}" | cut -d, -f1-5)
  [ "$(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc 0-9)" -ge 2000 ] || { echo "/home has less than 2.0 GB free: step 2 not started (it writes about 0.35 GB there for a while)"; return 3; }
  [ "$(ls "$ckd"/ckpt_*_s3[01].pt 2>/dev/null | wc -l)" = 10 ] || { echo "the ten checkpoints (seeds 30, 31) are not all in $ckd"; return 1; }
  for s in 30 31; do
    CUDA_VISIBLE_DEVICES=$g5 python -I logs/nano4/m219_prod.py --jobs 1:$s 2:$s 3:$s 4:$s all:$s --win-prev "$prev" --out "$arm" --ck-dir "$ckd" --no-periodic --save-out || return 1
  done
  python -I logs/nano4/m219_tta.py --seeds 30,31 --name "$var" || return 1
  [ -f "$c/var_S7t.npz" ] && [ -f "$c/var_S7tc.npz" ] && [ -f "$c/var_S6at.npz" ] || { echo "var_S7t / var_S7tc / var_S6at missing"; return 1; }
  specs=("SW1:S6at,$var,${var}c" "SW2:S6at,S7t,S7tc,$var,${var}c" "SW3:$var,${var}c")
  echo "== identifiers: ${specs[*]}"
  python -I logs/nano4/m164_ident.py --prior fit $(printf -- '--tag %s ' "${specs[@]}") || return 1
  best=$(python -I logs/nano4/m164_select.py "${sel[@]}") || return 1
  echo "== chosen identifier (label-free rule): $best"
  for t in "${sel[@]}"; do [ "$t" != "$best" ] && also+=("$t"); done
  python -I logs/nano4/m120_diag_mass.py SPALm~m15 SQ4~m15 $(printf -- '%s~m15 ' "${sel[@]}") || echo "(the recognition-share report failed; the test goes on)"
  python -I logs/nano4/m165_pool.py --sets SQ4 "$best" "${also[@]}" --json "m219_pool.json" --dump "m219_dump.npz" || echo "(the paired comparison failed; the test goes on)"
  local rc=0; rm -f results/m219/"$arm"/out_*_s3[01].npz; echo "== the per-network output files of this run were removed (regenerable from the checkpoints with --save-out)"
  return $rc
}
main "$@"; exit $?
