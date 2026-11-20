#!/usr/bin/env bash
# Section 4M M221 step 2 (2026-11-18; registered in the design doc 01:54, before any reading of m221_prod.py) - as
# m221_step2.sh, for the networks with external strangers AND the window input (arm wg25), two seeds as one source. Needs the five slim
# checkpoints of each seed in <ckpt dir> (default results/m221/wg25_ckpt):
#   per-network output files (m221_prod.py --save-out; no training) -> orientation-averaged queries -> var_<var> (prototypes)
#   and var_<var>c (learned centres) -> identifiers <P>1 = block + new, <P>2 = block + S7t + new, <P>3 = block + S6at + S7t + new,
#   <P>4 = block + S6at + new (S7t replaced); <P>5 with the centres (report only) -> label-free choice among <P>1..<P>4
#   -> recognised-share report -> paired pool comparison against SQ4 on both seed sets -> (with a name) the registered pool test.
#   bash logs/nano4/m221_step2.sh <seeds, e.g. 30,31> <var, e.g. S13u> <tag prefix, e.g. SC> [candidate name, e.g. ct122 | -] [ckpt dir]
main() {
  local seeds="${1:?seeds}" var="${2:?var}" pre="${3:?tag prefix}" name="${4:--}" ckd="${5:-results/m221/wg25_ckpt}" arm=wg25 c="results/m84_cache" s specs best t also=() g5 rc=0
  local sel=("${pre}1" "${pre}2" "${pre}3" "${pre}4") rep=("${pre}5")
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 M221_ARM="$arm" M221_CK="$ckd" M221_TMP="/dev/shm/m221/tta_${arm}_${var}"; g5=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}" | cut -d, -f1-5)
  [ "$(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc 0-9)" -ge 2000 ] || { echo "/home has less than 2.0 GB free: step 2 not started"; return 3; }
  ls results/r_variants | grep -q "^m127_${pre}[1-5]\.npz" && { echo "identifier names ${pre}1..5 exist already - choose another prefix"; return 1; }
  for s in ${seeds//,/ }; do
    [ "$(ls "$ckd"/ckpt_*_s"$s".pt 2>/dev/null | wc -l)" = 5 ] || { echo "the five checkpoints of seed $s are not in $ckd"; return 1; }
    CUDA_VISIBLE_DEVICES=$g5 python -I logs/nano4/m221_prod.py --jobs 1:$s 2:$s 3:$s 4:$s all:$s --ext-share 0.25 --oe 0 --out "$arm" --ck-dir "$ckd" --no-periodic --save-out || return 1
  done
  python -I logs/nano4/m221_tta.py --seeds "$seeds" --name "$var" || return 1
  [ -f "$c/var_S7t.npz" ] && [ -f "$c/var_S7tc.npz" ] && [ -f "$c/var_S6at.npz" ] || { echo "var_S7t / var_S7tc / var_S6at missing"; return 1; }
  specs=("${pre}1:$var" "${pre}2:S7t,$var" "${pre}3:S6at,S7t,$var" "${pre}4:S6at,$var" "${pre}5:S6at,S7t,S7tc,$var,${var}c")
  echo "== identifiers: ${specs[*]}"
  python -I logs/nano4/m164_ident.py --prior fit $(printf -- '--tag %s ' "${specs[@]}") || return 1
  python -I logs/nano4/m164_select.py "${sel[@]}" "${rep[@]}" > /dev/null || true
  best=$(python -I logs/nano4/m164_select.py "${sel[@]}") || return 1
  echo "== chosen identifier (label-free rule, prototype compositions only): $best"
  for t in "${sel[@]}" "${rep[@]}"; do [ "$t" != "$best" ] && also+=("$t"); done
  python -I logs/nano4/m120_diag_mass.py SPALm~m15 SQ4~m15 $(printf -- '%s~m15 ' "${sel[@]}" "${rep[@]}") || echo "(the recognition-share report failed; the test goes on)"
  python -I logs/nano4/m165_pool.py --sets SQ4 "$best" "${also[@]}" --json "m221_${pre}_pool.json" --dump "m221_${pre}_dump.npz" || echo "(the paired comparison failed; the test goes on)"
  if [ "$name" != "-" ]; then python -I logs/nano4/m163_pool.py --new "$best" --also "${also[@]}" --control SPALm --name "$name"; rc=$?; fi
  rm -f results/m221/"$arm"/out_*_s*.npz; echo "== the per-network output files of this run were removed (regenerable from the checkpoints with --save-out)"
  return $rc
}
main "$@"; exit $?
