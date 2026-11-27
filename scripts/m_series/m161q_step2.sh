#!/usr/bin/env bash
# Section 4M M161q step 2 (2026-11-15; design doc M161q, registered before any anchor / pool result of this set):
#   orientation-averaged queries of the second production set -> var_<var> (prototypes) and var_<var>c (learned centres);
#   the identifier compositions SQ1..SQ8 (simplest first), each with the activity prior fitted on the held-out own games;
#   the label-free choice (m164_select.py: test-side recognised share, tolerance 0.003, simplest first);
#   the recognised-share report; the registered pool test of the chosen identifier against the ct106 rule
#   (new source only, look-alike reading unchanged; M37 on both seed sets; on a pass the candidate file <name>).
#   bash scripts/m_series/m161q_step2.sh 30,31,32 S7t ct112
main() {
  local seeds="$1" var="$2" name="$3" old="P,S2y,S4ft,S4Lt" byg="" c="results/m84_cache" tags=() specs=() t best also=()
  python -I scripts/m_series/m161q_tta.py --seeds "$seeds" --name "$var" || return 1
  [ -f "$c/var_S6at.npz" ] && byg="S6at"
  [ -f "$c/var_S6ft.npz" ] && byg="${byg:+$byg,}S6ft"
  specs+=("SQ1:$var" "SQ2:$var,${var}c")
  [ -n "$byg" ] && specs+=("SQ3:$byg,$var" "SQ4:$byg,$var,${var}c")
  specs+=("SQ5:$old,$var" "SQ6:$old,$var,${var}c")
  [ -n "$byg" ] && specs+=("SQ7:$old,$byg,$var" "SQ8:$old,$byg,$var,${var}c")
  for t in "${specs[@]}"; do tags+=("${t%%:*}"); done
  echo "== identifiers: ${specs[*]}"
  python -I scripts/m_series/m164_ident.py --prior fit $(printf -- '--tag %s ' "${specs[@]}") || return 1
  best=$(python -I scripts/m_series/m164_select.py "${tags[@]}") || return 1
  echo "== chosen identifier (label-free rule): $best"
  for t in "${tags[@]}"; do [ "$t" != "$best" ] && also+=("$t"); done
  python -I scripts/m_series/m120_diag_mass.py SPAL~m15 SPALm~m15 $(printf -- '%s~m15 ' "${tags[@]}") || echo "(the recognition-share report failed; the test goes on)"
  python -I scripts/m_series/m163_pool.py --new "$best" --also "${also[@]}" --control SPALm --name "$name"
}
main "$@"; exit $?
