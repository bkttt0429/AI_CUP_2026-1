#!/usr/bin/env bash
# M230b: the Public log-score comparison over every identifier that has test-side files (posteriors already stored are reused).
main() {
  local x ok=(); export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2; mkdir -p /dev/shm/m230b_$$
  for x in "$@"; do
    [ -f results/r_variants/m230_post_$x.npz ] || python -I logs/nano4/m230_post.py --new "$x" --json "m230_cv_$x.json" --build-anyway --out-dir /dev/shm/m230b_$$ > /dev/shm/m230b_$$/$x.out 2>&1
    if [ -f results/r_variants/m230_post_$x.npz ]; then ok+=("$x"); grep -h "PRIMARY" /dev/shm/m230b_$$/$x.out 2>/dev/null | cut -c1-120 | sed "s/^/$x /"; else echo "no posterior for $x: $(tail -1 /dev/shm/m230b_$$/$x.out | cut -c1-160)"; fi
  done
  rm -rf /dev/shm/m230b_$$; python -I logs/nano4/m230_like.py SQ4 SZ3 SY3 SV1 SC2 SC3 SW1 SW2 "${ok[@]}"
}
main "$@"; exit $?
