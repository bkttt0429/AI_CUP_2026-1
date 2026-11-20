#!/usr/bin/env bash
# M230: test-side pooled posteriors of several identifiers, then the Public log-score comparison (report only).
main() {
  local x; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2; mkdir -p /dev/shm/m230_$$
  for x in "$@"; do
    python -I logs/nano4/m230_post.py --new "$x" --json "m230_cv_$x.json" --build-anyway --out-dir /dev/shm/m230_$$ | grep "PRIMARY\|REGISTERED\|written\|DRY RUN" | cut -c1-200 || { echo "posterior of $x FAILED"; rm -rf /dev/shm/m230_$$; return 1; }
  done
  rm -rf /dev/shm/m230_$$; python -I logs/nano4/m230_like.py "$@"
}
main "$@"; exit $?
