#!/usr/bin/env bash
# The fine-tuned L1b trunks (about 103 MB each) are stored in parts because GitHub refuses files over 100 MB.
# This rebuilds results/l1b/heads3_{A,B}.trunk.pt from the parts and checks every L1b file against heads3.sha256.
#   bash scripts/join_trunk_parts.sh
main() {
  set -euo pipefail
  cd "$(dirname "$0")/../results/l1b"
  for h in A B; do
    cat heads3_$h.trunk.pt.part? > heads3_$h.trunk.pt
  done
  sha256sum -c heads3.sha256
}
main "$@"; exit $?
