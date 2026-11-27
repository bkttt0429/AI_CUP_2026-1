#!/usr/bin/env bash
# Closing (2026-11-21): sizes and sha256 of the large files of the final chain that stay on nano4 (not in git).
#   sbatch ... scripts/nano4/run.sbatch bash logs/nano4/m265_manifest.sh <output file>
main() {
  set -euo pipefail
  out="$1"; : > "$out.tmp"
  for d in /work/u2507115/m161q_ckpt /work/u2507115/m161p_ckpt results/m88 results/m90 results/m101 results/m101c results/m120 results/m161q results/m161p results/m77 results/m82 results/m84_cache models/katago; do
    [ -e "$d" ] || { echo "missing: $d" >&2; continue; }
    find -L "$d" -maxdepth 2 -type f -size +1M | sort | while read -r f; do
      printf '%s  %s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -L -c %s "$f")" "$f" >> "$out.tmp"
    done
  done
  mv "$out.tmp" "$out"; echo "written $out: $(wc -l < "$out") files, $(awk '{s+=$2} END {printf "%.1f GB", s/1e9}' "$out")"
}
main "$@"; exit $?
