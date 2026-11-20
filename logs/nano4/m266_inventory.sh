#!/usr/bin/env bash
# Closing (2026-11-21): read-only inventory of the rank project before it is moved off nano4. Sizes in MB (disk usage), file counts,
# symlinks with their targets, the largest files. Nothing is changed.
#   sbatch ... scripts/nano4/run.sbatch bash logs/nano4/m266_inventory.sh reports/closing_20261121/inventory_raw.txt
main() {
  set -uo pipefail
  out="$1"; root=/home/u2507115/BK_AICUP_2026-1; cd "$root"
  {
    echo "== $(date '+%F %T') inventory of $root"
    echo "== top level (MB, files)"; for d in $(ls -A); do printf '%8d MB %8d files  %s%s\n' "$(du -sm "$d" 2>/dev/null | cut -f1)" "$(find "$d" -type f 2>/dev/null | wc -l)" "$d" "$([ -L "$d" ] && echo " -> $(readlink -f "$d")")"; done | sort -rn
    echo "== results/* (MB, files; symlinks marked)"; for d in results/*; do printf '%8d MB %8d files  %s%s\n' "$(du -sm "$d" 2>/dev/null | cut -f1)" "$(find -L "$d" -type f 2>/dev/null | wc -l)" "$d" "$([ -L "$d" ] && echo " -> $(readlink -f "$d") [$(du -smL "$d" 2>/dev/null | cut -f1) MB behind the link]")"; done | sort -rn
    echo "== results/r_variants by extension (MB)"; find results/r_variants -type f -printf '%s %f\n' | awk '{n=split($2,a,"."); e=a[n]; s[e]+=$1; c[e]++} END {for (e in s) printf "%8.0f MB %7d files .%s\n", s[e]/1e6, c[e], e}' | sort -rn | head
    echo "== results/trunk_seq/* and results/l1b/* (MB)"; du -sm results/trunk_seq/* results/l1b/* 2>/dev/null | sort -rn | head -40
    echo "== logs by extension (MB)"; find logs -type f -printf '%s %f\n' | awk '{n=split($2,a,"."); e=a[n]; s[e]+=$1; c[e]++} END {for (e in s) printf "%8.0f MB %7d files .%s\n", s[e]/1e6, c[e], e}' | sort -rn | head
    echo "== data, models, tools, reports (MB)"; du -sm data/* models/* tools/* reports/* 2>/dev/null | sort -rn | head -30
    echo "== all symlinks up to depth 3"; find . -maxdepth 3 -type l -printf '%p -> %l\n' | sort
    echo "== symlinks deeper than depth 3 (count by directory)"; find . -mindepth 4 -type l -printf '%h\n' 2>/dev/null | sort | uniq -c | sort -rn | head -20
    echo "== 40 largest files"; find . -type f -size +300M -printf '%s %p\n' 2>/dev/null | sort -rn | head -40 | awk '{printf "%8.0f MB %s\n", $1/1e6, $2}'
    echo "== /work/u2507115 (MB)"; for d in /work/u2507115/*; do printf '%8d MB  %s\n' "$(du -sm "$d" 2>/dev/null | cut -f1)" "$d"; done | sort -rn
    echo "== /work/u2507115/aicup1_moved (MB)"; du -sm /work/u2507115/aicup1_moved/* 2>/dev/null | sort -rn
    echo "== ~/.cache entries over 5 MB"; du -sm /home/u2507115/.cache/* 2>/dev/null | awk '$1 >= 5' | sort -rn
    echo "== home top level over 100 MB"; du -sm /home/u2507115/* /home/u2507115/.[a-z]* 2>/dev/null | awk '$1 >= 100' | sort -rn
    echo "== links from the player-identification project into this tree"; find /home/u2507115/BK_AICUP_2026-2 -maxdepth 4 -type l -lname '*BK_AICUP_2026-1*' -printf '%p -> %l\n' 2>/dev/null | head -40; find /home/u2507115/BK_AICUP_2026-2 -maxdepth 4 -type l -lname '*aicup1_moved*' -printf '%p -> %l\n' 2>/dev/null | head -20
    echo "== done $(date '+%F %T')"
  } > "$out.tmp" 2>&1
  mv "$out.tmp" "$out"; echo "written $out ($(wc -l < "$out") lines)"
}
main "$@"; exit $?
