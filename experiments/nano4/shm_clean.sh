#!/usr/bin/env bash
# Remove this session's scratch from a node's memory disk (2026-11-18). With a glob and a destination folder every matching
# file must first be byte-identical to its copy in the destination; otherwise nothing is removed.
#   bash logs/nano4/shm_clean.sh <folder under /dev/shm> [<quoted glob of files that must have been copied> <destination>]
main() {
  local d=${1:?folder} g=${2:-} dest=${3:-} f n=0
  case "$d" in /dev/shm/m21[689]*|/dev/shm/m221*) ;; *) echo "refusing to remove $d"; return 2;; esac
  if [ -n "$g" ]; then
    for f in $g; do [ -e "$f" ] || continue; cmp -s "$f" "$dest/$(basename "$f")" || { echo "NOT identical: $f vs $dest - nothing removed"; return 1; }; n=$((n + 1)); done
    echo "== $n file(s) verified against $dest"
  fi
  rm -rf "$d" && echo "== $(date '+%F %T') $(hostname): removed $d; /dev/shm now holds: $(ls /dev/shm | tr '\n' ' '); free $(df -BG --output=avail /dev/shm | tail -1 | tr -d ' ')"
}
main "$@"; exit $?
