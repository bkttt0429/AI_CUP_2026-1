#!/usr/bin/env bash
# Closing (2026-11-21): copy one part of the /work staging area into /home so that it can be downloaded from there
# (the user cannot fetch from /work). Real copies - /home is another file system - so the parts are sized to the free space.
#   sbatch ... scripts/nano4/run.sbatch bash logs/nano4/m268_work_part.sh <part number> <relative path> [...]
# Stops (and removes its own partial copy) if /home would fall below FLOOR MB of free space or a write fails.
main() {
  set -uo pipefail
  part="$1"; shift; SRC=/work/u2507115/AICUP1_MOVE_20261121; DST=/home/u2507115/AICUP1_MOVE_20261121_work_part$part; FLOOR=2300
  [ -e "$DST" ] && { echo "$DST exists already"; exit 1; }
  need=$(cd "$SRC" && du -smcL "$@" | tail -1 | cut -f1); avail=$(df -m --output=avail /home/u2507115 | tail -1)
  echo "part $part: $need MB to copy, /home has $avail MB free (floor $FLOOR)"
  [ $((avail - need)) -ge $FLOOR ] || { echo "not enough room: nothing copied"; exit 1; }
  fail() { echo "FAILED: $1 - removing the partial copy $DST"; rm -rf "$DST"; exit 1; }
  mkdir -p "$DST"
  for rel in "$@"; do
    mkdir -p "$DST/$(dirname "$rel")"; rsync -a "$SRC/$rel" "$DST/$(dirname "$rel")/" || fail "copy of $rel"
    a=$(df -m --output=avail /home/u2507115 | tail -1); echo "  copied $rel; /home free $a MB"; [ "$a" -ge $FLOOR ] || fail "free space fell to $a MB"
  done
  ( cd "$DST" && find . -type f ! -name "MANIFEST_part$part.sha256" ! -name 'MANIFEST_work.sha256' -printf '%p\n' | sort > /tmp/m268_files.$$ && grep -F -f <(sed 's/^/  /' /tmp/m268_files.$$) "$SRC/MANIFEST_work.sha256" > "MANIFEST_part$part.sha256"; rm -f /tmp/m268_files.$$
    echo "manifest lines for this part: $(wc -l < MANIFEST_part$part.sha256) (files copied: $(find . -type f ! -name 'MANIFEST_*' | wc -l))"; sha256sum -c --quiet "MANIFEST_part$part.sha256" && echo "sha256 of every copied file matches the manifest" ) || fail "verification"
  echo "done: $DST = $(du -sm "$DST" | cut -f1) MB, $(find "$DST" -type f | wc -l) files; /home free $(df -m --output=avail /home/u2507115 | tail -1) MB"
}
main "$@"; exit $?
