# Helpers for the node scripts of a multi-node job (sourced). 2026-11-17.
# Nothing bulky is written to /work before the flag file exists AND /work has at least 8 GB free (the other session's space plan).
work_free_gb() { df -BG --output=avail /work/u2507115 | tail -1 | tr -dc 0-9; }
copy_new() {  # $1 = source folder, $2 = destination folder: every finished file that is missing or older at the destination
  local f d; mkdir -p "$2" || return 1
  for f in "$1"/*; do
    [ -f "$f" ] || continue; case "$f" in *.tmp|*.part) continue;; esac
    d="$2/$(basename "$f")"
    if [ ! -e "$d" ] || [ "$f" -nt "$d" ]; then cp "$f" "$d.part" && mv "$d.part" "$d" || return 1; fi
  done
}
save_loop() {  # $1 = source, $2 = destination, $3 = flag file, $4 = pid to follow: copies every 15 minutes while the pid lives
  while kill -0 "$4" 2>/dev/null; do
    if [ -e "$3" ] && [ "$(work_free_gb)" -ge 8 ]; then copy_new "$1" "$2"; fi
    sleep 900
  done
}
save_final() {  # $1 = source, $2 = destination, $3 = flag file ($3.drop = give up): waits (at most $4 hours) until it may copy
  local i
  for ((i = 0; i < ${4:-20} * 12; i++)); do
    [ -e "$3.drop" ] && { echo "== $(date '+%F %T') $3.drop found: $1 is not copied"; return 0; }
    if [ -e "$3" ] && [ "$(work_free_gb)" -ge 8 ]; then copy_new "$1" "$2" && { echo "== $(date '+%F %T') copied $1 -> $2 ($(du -sh "$2" | cut -f1)); /work free $(work_free_gb) GB"; return 0; }; fi
    sleep 300
  done
  echo "== $(date '+%F %T') gave up waiting for $3: $1 was NOT copied"; return 1
}
first_gpus() { echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | cut -d, -f1-"$1"; }
