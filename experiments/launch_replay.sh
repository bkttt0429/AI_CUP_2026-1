#!/usr/bin/env bash
# Open a replay page in Chrome with the NVIDIA acceleration flags verified on lab01
# (--enable-gpu --use-angle=vulkan; see reports/lab01_go_gpu_capacity_20261015).
#
#   scripts/launch_replay.sh PAGE.html                 one page (keep it at 50-100 boards)
#   scripts/launch_replay.sh --windows N PAGE.html     N separate windows, board i in window i (N <= 40)
#
# On the lab01 desktop it opens visible windows; over SSH, prefix DISPLAY=:1 to show them on the
# desktop session, or leave DISPLAY unset for headless Chrome (DevTools bound to 127.0.0.1).
# Uses its own profile under results/replay/chrome-profile; no machine-wide setting changes.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
windows=0
if [[ "${1:-}" == "--windows" ]]; then windows="$2"; shift 2; fi
page="$(realpath "${1:?usage: launch_replay.sh [--windows N] PAGE.html}")"; shift
if (( windows > 40 )); then echo "at most 40 windows (measured P95 52 ms at 40)" >&2; exit 1; fi
flags=(--enable-gpu --use-angle=vulkan --no-first-run --no-default-browser-check
       "--user-data-dir=$root/results/replay/chrome-profile")
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  flags+=(--headless=new --remote-debugging-address=127.0.0.1 --remote-debugging-port=0)
fi
if (( windows == 0 )); then
  exec /usr/bin/google-chrome "${flags[@]}" "file://$page" "$@"
fi
/usr/bin/google-chrome "${flags[@]}" --new-window "file://$page?board=0" "$@" &
sleep 2
for (( i = 1; i < windows; i++ )); do
  /usr/bin/google-chrome "${flags[@]}" --new-window "file://$page?board=$i" >/dev/null 2>&1
done
wait
