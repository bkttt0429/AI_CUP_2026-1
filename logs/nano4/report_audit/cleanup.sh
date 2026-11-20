#!/bin/bash
# BK_AICUP_2026-1 cleanup. Without --apply it only prints what would go; with --apply it deletes exactly that list.
set -euo pipefail
cd "$HOME/BK_AICUP_2026-1"
A="$HOME/.cache/report_audit"; L="$A/cleanup_list.txt"; : > "$L"
PARTIAL="results/l1b/scores5 results/l1b/scores5_opp results/l1b/scores results/l1b/scores_B results/l1b/scores2 results/l1b/scores2_B results/l1b/scores_init results/l1b/se_A_scores results/l1b/se_A_opp results/l1b/se_B_scores results/l1b/se_B_opp results/l1b/cache_0 results/seq/sum results/seq/gru results/seq/conv results/seq/ctm_seq"
WHOLE="results/ladder/cv_smoke results/nano4_smoke results/smoke results/quality_smoke results/mock_fold0_40_fused4b models/baseline_smoke data/baseline_smoke models/baseline_seen data/baseline_seen models/baseline_official_s43 data/baseline_official analysis_logs"
git ls-files > "$A/tracked.txt"
for d in $PARTIAL; do [ -d "$d" ] && find "$d" -type f | sort | comm -23 - <(LC_ALL=C sort "$A/tracked.txt" | LC_ALL=C grep "^$d/" | LC_ALL=C sort) >> "$L" || true; done
ls results/l1b/heads5*.pt 2>/dev/null >> "$L" || true
for d in $WHOLE; do [ -d "$d" ] && find "$d" -type f >> "$L" || true; done
for f in engine.err results/engine_smoke.err; do [ -f "$f" ] && echo "$f" >> "$L" || true; done
if grep -E '_hd(/|patch/)|results/l1b/(cache_A|cache_B|scores3|heads3|heads2)|^results/scores|^results/models|^release/|^data/training|^data/mock_tests|models/katago|models/baseline_official/' "$L"; then echo "PROTECTED path on the list, abort"; exit 1; fi
if LC_ALL=C sort "$L" | LC_ALL=C comm -12 - <(LC_ALL=C sort "$A/tracked.txt") | grep .; then echo "tracked file on the list, abort"; exit 1; fi
echo "files: $(wc -l < "$L")"; tr '\n' '\0' < "$L" | du -ch --files0-from=- | tail -1
awk -F/ '{ if ($1=="results"&&($2=="l1b"||$2=="seq"||$2=="ladder")) print $1"/"$2"/"$3; else print $1"/"$2 }' "$L" | sed 's#heads5.*#heads5*.pt#' | sort | uniq -c
[ "${1:-}" = "--apply" ] || { echo "dry run: nothing deleted; rerun with --apply"; exit 0; }
[ -z "$(squeue -u "$USER" -h)" ] || { echo "jobs are running, abort"; exit 1; }
tr '\n' '\0' < "$L" | xargs -0 rm -f
for d in $PARTIAL $WHOLE; do [ -d "$d" ] && find "$d" -depth -type d -empty -delete || true; done
rm -rf "$HOME/BK_AICUP_tsfel_venv"
git status --short | cut -c1-70; echo "tracked files missing: $(git ls-files -d | wc -l)"; git worktree list; df -h "$HOME" | tail -1
