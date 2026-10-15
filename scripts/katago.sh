#!/usr/bin/env bash
# Run the KataGo v1.15.0 Eigen binary on lab01. The prebuilt binary links
# libzip.so.5 and OpenSSL 1.1, which lab01 lacks system-wide, so they live in a
# dedicated conda env and are exposed only to this process.
ROOT="$HOME/AICUP_2026-1"
exec env LD_LIBRARY_PATH="$ROOT/tools/katago-libs/lib" "$ROOT/tools/katago/katago" "$@"
