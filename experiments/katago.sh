#!/usr/bin/env bash
# Run the KataGo v1.15.0 Eigen binary on lab01. The prebuilt binary links
# libzip.so.5 and OpenSSL 1.1, which lab01 lacks system-wide, so they live in a
# dedicated conda env and are exposed only to this process. Other hosts (nano4)
# have no such env, so the binary may not start there.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec env LD_LIBRARY_PATH="$ROOT/tools/katago-libs/lib" "$ROOT/tools/katago/katago" "$@"
