#!/usr/bin/env bash
# One-time Python environment for this project on NCHC nano4. Run it on the login node (it
# needs internet, not a GPU), from anywhere inside the extracted project:
#   bash scripts/nano4/setup.sh                    # torch 2.14.1 built for CUDA 12.6 (driver >= R525)
#   TORCH_CUDA=cu130 bash scripts/nano4/setup.sh   # the CUDA 13.0 build lab01 uses (driver >= R580)
# Everything stays inside the project: uv in tools/uv, Python in tools/uv-python, the download
# cache in tools/uv-cache and the env in .venv. Rerunning is safe and skips finished work.
set -euo pipefail

UV_VERSION=0.12.21
UV_SHA256_x86_64=23f02075b652bb1df64178cfae41b5caf160822e720e2663568f3f5d63bc52c0
UV_SHA256_aarch64=030b69227b40af8c1981b7301793dc66e71ed3c796ea8688209dd268bd91ec51
PYTHON_VERSION=3.12
TORCH_VERSION=2.14.1
TORCH_CUDA="${TORCH_CUDA:-cu126}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export UV_CACHE_DIR="$ROOT/tools/uv-cache" UV_PYTHON_INSTALL_DIR="$ROOT/tools/uv-python"
export UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1
UV="$ROOT/tools/uv/uv"
step() { printf '\n== %s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "setup failed: $*" >&2; exit 1; }

step "checks in $ROOT"
for f in requirements.txt models/katago/b18c384nbt-humanv0.ckpt repo/KataGo/python/katago; do
  [ -e "$f" ] || die "$f is missing: run this inside the extracted AICUP_2026-1 folder"
done
arch=$(uname -m)
case "$arch" in
  x86_64) uv_sha=$UV_SHA256_x86_64 ;;
  aarch64) uv_sha=$UV_SHA256_aarch64 ;;
  *) die "unsupported CPU architecture $arch" ;;
esac
glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
[ -n "$glibc" ] || die "not a glibc system"
[ "$(printf '%s\n' 2.28 "$glibc" | sort -V | head -n 1)" = 2.28 ] || die "glibc $glibc is older than 2.28, which the torch wheels need"
for url in https://github.com https://pypi.org/simple/ "https://download.pytorch.org/whl/$TORCH_CUDA/"; do
  curl -sSfI --max-time 30 -o /dev/null "$url" || die "cannot reach $url from $(hostname) (proxy needed?)"
done
echo "$arch, glibc $glibc, free space here $(df -h . | awk 'NR == 2 {print $4}') (the env needs about 6 GB)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU on this node: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
fi

step "uv $UV_VERSION"
if [ -x "$UV" ] && [ "$("$UV" --version | awk '{print $2}')" = "$UV_VERSION" ]; then
  echo "already installed"
else
  tmp=$(mktemp -d)
  curl -sSfL --retry 3 -o "$tmp/uv.tar.gz" \
    "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$arch-unknown-linux-gnu.tar.gz"
  echo "$uv_sha  $tmp/uv.tar.gz" | sha256sum -c --quiet - || die "the uv download does not match its published checksum"
  mkdir -p tools/uv
  tar -xzf "$tmp/uv.tar.gz" -C tools/uv --strip-components=1
  rm -rf "$tmp"
  "$UV" --version
fi

step "Python $PYTHON_VERSION and .venv"
"$UV" python install --no-bin "$PYTHON_VERSION"  # no python3.12 link in ~/.local/bin
if [ -x .venv/bin/python ]; then
  echo "reusing .venv ($(.venv/bin/python -V))"
else
  "$UV" venv --python "$PYTHON_VERSION" .venv
fi

step "torch $TORCH_VERSION+$TORCH_CUDA"
"$UV" pip install --python .venv/bin/python --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
  "torch==$TORCH_VERSION+$TORCH_CUDA"

step "pinned packages from requirements.txt"
"$UV" pip install --python .venv/bin/python -r requirements.txt

step "KataGo python package on the env path"
site=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$ROOT/repo/KataGo/python" > "$site/katago.pth"

step "import check"
.venv/bin/python -I - <<'EOF'
import sys
sys.path.insert(0, "src")
import numpy, pandas, torch
import aicup.humansl  # also imports the KataGo modules
print(f"python {sys.version.split()[0]} | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
      f"numpy {numpy.__version__} | pandas {pandas.__version__} | katago ok")
EOF
mkdir -p logs/nano4
"$UV" pip freeze --python .venv/bin/python > logs/nano4/env-freeze.txt
"$UV" cache prune --ci >/dev/null 2>&1 || true  # the env keeps its files; drop the wheel copies

step "done"
echo "env: $ROOT/.venv (package list in logs/nano4/env-freeze.txt)"
echo "next: cd $ROOT && sbatch experiments/nano4/smoke.sbatch"
