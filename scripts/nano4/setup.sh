#!/usr/bin/env bash
# 在國網 nano4 上為本專案建立一次性的 Python 環境。請在登入節點執行（它需要網路，
# 不需要 GPU），在解開的專案資料夾內任何位置執行都可以：
#   bash scripts/nano4/setup.sh                    # torch 2.14.1 的 CUDA 12.6 版本（驅動程式 >= R525）
#   TORCH_CUDA=cu130 bash scripts/nano4/setup.sh   # lab01 使用的 CUDA 13.0 版本（驅動程式 >= R580）
# 所有東西都留在專案內：uv 在 tools/uv、Python 在 tools/uv-python、下載快取在
# tools/uv-cache，環境在 .venv。重複執行是安全的，已完成的工作會略過。
set -euo pipefail  # 任何錯誤都立即中止，避免裝出不完整的環境

UV_VERSION=0.12.21  # 固定的 uv 版本
UV_SHA256_x86_64=23f02075b652bb1df64178cfae41b5caf160822e720e2663568f3f5d63bc52c0  # x86_64 版 uv 壓縮檔公布的 sha256
UV_SHA256_aarch64=030b69227b40af8c1981b7301793dc66e71ed3c796ea8688209dd268bd91ec51  # aarch64 版 uv 壓縮檔公布的 sha256
PYTHON_VERSION=3.12  # 專案使用的 Python 版本
TORCH_VERSION=2.14.1  # 固定的 torch 版本
TORCH_CUDA="${TORCH_CUDA:-cu126}"  # torch 的 CUDA 版本，預設 CUDA 12.6

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # 專案根目錄（本檔位於 scripts/nano4/）
cd "$ROOT"  # 之後的相對路徑都以根目錄為準
export UV_CACHE_DIR="$ROOT/tools/uv-cache" UV_PYTHON_INSTALL_DIR="$ROOT/tools/uv-python"  # 下載快取與 Python 都放在專案內
export UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1  # 只用 uv 自己安裝的 Python，不讀外部的 uv 設定檔
UV="$ROOT/tools/uv/uv"  # 專案內 uv 執行檔的路徑
step() { printf '\n== %s %s\n' "$(date +%H:%M:%S)" "$*"; }  # 印出帶時間的步驟標題
die() { echo "setup failed: $*" >&2; exit 1; }  # 印出失敗原因並中止

step "checks in $ROOT"  # 第一步：事前檢查
for f in requirements.txt models/katago/b18c384nbt-humanv0.ckpt repo/KataGo/python/katago; do  # 逐一確認必要的檔案與資料夾
  [ -e "$f" ] || die "$f is missing: run this inside the extracted AICUP_2026-1 folder"  # 缺少就表示不在解開的專案內執行
done  # 結束檔案檢查
arch=$(uname -m)  # CPU 架構，用來挑選 uv 的下載檔
case "$arch" in  # 依架構選擇對應的 sha256
  x86_64) uv_sha=$UV_SHA256_x86_64 ;;  # x86_64 節點
  aarch64) uv_sha=$UV_SHA256_aarch64 ;;  # aarch64 節點
  *) die "unsupported CPU architecture $arch" ;;  # 其他架構不支援
esac  # 結束架構判斷
glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')  # 取得 glibc 版本
[ -n "$glibc" ] || die "not a glibc system"  # 不是 glibc 系統就中止
[ "$(printf '%s\n' 2.28 "$glibc" | sort -V | head -n 1)" = 2.28 ] || die "glibc $glibc is older than 2.28, which the torch wheels need"  # torch 的 wheel 需要 glibc 2.28 以上
for url in https://github.com https://pypi.org/simple/ "https://download.pytorch.org/whl/$TORCH_CUDA/"; do  # 確認三個下載來源都連得到
  curl -sSfI --max-time 30 -o /dev/null "$url" || die "cannot reach $url from $(hostname) (proxy needed?)"  # 連不到就中止，並提示可能需要代理伺服器（proxy）
done  # 結束連線檢查
echo "$arch, glibc $glibc, free space here $(df -h . | awk 'NR == 2 {print $4}') (the env needs about 6 GB)"  # 印出架構、glibc 版本與剩餘空間
if command -v nvidia-smi >/dev/null 2>&1; then  # 這個節點有 nvidia-smi 時
  echo "GPU on this node: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1)"  # 順便印出 GPU 型號與驅動程式版本
fi  # 結束 GPU 資訊

step "uv $UV_VERSION"  # 第二步：安裝 uv
if [ -x "$UV" ] && [ "$("$UV" --version | awk '{print $2}')" = "$UV_VERSION" ]; then  # 已經裝好同一版的 uv
  echo "already installed"  # 略過下載
else  # 還沒有 uv 或版本不同
  tmp=$(mktemp -d)  # 下載用的暫存資料夾
  curl -sSfL --retry 3 -o "$tmp/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$arch-unknown-linux-gnu.tar.gz"  # 從 GitHub 下載對應架構的 uv
  echo "$uv_sha  $tmp/uv.tar.gz" | sha256sum -c --quiet - || die "the uv download does not match its published checksum"  # 核對公布的 sha256，不符就中止
  mkdir -p tools/uv  # uv 的安裝位置
  tar -xzf "$tmp/uv.tar.gz" -C tools/uv --strip-components=1  # 解開到 tools/uv
  rm -rf "$tmp"  # 刪除暫存資料夾
  "$UV" --version  # 確認 uv 可以執行
fi  # 結束 uv 安裝

step "Python $PYTHON_VERSION and .venv"  # 第三步：Python 與虛擬環境
"$UV" python install --no-bin "$PYTHON_VERSION"  # 安裝專案內的 Python；不在 ~/.local/bin 建立 python3.12 連結
if [ -x .venv/bin/python ]; then  # 已經有 .venv
  echo "reusing .venv ($(.venv/bin/python -V))"  # 沿用既有的環境
else  # 還沒有 .venv
  "$UV" venv --python "$PYTHON_VERSION" .venv  # 以指定的 Python 版本建立 .venv
fi  # 結束 .venv 建立

step "torch $TORCH_VERSION+$TORCH_CUDA"  # 第四步：安裝 torch
"$UV" pip install --python .venv/bin/python --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" "torch==$TORCH_VERSION+$TORCH_CUDA"  # 從 PyTorch 官方索引安裝指定 CUDA 版本的 torch

step "pinned packages from requirements.txt"  # 第五步：其他套件
"$UV" pip install --python .venv/bin/python -r requirements.txt  # 依 requirements.txt 安裝固定版本的其餘套件
"$UV" pip install --python .venv/bin/python -r requirements-dev.txt  # 安裝 tools/run_checks.sh 用到的檢查工具（ruff）

step "KataGo python package and AI_CUP_2026-2's aicup_pid on the env path"  # 第六步：讓環境找得到 KataGo 的 Python 套件與競賽一的套件
site=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')  # 取得 .venv 的 site-packages 路徑
echo "$ROOT/repo/KataGo/python" > "$site/katago.pth"  # 以 .pth 檔把 KataGo 的 python 資料夾加入匯入路徑
echo "$(dirname "$ROOT")/AI_CUP_2026-2/src" > "$site/aicup_pid.pth"  # 以 .pth 檔加入並排的競賽一套件 aicup_pid（第二段的對手辨識程式會匯入）

step "import check"  # 第七步：匯入檢查
.venv/bin/python -I - <<'EOF'  # 以 heredoc 傳入檢查程式：匯入主要套件與 KataGo 模組
import sys
sys.path.insert(0, "src")
import numpy, pandas, torch
import aicup.humansl  # also imports the KataGo modules
print(f"python {sys.version.split()[0]} | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
      f"numpy {numpy.__version__} | pandas {pandas.__version__} | katago ok")
EOF
mkdir -p logs/nano4  # 套件清單的輸出資料夾
"$UV" pip freeze --python .venv/bin/python > logs/nano4/env-freeze.txt  # 記錄實際安裝的套件與版本
"$UV" cache prune --ci >/dev/null 2>&1 || true  # 環境已保有自己的檔案；刪除快取中的 wheel 副本

step "done"  # 完成
echo "env: $ROOT/.venv (package list in logs/nano4/env-freeze.txt)"  # 印出環境位置與套件清單的位置
echo "next: cd $ROOT && sbatch -J checks -t 00:30:00 scripts/nano4/run.sbatch bash tools/run_checks.sh --all"  # 提示下一步：在計算節點執行全部檢查
