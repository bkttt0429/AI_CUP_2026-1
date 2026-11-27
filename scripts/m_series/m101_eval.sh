#!/usr/bin/env bash
# M101：把兩個臂（results/m101 = 加入外部玩家 -> S3，results/m101c = 對照組 -> S3c；種子 3）併入配對器
# 快取，組出平均來源 S2x（S2e 的三個種子 + S3）與 S2y（S2e + S3 + S3c），並評估各識別器。
set -uo pipefail  # 未定義變數即報錯；管線中任一段失敗即視為失敗
python -I scripts/m_series/m88_seqid.py --stage merge --full --seeds 3 --out m101:S3 || exit 1  # 合併加入外部玩家的臂 -> var_S3
python -I scripts/m_series/m88_seqid.py --stage merge --full --seeds 3 --out m101c:S3c || exit 1  # 合併對照臂 -> var_S3c
python -I - <<'PY' || exit 1  # 內嵌程式：串接各來源的成員，寫出 var_S2x 與 var_S2y
import numpy as np
from pathlib import Path
C = Path("results/m84_cache"); a, b, c = (np.load(C / f"var_{k}.npz") for k in ("S2e", "S3", "S3c")); assert a["protos"].shape[0] == 3 and b["protos"].shape[0] == 1 and c["protos"].shape[0] == 1
np.savez(C / "var_S2x.npz", protos=np.concatenate([a["protos"], b["protos"]]), qvec=np.concatenate([a["qvec"], b["qvec"]]))
np.savez(C / "var_S2y.npz", protos=np.concatenate([a["protos"], b["protos"], c["protos"]]), qvec=np.concatenate([a["qvec"], b["qvec"], c["qvec"]])); print("var_S2x (4 members) and var_S2y (5 members) written", flush=True)
PY
for t in S3 S3c S2x SP2x S2y SP2y; do python -I scripts/m_series/m84_variant.py --stage eval --tag "$t"; done  # 逐一評估六個識別器標籤（定義見 m84_variant.py 的 TAGS）
