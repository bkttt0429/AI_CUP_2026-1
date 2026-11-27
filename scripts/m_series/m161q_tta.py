#!/usr/bin/env python
"""Section 4M M161q: orientation-averaged queries (m124_tta_queries.py, unchanged) for the networks of m161q_prod.py, and the
learned centres as a second source.

  python -I scripts/m_series/m161q_tta.py --seeds 30,31,32 --name S7t
      reads results/m161q and /work/u2507115/m161q_ckpt (width 512 / feed-forward 2,048), parts in results/m161q_tta, writes
      results/m84_cache/var_S7t.npz (prototypes; m84_variant's format) and var_S7tc.npz (protos = the learned centres,
      has = the players with a centre [members, 5, C], the same query vectors) for m164_ident.py.
"""
import argparse  # 解析命令列參數
import sys  # 設定模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 讀寫 npz 與陣列運算

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m124_tta_queries as t  # noqa: E402  M124 的八向平均查詢程式

base = t.base  # M124 使用的 m88 模組（模型結構與常數）
t.OUT, t.CK, t.TMP = ROOT / "results/m161q", Path("/work/u2507115/m161q_ckpt"), ROOT / "results/m161q_tta"  # 改讀 M161q 的網路與檢查點，中間分段檔放到 results/m161q_tta
base.D, base.FFN = 512, 2048  # M161q 是加寬的網路：寬度 512、前饋 2,048

if __name__ == "__main__":  # 直接執行時
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0"); ap.add_argument("--name", default="S4t"); a_, _ = ap.parse_known_args(); seeds = [int(x) for x in a_.seeds.split(",")]  # 先讀出種子與變體名（其餘參數留給 M124 的 main）
    t.main()  # 執行 M124 的八向平均，寫出 var_<name>.npz
    V = np.load(base.CACHE / f"var_{a_.name}.npz"); C = V["protos"].shape[2]; cen = np.zeros_like(V["protos"]); has = np.zeros((len(seeds), 5, C), bool)  # 讀回剛寫的變體檔，準備中心與遮罩陣列
    for si, s in enumerate(seeds):  # 每個種子
        for j, v in enumerate(base.NETS):  # 每個網路（1-4 與全體）
            z = np.load(t.OUT / f"out_{v}_s{s}.npz"); assert "centres" in z.files, f"net {v} seed {s} has no centres"; cen[si, j] = z["centres"]; has[si, j] = z["has_centre"]  # 取出學到的中心與哪些玩家有中心
    np.savez(base.CACHE / f"var_{a_.name}c.npz", protos=cen, qvec=V["qvec"], has=has)  # 寫出以中心當原型的第二個來源
    print(f"var_{a_.name}c written: centres of {len(seeds)} seed(s); players with a centre per network " + ", ".join(str(int(has[0, j].sum())) for j in range(5)), flush=True)  # 回報各網路有中心的玩家數
