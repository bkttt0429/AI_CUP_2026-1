#!/usr/bin/env python
"""Section 4M M161: orientation-averaged queries (m124_tta_queries.py, unchanged) for the networks of m161p_prod.py.

  python -I scripts/m_series/m161p_tta.py --seeds 20 --name S6t      (reads results/m161p and /work/u2507115/m161p_ckpt, parts in
                                                                results/m161p_tta, writes results/m84_cache/var_S6t.npz)
"""
import sys  # 設定模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m124_tta_queries as t  # noqa: E402  M124 的八向平均查詢程式

t.OUT, t.CK, t.TMP = ROOT / "results/m161p", Path("/work/u2507115/m161p_ckpt"), ROOT / "results/m161p_tta"  # 改讀 M161p 的網路與檢查點，中間分段檔放到 results/m161p_tta

if __name__ == "__main__":  # 直接執行時
    t.main()  # 執行 M124 的主流程（m161_step2.sh 以 --name S6at 產生 var_S6at.npz）
