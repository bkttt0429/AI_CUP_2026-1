#!/usr/bin/env python
"""M120 diagnostic (report only): how much of the opponent side does each identifier really recognise?
Per identifier (the m84_<tag>_val.npz files): the estimated share of true identifications among the opponent-side queries
(sum over the margin bands of max(share_b - nominal_b, 0), bands = quantiles of the impostor margins, m71_step2.fit), by group,
the share of opponent-side queries in the bands >= q90 / >= q98, and how concentrated the readings of the top bands are
(label of the identified candidate minus the target's label: same level, within one). Next to it the fused held-out top-1
of the target-side own games (the number the networks were selected on)."""
import json  # 讀取各識別器的評估檔、寫出結果
import sys  # 讀取命令列參數與調整搜尋路徑
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同資料夾的實驗模組可被匯入
import m86_bundle as b  # noqa: E402  匯入前須先設定 sys.path；讀取各識別器來源
s2 = b.s2  # M71 第 2 步的分帶與校準
RV = ROOT / "results/r_variants"  # 識別器檔與輸出的資料夾
ids = sys.argv[1:] or ["5e", "S2", "SP2", "S2e", "SP2e", "S2y", "SP2y", "S3c", "S4", "SP4"]  # 要診斷的識別器標籤（未指定時用預設十個）
V, T, key_pid, used = b.sources(ids)  # 讀入驗證端與測試端的識別器輸出
nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))  # 各分帶的名目占比
print(f"{'id':7s} held-out  mass(all)  even   HD1-B  HD1-W   >=q90   >=q98   top bands: same / within one   test: >=q90  >=q98")  # 表頭
out = {}  # 收集各識別器的結果
for x in used:  # 逐一識別器
    z = V[x]; ones = np.ones(len(z["y"]), bool); edges, cal, band, g = s2.fit(z, ones); ok = g >= 0  # 以全部驗證查詢擬合分帶；只看有證據的組別
    mass = {k: float(np.maximum(cal[k]["share"] - nominal, 0).sum()) for k in (0, 1, 2)}; w = np.array([(g == k).sum() for k in (0, 1, 2)], float); w /= w.sum()  # 各組超出名目占比的總量（真正認出的估計）與組別權重
    top = ok & (band >= 5); d = (z["opp_lab"] - z["y"])[top & (g == 0)]  # 最高幾帶（>= q98）互先局的標籤差
    t = T[x]; tb = np.where(~t["target_black"], np.searchsorted(edges[True], t["opp_m"]), np.searchsorted(edges[False], t["opp_m"])); tg = s2.group_of(t["hd"], t["target_black"]); tok = tg >= 0  # 測試端對手查詢的分帶與組別
    ev = RV / f"m84_{x}_eval.json"; h = None  # 該識別器的評估檔
    if ev.exists():  # 有評估檔時
        e = json.loads(ev.read_text()); h = (e.get("heldout", {}).get("fused") or e.get("heldout_fused") or {}).get("top1")  # 留出本人對局的融合 top-1（相容兩種欄位格式）
    row = {"heldout": h, "mass": float(sum(w[k] * mass[k] for k in (0, 1, 2))), "mass_by_group": mass, "q90": float((band[ok] >= 3).mean()), "q98": float((band[ok] >= 5).mean()), "top_same": float((d == 0).mean()), "top_within1": float((np.abs(d) <= 1).mean()),  # 驗證端：認出量、高分帶占比、最高帶的同級率
           "test_q90": float((tb[tok] >= 3).mean()), "test_q98": float((tb[tok] >= 5).mean())}  # 測試端落在高分帶的占比
    out[x] = row  # 存入結果
    print(f"{x:7s} {h if h is None else round(h, 3)!s:8s}  {row['mass']:.4f}     {mass[0]:.4f} {mass[1]:.4f} {mass[2]:.4f}  {row['q90']:.4f}  {row['q98']:.4f}   {row['top_same']:.3f} / {row['top_within1']:.3f}                 {row['test_q90']:.4f}  {row['test_q98']:.4f}")  # 印出一列
(RV / "m120_diag_mass.json").write_text(json.dumps(out, indent=1))  # 寫出 JSON 結果
