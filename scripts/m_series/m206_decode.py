#!/usr/bin/env python
"""M206: decode a probe that changes a few Public answers of the reference file (design doc M206).

score * 400 = E + exp(-1) * (W - E) with E = exact answers, W = answers within one level; the leaderboard truncates to 7
decimals. For a probe with n changed Public answers: the integer pairs (dE, dW) compatible with the two printed scores.

  python -I scripts/m_series/m206_decode.py upload_probe/probe_4D5D_locked.csv 0.8300000 [--ref submission_rank_ct112.csv --ref-score 0.8331743]
"""
import csv  # 讀取提交檔
import math  # 計算 exp(-1)
import sys  # 讀取命令列參數

E1 = math.exp(-1.0)  # 差一級時的單題得分
args = [a for a in sys.argv[1:] if not a.startswith("--")]; opt = dict(zip(sys.argv[1:], sys.argv[2:]))  # 位置參數與各 --選項 的值
ref = opt.get("--ref", "submission_rank_ct112.csv"); s0 = float(opt.get("--ref-score", "0.8331743")); probe, s1 = args[0], float(args[1])  # 參考檔與其分數、探測檔與其分數
rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(f))}  # noqa: E731  讀成題號對答案的字典
a, b = rd(ref), rd(probe); allch = [q for q in a if a[q] != b[q]]; ch = [q for q in allch if q.startswith("pub")]; n = len(ch)  # 探測檔改動的題目；Private 的答案不計入 Public 分數
if len(allch) != n:  # 探測檔也改了 Private 題
    print(f"note: the file also changes {len(allch) - n} Private answers (not readable from the Public score)")  # 提示這些改動無法從 Public 分數讀出
# 真實分數落在 [s, s + 1e-7)：兩個真實值的差與 400 * (s1 - s0) 相差在 (-4e-5, +4e-5) 之內
d = 400.0 * (s1 - s0); sol = [(de, dw) for de in range(-n, n + 1) for dw in range(-n, n + 1) if abs((1 - E1) * de + E1 * dw - d) < 4.2e-5 and abs(dw - de) <= n]  # 分數差換成 400 題的點數，列舉相容的 (dE, dW) 整數對
print(f"{probe}: {n} Public answers changed ({', '.join(f'{q} {a[q]}->{b[q]}' for q in ch[:12])}{' ...' if n > 12 else ''}); score {s1:.7f} against {s0:.7f} = {d:+.4f} points")  # 印出改動的題目與分數差
print("compatible (exact change dE, within-one change dW): " + (", ".join(f"({de:+d}, {dw:+d})" for de, dw in sol) if sol else "NONE - check the digits"))  # 印出所有相容的解，沒有解時提示檢查位數
