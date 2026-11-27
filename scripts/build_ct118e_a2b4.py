#!/usr/bin/env python3
"""Build the final file ct118e_a2b4 = ct118e + 4 Private A->B (participant's decision 2026-11-23: 「好 改那4題」).

This is the script that wrote submission_rank_ct118e_a2b4.csv (md5 2c2e4ad637a6407a35de7f599c4ac641) on 2026-11-23 02:45;
only this docstring was edited when it was added to the repo. The whole chain from ct112 is checked by
scripts/verify_final_chain.py.

Usage: python3 -I scripts/build_ct118e_a2b4.py <input ct118e csv> <output csv>
Checks: input md5 = ct118e; header; 800 rows (400 pub / 400 priv); LF only; the 4 targets are A in the input.
Output: same bytes except the 4 Private rows A -> B; verified by re-reading.
"""
import hashlib  # 計算輸入與輸出檔的 md5
import sys  # 讀取命令列參數

CT118E_MD5 = "0ef7368fb85cf1eecde2eb18b99367cb"  # ct118e 的 md5，確認輸入檔正確
TARGETS = ["priv_q_0062", "priv_q_0125", "priv_q_0204", "priv_q_0325"]  # 要由 A 改成 B 的 4 題 Private
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]  # 合法的 10 個等級


def rows_of(raw):  # 把檔案位元組拆成各列文字
    text = raw.decode("utf-8")  # 以 UTF-8 解碼
    lines = text.split("\n")  # 只以 LF 切列
    if lines and lines[-1] == "":  # 檔尾有換行時會多出一個空字串
        lines = lines[:-1]  # 去掉該空字串
    return lines  # 回傳各列（含標頭）


def main():  # 建檔流程的進入點
    src, dst = sys.argv[1], sys.argv[2]  # 輸入的 ct118e 與輸出檔路徑
    raw = open(src, "rb").read()  # 以位元組讀入，保留原始換行
    md5 = hashlib.md5(raw).hexdigest()  # 輸入檔的 md5，用來確認輸入就是 ct118e
    print(f"input {src}: {len(raw)} bytes, md5 {md5} ({'= ct118e' if md5 == CT118E_MD5 else 'NOT ct118e'})")  # 印出輸入檔資訊
    assert md5 == CT118E_MD5, "input is not ct118e - stop"  # 輸入必須是 ct118e 才繼續
    assert b"\r" not in raw, "CR found"  # 確認只用 LF 換行
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM found"  # 確認沒有 UTF-8 BOM
    lines = rows_of(raw)  # 拆成各列
    assert lines[0] == "question_id,pred_rank", lines[0]  # 標頭必須是官方格式
    body = [ln.split(",") for ln in lines[1:]]  # 每列拆成題號與答案
    assert len(body) == 800 and all(len(r) == 2 for r in body)  # 共 800 列且每列兩欄
    ids = [r[0] for r in body]  # 題號清單（保留原順序）
    assert len(set(ids)) == 800  # 題號不可重複
    assert sum(i.startswith("pub_q_") for i in ids) == 400 and sum(i.startswith("priv_q_") for i in ids) == 400  # Public、Private 各 400 題
    assert all(r[1] in LEVELS for r in body)  # 每個答案都是合法等級
    ans = dict(body)  # 題號對答案的字典
    for q in TARGETS:  # 逐一檢查 4 個目標題
        print(f"  {q}: input answer {ans[q]}")  # 印出目標題的原答案
        assert ans[q] == "A", f"{q} is not A in the input"  # 目標題在輸入中必須是 A
    out_lines = [lines[0]] + [f"{q},{'B' if q in TARGETS else a}" for q, a in body]  # 目標題改成 B，其餘照抄
    out = ("\n".join(out_lines) + ("\n" if raw.endswith(b"\n") else "")).encode("utf-8")  # 檔尾換行與輸入一致
    with open(dst, "wb") as fh:  # 以位元組寫出，避免換行被轉換
        fh.write(out)  # 寫入輸出檔
    # 重新讀回並驗證
    back = rows_of(open(dst, "rb").read())  # 讀回剛寫出的檔
    nb = [ln.split(",") for ln in back[1:]]  # 讀回的題號與答案
    assert back[0] == lines[0] and [r[0] for r in nb] == ids  # 標頭與題號順序不變
    diff = [(q, a, b) for (q, a), (_, b) in zip(body, nb) if a != b]  # 找出答案有變的題目
    assert sorted(d[0] for d in diff) == sorted(TARGETS) and all(a == "A" and b == "B" for _, a, b in diff)  # 只有 4 個目標題由 A 改成 B
    assert all(a == b for (q, a), (_, b) in zip(body, nb) if q.startswith("pub_q_"))  # Public 的答案完全不變
    omd5 = hashlib.md5(open(dst, "rb").read()).hexdigest()  # 輸出檔的 md5，印出供核對
    print(f"output {dst}: {len(out)} bytes, md5 {omd5}")  # 印出輸出檔資訊
    print(f"  changed rows: {len(diff)} (all Private, A -> B): " + ", ".join(d[0] for d in diff))  # 列出改動的題號
    print("  Public 400 rows identical to ct118e; order, header and line endings unchanged")  # 回報其餘內容未變


if __name__ == "__main__":  # 直接執行時
    main()  # 執行建檔
