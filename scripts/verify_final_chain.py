#!/usr/bin/env python3
"""Check, byte for byte, the chain from the model output ct112 to the final file ct118e_a2b4 (standard library only).

Every answer that differs between ct112 and the final file is listed in STEPS below: question, answer before, answer
after, the step that changed it and the record entry (M### = docs/records/architecture_v2_2026-11-02.md).

  backward  start from the final file and undo the steps one by one; every intermediate file must have the md5 that was
            registered when it was built (ct118e, ct118d, ct117, ct116, ct112).
  forward   start from ct112 -- the one rebuilt by the backward pass, or --ct112 FILE, for example the file rebuilt from
            the stored model outputs with
              python -I scripts/m_series/m163_pool.py --new SQ4 --control SPALm --name ctverify
            (needs the data and results/ of the local backup in the repo layout) -- and apply the steps; every file on
            the way, and the end result, must have its registered md5.
  --ct116 / --ct117 / --ct118d / --ct118e FILE  also compare stored copies of those files with the rebuilt ones.
  --write DIR   write the rebuilt intermediate files to DIR (never overwrites).

  python3 -I scripts/verify_final_chain.py submission_rank_ct118e_a2b4.csv [--ct112 FILE] [--ct116 FILE] [--ct118e FILE]

Exit status 0 = every check passed; 1 = a check failed (the failing check is printed).
"""
import argparse  # 解析命令列參數
import hashlib  # 計算 md5
import sys  # 結束碼與錯誤結束
from pathlib import Path  # 讀寫檔案

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]  # 比賽的 10 個等級，答案只能是其中之一
HEADER = "question_id,pred_rank"  # 提交檔的標題列

# 各檔建檔時登記的 md5（有完整記錄的用完整值，否則用前 12 個十六進位字元）及其 Public 讀數。
MD5 = {  # 檔名 -> 登記的 md5
    "ct112": "7d99d6f2c5b7",  # 合併層的輸出，只記前 12 碼
    "ct116": "6b053db6672cd31cbb2e6b742de18649",  # 完整 md5
    "ct117": "feaaca4f827a",  # 沒有上傳過的中間檔，只記前 12 碼
    "ct118d": "12165c64bcfa",  # 只記前 12 碼
    "ct118e": "0ef7368fb85cf1eecde2eb18b99367cb",  # 備援檔，完整 md5
    "ct118e_a2b4": "2c2e4ad637a6407a35de7f599c4ac641",  # 最終檔，完整 md5
}  # 結束 md5 表
PUBLIC = {"ct112": "0.8331743", "ct116": "0.8394955", "ct117": "never uploaded", "ct118d": "0.8546379",  # 各檔的 Public 讀數（只用於印出）
          "ct118e": "0.8587182", "ct118e_a2b4": "0.8587182"}  # 最終檔的 Public 答案與 ct118e 相同，讀數也相同

# （建出的檔、來源檔、紀錄條目、決定的內容、[(題目, 改前答案, 改後答案), ...]）
STEPS = [  # 從 ct112 到最終檔的每一步，依時間順序
    ("ct116", "ct112", "M209 (2026-11-17)",  # 第一步：ct112 -> ct116
     "Private: the 4D -> 5D push unlocked (6 questions back to 4D) and 'never leave A' relaxed on the two questions with "  # 決定的內容（前半）
     "the highest B probability (participant's decision); Public: 7 answers from the decoded Public readings",  # 決定的內容（後半）
     [("priv_q_0039", "5D", "4D"), ("priv_q_0144", "5D", "4D"), ("priv_q_0162", "5D", "4D"),  # Private：解除 4D -> 5D 推移的鎖定、由 5D 改回 4D 的 6 題（前 3 題）
      ("priv_q_0167", "5D", "4D"), ("priv_q_0187", "5D", "4D"), ("priv_q_0274", "5D", "4D"),  # （後 3 題）
      ("priv_q_0084", "A", "B"), ("priv_q_0087", "A", "B"),  # Private：放寬「不離開 A」的 2 題
      ("pub_q_0109", "A", "B"), ("pub_q_0243", "A", "B"), ("pub_q_0266", "A", "B"), ("pub_q_0364", "A", "B"),  # Public：依讀數改成 B 的 4 題
      ("pub_q_0066", "5D", "4D"), ("pub_q_0077", "5D", "4D"), ("pub_q_0157", "A", "2D")]),  # Public：其餘 3 題，同樣依 Public 讀數修正
    ("ct117", "ct116", "M211 / M212 (2026-11-17)",  # 第二步：ct116 -> ct117（只改 Public）
     "Public only: 10 answers whose solved truth weight was >= 0.95 and that at least two earlier uploads had changed, "  # 決定的內容（前半）
     "plus pub_q_0103 -> B and pub_q_0157 -> 4D",  # 決定的內容（後半）
     [("pub_q_0013", "A", "B"), ("pub_q_0036", "C", "D"), ("pub_q_0047", "1D", "3D"), ("pub_q_0110", "A", "1D"),  # 真值權重 >= 0.95 的 10 題之第 1-4 題
      ("pub_q_0142", "4D", "3D"), ("pub_q_0286", "C", "A"), ("pub_q_0290", "1D", "2D"), ("pub_q_0346", "A", "6D"),  # 同一批 10 題之第 5-8 題
      ("pub_q_0378", "2D", "1D"), ("pub_q_0388", "1D", "A"), ("pub_q_0103", "A", "B"), ("pub_q_0157", "2D", "4D")]),  # 同一批 10 題之第 9、10 題，再加另改的 pub_q_0103 與 pub_q_0157
    ("ct118d", "ct117", "M220 (2026-11-18)",  # 第三步：ct117 -> ct118d（只改 Public）
     "Public only: the questions of probe_C2B after the local exact update on its reading (+2 exact, -1 within one)",  # 決定的內容
     [("pub_q_0114", "C", "B"), ("pub_q_0125", "C", "B"), ("pub_q_0217", "C", "B"), ("pub_q_0359", "C", "B"),  # C 改 B 的 4 題
      ("pub_q_0162", "C", "D"), ("pub_q_0192", "C", "D")]),  # C 改 D 的 2 題
    ("ct118e", "ct118d", "M223 (2026-11-18)",  # 第四步：ct118d -> ct118e（只改 Public）
     "Public only: three answers set to the largest expected score after the reading of ct118d",  # 決定的內容
     [("pub_q_0286", "A", "C"), ("pub_q_0192", "D", "B"), ("pub_q_0103", "B", "A")]),  # 改動的 3 題
    ("ct118e_a2b4", "ct118e", "final file (2026-11-23)",  # 第五步：ct118e -> 最終檔（只改 Private）
     "Private: 'never leave A' relaxed on the other four questions where the pooled posterior answers B "  # 決定的內容（前半）
     "(participant's decision; scripts/build_ct118e_a2b4.py)",  # 決定的內容（後半）
     [("priv_q_0062", "A", "B"), ("priv_q_0125", "A", "B"), ("priv_q_0204", "A", "B"), ("priv_q_0325", "A", "B")]),  # 由 A 改成 B 的 4 題
]  # 結束步驟清單

FAILED = []  # 失敗的檢查；最後依它決定結束碼


def check(ok, what, show=True):  # 記錄並印出一項檢查
    if show or not ok:  # 要顯示或失敗時才印
        print(f"  [{'ok' if ok else 'FAIL'}] {what}")  # 印出結果與說明
    if not ok:  # 檢查失敗
        FAILED.append(what)  # 記下，最後決定結束碼
    return ok  # 回傳結果，方便串接


def md5_ok(name, digest):  # md5 是否符合登記值
    return digest.startswith(MD5[name])  # 以前綴比對：只登記前 12 碼的檔比前 12 碼，登記完整 md5 的檔等於整串比對


def parse(raw, label, show=True):  # 檢查格式並解析提交檔
    """Rows of a submission file; every format rule of the competition file is checked (printed only on failure if not show)."""
    good = check(not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n"),  # 沒有 BOM、沒有 CR、結尾有換行
                 f"{label}: UTF-8 without BOM, LF line ends, final newline", show)  # 這項檢查的說明
    lines = raw.decode("utf-8").split("\n")[:-1]  # 拆成各列，去掉最後換行之後的空字串
    good &= check(lines[0] == HEADER, f"{label}: header '{HEADER}'", show)  # 標題列正確
    body = [ln.split(",") for ln in lines[1:]]  # 答案列拆成欄位
    good &= check(len(body) == 800 and all(len(r) == 2 for r in body), f"{label}: 800 rows of two fields", show)  # 800 列，每列兩欄
    ids = [r[0] for r in body]  # 題號
    good &= check(len(set(ids)) == 800 and sum(q.startswith("pub_q_") for q in ids) == 400  # 題號不重複、Public 400 題
                  and sum(q.startswith("priv_q_") for q in ids) == 400, f"{label}: 400 Public + 400 Private, no repeats", show)  # Private 400 題
    good &= check(all(r[1] in LEVELS for r in body), f"{label}: every answer is one of the 10 levels", show)  # 每個答案都是合法的等級
    if not good:  # 格式不合
        sys.exit(f"{label} is not a valid submission file - stop")  # 無法再核對，直接結束（結束碼 1）
    return body  # 回傳 [題號, 答案] 的列表


def render(rows):  # 各列寫回提交檔的位元組
    return ("\n".join([HEADER] + [f"{q},{a}" for q, a in rows]) + "\n").encode("utf-8")  # 標題列加各列，以 LF 分隔並以換行結尾


def apply(rows, edits, forward, label):  # 套用或還原一個步驟的改動
    """Change the listed answers; every listed question must hold the expected answer before the change."""
    pos = {q: i for i, (q, _) in enumerate(rows)}  # 題號 -> 列的位置
    out = [list(r) for r in rows]  # 可修改的副本
    bad = []  # 答案不符預期的題目
    for q, before, after in edits:  # 逐題處理
        have, want = (before, after) if forward else (after, before)  # 正向由改前到改後，反向則相反
        if q not in pos or out[pos[q]][1] != have:  # 題目不存在，或目前的答案不是預期值
            bad.append(f"{q} holds {out[pos[q]][1] if q in pos else 'nothing'}, expected {have}")  # 記下實際與預期的答案
            continue  # 不改這一題
        out[pos[q]][1] = want  # 改成目標答案
    check(not bad, f"{label}: the {len(edits)} listed questions hold the expected answers" + (f" ({'; '.join(bad)})" if bad else ""))  # 列出的每一題都必須符合預期
    return [tuple(r) for r in out]  # 回傳新的各列


def report(name, raw, write_dir):  # 核對一個檔的 md5，需要時寫出
    d = hashlib.md5(raw).hexdigest()  # 算 md5
    check(md5_ok(name, d), f"{name}: md5 {d} {'matches' if md5_ok(name, d) else 'DIFFERS FROM'} the registered {MD5[name]}"  # md5 必須符合登記值
          f" (Public {PUBLIC[name]})")  # 附上 Public 讀數
    if write_dir:  # 要寫出中間檔
        f = write_dir / f"submission_rank_{name}.csv"  # 輸出檔名
        if f.exists():  # 已經存在
            print(f"    {f} exists - not overwritten")  # 不覆蓋
        else:  # 還不存在
            f.write_bytes(raw)  # 寫出
            print(f"    written {f}")  # 回報
    return d  # 回傳 md5


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("final", help="the final file, submission_rank_ct118e_a2b4.csv")  # 要核對的最終檔
    for name in ("ct112", "ct116", "ct117", "ct118d", "ct118e"):  # 每個中間檔都可以另給一份
        ap.add_argument(f"--{name}", type=Path, default=None, help=f"a stored or rebuilt copy of {name} to compare")  # 保存或重建的副本
    ap.add_argument("--write", type=Path, default=None, help="write the rebuilt intermediate files here (never overwrites)")  # 寫出重建的中間檔
    a = ap.parse_args()  # 解析參數
    if a.write:  # 要寫出中間檔
        a.write.mkdir(parents=True, exist_ok=True)  # 先建立目錄

    print(f"== the final file {a.final}")  # 第一段：最終檔本身
    raw = Path(a.final).read_bytes()  # 讀原始位元組
    final = parse(raw, "final file")  # 檢查格式並解析
    check(render(final) == raw, "final file: rewriting its rows gives the same bytes (no stray spaces or quotes)")  # 重寫後必須逐位元組相同
    report("ct118e_a2b4", raw, None)  # 核對最終檔的 md5
    order = [q for q, _ in final]  # 最終檔的題目順序

    print("\n== backward: undo the steps from the final file down to ct112")  # 第二段：反向還原
    rows, built = final, {"ct118e_a2b4": raw}  # 目前的各列與已重建的檔
    for name, src, entry, _, edits in reversed(STEPS):  # 從最後一步往回
        rows = apply(rows, edits, False, f"undo {name} -> {src}")  # 還原這一步
        built[src] = render(rows)  # 得到來源檔的位元組
        report(src, built[src], a.write)  # 核對 md5，需要時寫出

    print("\n== forward: apply the steps from ct112 to the final file")  # 第三段：正向套用
    if a.ct112:  # 用 --ct112 另給了一份 ct112（例如由模型輸出獨立重建的）
        print(f"  ct112 taken from {a.ct112}")  # 回報來源
        start = Path(a.ct112).read_bytes()  # 讀檔
        rows = parse(start, "given ct112")  # 檢查格式並解析
        check([q for q, _ in rows] == order, "given ct112: same question order as the final file")  # 題目順序必須與最終檔相同
        check(start == built["ct112"], "given ct112: identical to the ct112 rebuilt by the backward pass")  # 必須與反向還原的 ct112 相同
        report("ct112", start, None)  # 核對 md5
    else:  # 沒給就用反向還原的那份
        print("  ct112 taken from the backward pass (pass --ct112 FILE to start from an independently rebuilt one)")  # 回報來源
        rows = parse(built["ct112"], "rebuilt ct112", False)  # 解析（只在失敗時印出）
    for name, src, entry, why, edits in STEPS:  # 依序套用每一步
        n_pub = sum(q.startswith("pub_q_") for q, _, _ in edits)  # 這一步改了幾題 Public
        print(f"  -- {name} = {src} + {len(edits)} changes (Public {n_pub}, Private {len(edits) - n_pub}); {entry}: {why}")  # 說明這一步
        rows = apply(rows, edits, True, f"{src} -> {name}")  # 套用改動
        out = render(rows)  # 得到這一步的檔
        report(name, out, None)  # 核對 md5
        check(out == built[name], f"{name}: identical to the backward pass")  # 必須與反向還原的相同

    print("\n== what the final file changes against ct112")  # 第四段：最終檔相對 ct112 的改動
    base = dict(parse(built["ct112"], "rebuilt ct112", False))  # ct112 的答案
    diff = [(q, base[q], a_) for q, a_ in final if base[q] != a_]  # 答案不同的題目
    pub = [d for d in diff if d[0].startswith("pub_q_")]  # Public 的改動
    priv = [d for d in diff if d[0].startswith("priv_q_")]  # Private 的改動
    print(f"  Public {len(pub)} answers (all from Public readings): " + ", ".join(f"{q} {x}->{y}" for q, x, y in pub))  # 列出 Public 的改動
    print(f"  Private {len(priv)} answers (all participant's decisions): " + ", ".join(f"{q} {x}->{y}" for q, x, y in priv))  # 列出 Private 的改動
    want_priv = {q for s in STEPS for q, _, _ in s[4] if q.startswith("priv_q_")}  # STEPS 列出的 Private 題目
    check({q for q, _, _ in priv} == want_priv and len(priv) == 12, "Private: exactly the 12 listed questions differ from ct112")  # Private 恰好就是列出的 12 題
    e118 = dict(parse(built["ct118e"], "rebuilt ct118e", False))  # 重建的 ct118e
    check(all(e118[q] == x for q, x in final if q.startswith("pub_q_")), "Public: all 400 answers identical to ct118e")  # Public 400 題與 ct118e 完全相同

    for name in ("ct112", "ct116", "ct117", "ct118d", "ct118e"):  # 另外給的保存副本
        f = getattr(a, name)  # 對應的參數
        if f and name != "ct112":  # ct112 已在正向那一段處理
            print(f"\n== stored copy of {name}: {f}")  # 回報
            got = Path(f).read_bytes()  # 讀檔
            parse(got, f"stored {name}")  # 檢查格式
            check(got == built[name], f"stored {name}: identical to the rebuilt one")  # 必須與重建的相同
            report(name, got, None)  # 核對 md5

    print("\n== result: " + ("every check passed" if not FAILED else f"{len(FAILED)} check(s) FAILED:\n  - " + "\n  - ".join(FAILED)))  # 總結
    return 0 if not FAILED else 1  # 全部通過為 0，否則為 1


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
