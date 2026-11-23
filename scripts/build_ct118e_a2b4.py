#!/usr/bin/env python3
"""Build the final file ct118e_a2b4 = ct118e + 4 Private A->B (participant's decision 2026-11-23: 「好 改那4題」).

This is the script that wrote submission_rank_ct118e_a2b4.csv (md5 2c2e4ad637a6407a35de7f599c4ac641) on 2026-11-23 02:45;
only this docstring was edited when it was added to the repo. The whole chain from ct112 is checked by
scripts/verify_final_chain.py.

Usage: python3 -I scripts/build_ct118e_a2b4.py <input ct118e csv> <output csv>
Checks: input md5 = ct118e; header; 800 rows (400 pub / 400 priv); LF only; the 4 targets are A in the input.
Output: same bytes except the 4 Private rows A -> B; verified by re-reading.
"""
import hashlib
import sys

CT118E_MD5 = "0ef7368fb85cf1eecde2eb18b99367cb"
TARGETS = ["priv_q_0062", "priv_q_0125", "priv_q_0204", "priv_q_0325"]
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def rows_of(raw):
    text = raw.decode("utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def main():
    src, dst = sys.argv[1], sys.argv[2]
    raw = open(src, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    print(f"input {src}: {len(raw)} bytes, md5 {md5} ({'= ct118e' if md5 == CT118E_MD5 else 'NOT ct118e'})")
    assert md5 == CT118E_MD5, "input is not ct118e - stop"
    assert b"\r" not in raw, "CR found"
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM found"
    lines = rows_of(raw)
    assert lines[0] == "question_id,pred_rank", lines[0]
    body = [ln.split(",") for ln in lines[1:]]
    assert len(body) == 800 and all(len(r) == 2 for r in body)
    ids = [r[0] for r in body]
    assert len(set(ids)) == 800
    assert sum(i.startswith("pub_q_") for i in ids) == 400 and sum(i.startswith("priv_q_") for i in ids) == 400
    assert all(r[1] in LEVELS for r in body)
    ans = dict(body)
    for q in TARGETS:
        print(f"  {q}: input answer {ans[q]}")
        assert ans[q] == "A", f"{q} is not A in the input"
    out_lines = [lines[0]] + [f"{q},{'B' if q in TARGETS else a}" for q, a in body]
    out = ("\n".join(out_lines) + ("\n" if raw.endswith(b"\n") else "")).encode("utf-8")
    with open(dst, "wb") as fh:
        fh.write(out)
    # verify by re-reading
    back = rows_of(open(dst, "rb").read())
    nb = [ln.split(",") for ln in back[1:]]
    assert back[0] == lines[0] and [r[0] for r in nb] == ids
    diff = [(q, a, b) for (q, a), (_, b) in zip(body, nb) if a != b]
    assert sorted(d[0] for d in diff) == sorted(TARGETS) and all(a == "A" and b == "B" for _, a, b in diff)
    assert all(a == b for (q, a), (_, b) in zip(body, nb) if q.startswith("pub_q_"))
    omd5 = hashlib.md5(open(dst, "rb").read()).hexdigest()
    print(f"output {dst}: {len(out)} bytes, md5 {omd5}")
    print(f"  changed rows: {len(diff)} (all Private, A -> B): " + ", ".join(d[0] for d in diff))
    print("  Public 400 rows identical to ct118e; order, header and line endings unchanged")


if __name__ == "__main__":
    main()
