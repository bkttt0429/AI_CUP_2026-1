#!/usr/bin/env python3
"""Check, byte for byte, the chain from the model output ct112 to the final file ct118e_a2b4 (standard library only).

Every answer that differs between ct112 and the final file is listed in STEPS below: question, answer before, answer
after, the step that changed it and the record entry (M### = reports/architecture_v2_2026-11-02.md).

  backward  start from the final file and undo the steps one by one; every intermediate file must have the md5 that was
            registered when it was built (ct118e, ct118d, ct117, ct116, ct112).
  forward   start from ct112 -- the one rebuilt by the backward pass, or --ct112 FILE, for example the file rebuilt from
            the stored model outputs with
              python -I logs/nano4/m163_pool.py --new SQ4 --control SPALm --name ctverify
            (needs the data and results/ of the local backup in the repo layout) -- and apply the steps; every file on
            the way, and the end result, must have its registered md5.
  --ct116 / --ct117 / --ct118d / --ct118e FILE  also compare stored copies of those files with the rebuilt ones.
  --write DIR   write the rebuilt intermediate files to DIR (never overwrites).

  python3 -I scripts/verify_final_chain.py submission_rank_ct118e_a2b4.csv [--ct112 FILE] [--ct116 FILE] [--ct118e FILE]

Exit status 0 = every check passed; 1 = a check failed (the failing check is printed).
"""
import argparse
import hashlib
import sys
from pathlib import Path

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
HEADER = "question_id,pred_rank"

# Registered md5 of each file when it was built (full where recorded, else the first 12 hex digits) and its Public reading.
MD5 = {
    "ct112": "7d99d6f2c5b7",
    "ct116": "6b053db6672cd31cbb2e6b742de18649",
    "ct117": "feaaca4f827a",
    "ct118d": "12165c64bcfa",
    "ct118e": "0ef7368fb85cf1eecde2eb18b99367cb",
    "ct118e_a2b4": "2c2e4ad637a6407a35de7f599c4ac641",
}
PUBLIC = {"ct112": "0.8331743", "ct116": "0.8394955", "ct117": "never uploaded", "ct118d": "0.8546379",
          "ct118e": "0.8587182", "ct118e_a2b4": "0.8587182"}

# (file built, built from, record entry, what was decided, [(question, answer before, answer after), ...])
STEPS = [
    ("ct116", "ct112", "M209 (2026-11-17)",
     "Private: the 4D -> 5D push unlocked (6 questions back to 4D) and 'never leave A' relaxed on the two questions with "
     "the highest B probability (participant's decision); Public: 7 answers from the decoded Public readings",
     [("priv_q_0039", "5D", "4D"), ("priv_q_0144", "5D", "4D"), ("priv_q_0162", "5D", "4D"),
      ("priv_q_0167", "5D", "4D"), ("priv_q_0187", "5D", "4D"), ("priv_q_0274", "5D", "4D"),
      ("priv_q_0084", "A", "B"), ("priv_q_0087", "A", "B"),
      ("pub_q_0109", "A", "B"), ("pub_q_0243", "A", "B"), ("pub_q_0266", "A", "B"), ("pub_q_0364", "A", "B"),
      ("pub_q_0066", "5D", "4D"), ("pub_q_0077", "5D", "4D"), ("pub_q_0157", "A", "2D")]),
    ("ct117", "ct116", "M211 / M212 (2026-11-17)",
     "Public only: 10 answers whose solved truth weight was >= 0.95 and that at least two earlier uploads had changed, "
     "plus pub_q_0103 -> B and pub_q_0157 -> 4D",
     [("pub_q_0013", "A", "B"), ("pub_q_0036", "C", "D"), ("pub_q_0047", "1D", "3D"), ("pub_q_0110", "A", "1D"),
      ("pub_q_0142", "4D", "3D"), ("pub_q_0286", "C", "A"), ("pub_q_0290", "1D", "2D"), ("pub_q_0346", "A", "6D"),
      ("pub_q_0378", "2D", "1D"), ("pub_q_0388", "1D", "A"), ("pub_q_0103", "A", "B"), ("pub_q_0157", "2D", "4D")]),
    ("ct118d", "ct117", "M220 (2026-11-18)",
     "Public only: the questions of probe_C2B after the local exact update on its reading (+2 exact, -1 within one)",
     [("pub_q_0114", "C", "B"), ("pub_q_0125", "C", "B"), ("pub_q_0217", "C", "B"), ("pub_q_0359", "C", "B"),
      ("pub_q_0162", "C", "D"), ("pub_q_0192", "C", "D")]),
    ("ct118e", "ct118d", "M223 (2026-11-18)",
     "Public only: three answers set to the largest expected score after the reading of ct118d",
     [("pub_q_0286", "A", "C"), ("pub_q_0192", "D", "B"), ("pub_q_0103", "B", "A")]),
    ("ct118e_a2b4", "ct118e", "final file (2026-11-23)",
     "Private: 'never leave A' relaxed on the other four questions where the pooled posterior answers B "
     "(participant's decision; scripts/build_ct118e_a2b4.py)",
     [("priv_q_0062", "A", "B"), ("priv_q_0125", "A", "B"), ("priv_q_0204", "A", "B"), ("priv_q_0325", "A", "B")]),
]

FAILED = []


def check(ok, what, show=True):
    if show or not ok:
        print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILED.append(what)
    return ok


def md5_ok(name, digest):
    return digest.startswith(MD5[name])


def parse(raw, label, show=True):
    """Rows of a submission file; every format rule of the competition file is checked (printed only on failure if not show)."""
    good = check(not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n"),
                 f"{label}: UTF-8 without BOM, LF line ends, final newline", show)
    lines = raw.decode("utf-8").split("\n")[:-1]
    good &= check(lines[0] == HEADER, f"{label}: header '{HEADER}'", show)
    body = [ln.split(",") for ln in lines[1:]]
    good &= check(len(body) == 800 and all(len(r) == 2 for r in body), f"{label}: 800 rows of two fields", show)
    ids = [r[0] for r in body]
    good &= check(len(set(ids)) == 800 and sum(q.startswith("pub_q_") for q in ids) == 400
                  and sum(q.startswith("priv_q_") for q in ids) == 400, f"{label}: 400 Public + 400 Private, no repeats", show)
    good &= check(all(r[1] in LEVELS for r in body), f"{label}: every answer is one of the 10 levels", show)
    if not good:
        sys.exit(f"{label} is not a valid submission file - stop")
    return body


def render(rows):
    return ("\n".join([HEADER] + [f"{q},{a}" for q, a in rows]) + "\n").encode("utf-8")


def apply(rows, edits, forward, label):
    """Change the listed answers; every listed question must hold the expected answer before the change."""
    pos = {q: i for i, (q, _) in enumerate(rows)}
    out = [list(r) for r in rows]
    bad = []
    for q, before, after in edits:
        have, want = (before, after) if forward else (after, before)
        if q not in pos or out[pos[q]][1] != have:
            bad.append(f"{q} holds {out[pos[q]][1] if q in pos else 'nothing'}, expected {have}")
            continue
        out[pos[q]][1] = want
    check(not bad, f"{label}: the {len(edits)} listed questions hold the expected answers" + (f" ({'; '.join(bad)})" if bad else ""))
    return [tuple(r) for r in out]


def report(name, raw, write_dir):
    d = hashlib.md5(raw).hexdigest()
    check(md5_ok(name, d), f"{name}: md5 {d} {'matches' if md5_ok(name, d) else 'DIFFERS FROM'} the registered {MD5[name]}"
          f" (Public {PUBLIC[name]})")
    if write_dir:
        f = write_dir / f"submission_rank_{name}.csv"
        if f.exists():
            print(f"    {f} exists - not overwritten")
        else:
            f.write_bytes(raw)
            print(f"    written {f}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("final", help="the final file, submission_rank_ct118e_a2b4.csv")
    for name in ("ct112", "ct116", "ct117", "ct118d", "ct118e"):
        ap.add_argument(f"--{name}", type=Path, default=None, help=f"a stored or rebuilt copy of {name} to compare")
    ap.add_argument("--write", type=Path, default=None, help="write the rebuilt intermediate files here (never overwrites)")
    a = ap.parse_args()
    if a.write:
        a.write.mkdir(parents=True, exist_ok=True)

    print(f"== the final file {a.final}")
    raw = Path(a.final).read_bytes()
    final = parse(raw, "final file")
    check(render(final) == raw, "final file: rewriting its rows gives the same bytes (no stray spaces or quotes)")
    report("ct118e_a2b4", raw, None)
    order = [q for q, _ in final]

    print("\n== backward: undo the steps from the final file down to ct112")
    rows, built = final, {"ct118e_a2b4": raw}
    for name, src, entry, _, edits in reversed(STEPS):
        rows = apply(rows, edits, False, f"undo {name} -> {src}")
        built[src] = render(rows)
        report(src, built[src], a.write)

    print("\n== forward: apply the steps from ct112 to the final file")
    if a.ct112:
        print(f"  ct112 taken from {a.ct112}")
        start = Path(a.ct112).read_bytes()
        rows = parse(start, "given ct112")
        check([q for q, _ in rows] == order, "given ct112: same question order as the final file")
        check(start == built["ct112"], "given ct112: identical to the ct112 rebuilt by the backward pass")
        report("ct112", start, None)
    else:
        print("  ct112 taken from the backward pass (pass --ct112 FILE to start from an independently rebuilt one)")
        rows = parse(built["ct112"], "rebuilt ct112", False)
    for name, src, entry, why, edits in STEPS:
        n_pub = sum(q.startswith("pub_q_") for q, _, _ in edits)
        print(f"  -- {name} = {src} + {len(edits)} changes (Public {n_pub}, Private {len(edits) - n_pub}); {entry}: {why}")
        rows = apply(rows, edits, True, f"{src} -> {name}")
        out = render(rows)
        report(name, out, None)
        check(out == built[name], f"{name}: identical to the backward pass")

    print("\n== what the final file changes against ct112")
    base = dict(parse(built["ct112"], "rebuilt ct112", False))
    diff = [(q, base[q], a_) for q, a_ in final if base[q] != a_]
    pub = [d for d in diff if d[0].startswith("pub_q_")]
    priv = [d for d in diff if d[0].startswith("priv_q_")]
    print(f"  Public {len(pub)} answers (all from Public readings): " + ", ".join(f"{q} {x}->{y}" for q, x, y in pub))
    print(f"  Private {len(priv)} answers (all participant's decisions): " + ", ".join(f"{q} {x}->{y}" for q, x, y in priv))
    want_priv = {q for s in STEPS for q, _, _ in s[4] if q.startswith("priv_q_")}
    check({q for q, _, _ in priv} == want_priv and len(priv) == 12, "Private: exactly the 12 listed questions differ from ct112")
    e118 = dict(parse(built["ct118e"], "rebuilt ct118e", False))
    check(all(e118[q] == x for q, x in final if q.startswith("pub_q_")), "Public: all 400 answers identical to ct118e")

    for name in ("ct112", "ct116", "ct117", "ct118d", "ct118e"):
        f = getattr(a, name)
        if f and name != "ct112":
            print(f"\n== stored copy of {name}: {f}")
            got = Path(f).read_bytes()
            parse(got, f"stored {name}")
            check(got == built[name], f"stored {name}: identical to the rebuilt one")
            report(name, got, None)

    print("\n== result: " + ("every check passed" if not FAILED else f"{len(FAILED)} check(s) FAILED:\n  - " + "\n  - ".join(FAILED)))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
