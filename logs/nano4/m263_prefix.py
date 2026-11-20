#!/usr/bin/env python
"""M263 (2026-11-20), report only: games that share their first 40 / 60 moves but are not the same text (the same game recorded twice
with different endings?). Training rows against each other, test games against training rows, test games against each other.
(design doc M263, registered before any reading)

  python -I logs/nano4/m263_prefix.py
"""
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path.cwd(); MOVE = re.compile(r";[BW]\[[a-z]*\]"); LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def prefixes(sgf, ks=(40, 60)):
    mv = MOVE.findall(sgf); return len(mv), {k: hashlib.blake2b("".join(mv[:k]).encode(), digest_size=8).digest() for k in ks if len(mv) >= k}


def main():
    csv.field_size_limit(10 ** 9); train = {40: defaultdict(list), 60: defaultdict(list)}; meta = []; full = {}
    for r in LEVELS:
        with open(ROOT / f"data/training/train_{r}.csv") as fh:
            for row in csv.DictReader(fh):
                n, pf = prefixes(row["sgf_content"]); i = len(meta); meta.append((f"{r}:{row['game_id']}", row["player_id"], row["color"], r, n, hashlib.blake2b(row["sgf_content"].encode(), digest_size=8).digest()))
                for k, h in pf.items():
                    train[k][h].append(i)
        print(f"train_{r}: {len(meta)} rows so far", flush=True)
    rep = {"rows": len(meta)}
    for k in (40, 60):
        groups = [g for g in train[k].values() if len(g) >= 2]; same_txt = sum(1 for g in groups if len({meta[i][5] for i in g}) < len(g)); diff_owner = [g for g in groups if len({meta[i][1] for i in g}) >= 2]; same_owner = [g for g in groups if len({meta[i][1] for i in g}) == 1]
        print(f"\n== (a) training rows sharing their first {k} moves: {len(groups)} groups ({sum(len(g) for g in groups)} rows); groups with identical full text {same_txt}; groups of ONE owner {len(same_owner)}, of different owners {len(diff_owner)}")
        for g in sorted(diff_owner, key=len, reverse=True)[:12]:
            print("      " + " | ".join(f"{meta[i][0]} owner {meta[i][1][:8]} {meta[i][2]} {meta[i][3]} {meta[i][4]} moves" for i in g[:4]) + (f" ... (+{len(g) - 4})" if len(g) > 4 else ""))
        for g in sorted(same_owner, key=len, reverse=True)[:5]:
            print("      one owner: " + " | ".join(f"{meta[i][0]} {meta[i][2]} {meta[i][3]} {meta[i][4]} moves" for i in g[:4]))
        rep[f"train_{k}"] = {"groups": len(groups), "same_owner": len(same_owner), "different_owners": len(diff_owner), "rows": sum(len(g) for g in groups)}
    test = []; tt = {40: defaultdict(list), 60: defaultdict(list)}
    for half in ("public", "private"):
        with open(ROOT / f"data/tests/rank_prediction_test_{half}.csv") as fh:
            for row in csv.DictReader(fh):
                for j in range(1, int(row["num_games"]) + 1):
                    n, pf = prefixes(row[f"sgf_{j}"]); i = len(test); test.append((row["question_id"], j, row[f"color_{j}"], n))
                    for k, h in pf.items():
                        tt[k][h].append(i)
    for k in (40, 60):
        hit = [(i, train[k][h]) for h, g in tt[k].items() for i in g if h in train[k]]; grp = [g for g in tt[k].values() if len(g) >= 2]; cross = [g for g in grp if len({test[i][0] for i in g}) >= 2]
        print(f"\n== (b) test games whose first {k} moves equal a training row's: {len(hit)} of {len(test)} test games; (c) test games sharing their first {k} moves with another test game: {len(grp)} groups ({len(cross)} across different questions)")
        for i, g in hit[:20]:
            print(f"      {test[i][0]} game {test[i][1]} ({test[i][2]}, {test[i][3]} moves) == " + " | ".join(f"{meta[j][0]} owner {meta[j][1][:8]} {meta[j][2]} {meta[j][3]} {meta[j][4]} moves" for j in g[:3]))
        for g in grp[:10]:
            print("      test: " + " | ".join(f"{test[i][0]} game {test[i][1]} ({test[i][2]}, {test[i][3]} moves)" for i in g[:4]))
        rep[f"test_vs_train_{k}"] = len(hit); rep[f"test_groups_{k}"] = len(grp)
    b40, a60 = rep["test_vs_train_40"], rep["train_60"]["different_owners"]; verdict = "worth a formal source (registered separately)" if (b40 >= 20 or rep["train_40"]["different_owners"] >= 1000) else ("closed: no such duplicates in the data" if (b40 == 0 and rep["train_40"]["different_owners"] < 100) else "in between: listed, judged from the list")
    print(f"\nREGISTERED: test vs training (40 moves) {b40}; training groups of different owners (40 moves) {rep['train_40']['different_owners']}, (60 moves) {a60}  ->  {verdict}"); rep["verdict"] = verdict
    (ROOT / "results/r_variants/m263_prefix.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m263_prefix.json")


if __name__ == "__main__":
    main()
