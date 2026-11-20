"""Leak audit of the mixed-player training caches (section 4L.4): for each validation fold v,
 (1) the validation features and questions of r_val{v}_mix50 equal the plain r_val{v} cache (same evaluation set);
 (2) no game of a training question (mixed or not) belongs to a validation-fold player's scored selection;
 (3) every game of a mixed question belongs to a training-fold player of the SAME rank, players distinct, not the originating player;
 (4) no inner held-out player's game appears in any question of a non-held player (held players' games never train);
 (5) the inner set (held players' unmixed questions) holds only the held players' own games."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.l1bcv import players_from_scores  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
owner, prank, pfold = {}, {}, {}
for f in ts.FOLDS:
    hsl = read_scored_games(ts.HSL / f"fold{f}")
    for r in players_from_scores(hsl, folds_df).itertuples():
        prank[r.player_id], pfold[r.player_id] = r.rank, f
        for k in r.game_keys:
            owner[k] = r.player_id
rep = {}
for v in ts.FOLDS:
    plain, mix = (json.load(open(ROOT / f"results/trunk_seq/{t}.json")) for t in (f"r_val{v}", f"r_val{v}_mix50"))
    zp, zm = (np.load(ROOT / f"results/trunk_seq/{t}.npz") for t in (f"r_val{v}", f"r_val{v}_mix50"))
    same_val = (plain["val"] == mix["val"]) and np.array_equal(zp["X_va"], zm["X_va"]) and np.array_equal(zp["y_va"], zm["y_va"])
    tr = pd.DataFrame(mix["train"])
    val_players = {p for p, f in pfold.items() if f == v}
    train_folds = [f for f in ts.FOLDS if f != v]
    players = np.array(sorted(set(tr["player_id"])))
    held = set(np.random.RandomState(12345 + ts.fold_code(v)).permutation(players)[:int(0.1 * len(players))])
    bad_val, bad_rank, bad_dup, bad_self, bad_fold, held_in_train, inner_bad, n_mixed, n_games = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for row in tr.itertuples():
        owners = [owner.get(k) for k in row.game_keys]
        n_games += len(owners)
        bad_val += sum(o in val_players or o is None for o in owners)
        is_held = row.player_id in held
        if not is_held:
            held_in_train += sum(o in held for o in owners)
        if row.mixed:
            n_mixed += 1
            bad_rank += sum(prank.get(o) != row.rank for o in owners)
            bad_dup += len(owners) - len(set(owners))
            bad_self += sum(o == row.player_id for o in owners)
            bad_fold += sum(pfold.get(o) not in train_folds for o in owners)
        elif is_held:
            inner_bad += sum(o != row.player_id for o in owners)
    rep[v] = {"same_validation_set": bool(same_val), "train_questions": len(tr), "mixed": n_mixed, "held_players": len(held), "games_in_training_questions": n_games,
              "games_of_validation_players_or_unknown": bad_val, "mixed_games_other_rank": bad_rank, "mixed_games_duplicate_player": bad_dup,
              "mixed_games_own_player": bad_self, "mixed_games_outside_training_folds": bad_fold, "held_players_games_in_nonheld_questions": held_in_train,
              "inner_questions_with_foreign_games": inner_bad}
    print(v, rep[v], flush=True)
ok = all(r["same_validation_set"] and r["games_of_validation_players_or_unknown"] == 0 and r["mixed_games_other_rank"] == 0 and r["mixed_games_duplicate_player"] == 0
         and r["mixed_games_own_player"] == 0 and r["mixed_games_outside_training_folds"] == 0 and r["held_players_games_in_nonheld_questions"] == 0
         and r["inner_questions_with_foreign_games"] == 0 for r in rep.values())
print("LEAK AUDIT", "PASS" if ok else "FAIL")
Path(ROOT / "results/trunk_seq/es/mix_leak_audit.json").write_text(json.dumps({"folds": rep, "pass": ok}, indent=1))
