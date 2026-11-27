"""Do the rank-prediction test games appear in competition 1's official candidate pools (400 players x 100 games, public and private), and are
the pool player ids the training player ids? Exact move-sequence hashes (as logs/nano4/dup_scan.py). If test games sit in a pool under a
player id that is also a training player, the chain test game -> pool player -> training label is exact.
  python -I logs/nano4/cross_task_overlap.py --out results/r_variants/cross_task_overlap.json
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd(); R2 = Path.home() / "BK_AICUP_2026-2"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.scoring import TRAIN_DIR  # noqa: E402
import predict_submission as ps  # noqa: E402


def h(sgf):
    return hashlib.sha1("".join(re.findall(r";([BW]\[[a-z]{0,2}\])", sgf)).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "rank"]) for p in sorted(Path(TRAIN_DIR).glob("train_*.csv"))], ignore_index=True)
    train_label = rows.drop_duplicates("player_id").set_index("player_id")["rank"]; train_players = set(train_label.index)
    rep = {}
    pools = {}
    for name in ("public", "private"):
        f = R2 / "data/tests" / f"player_identification_candidates_{name}.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f, dtype=str, keep_default_na=False); c["h"] = c.sgf_content.map(h); c["pid_core"] = c.player_id.str.replace(r"^(pub|priv)_p_", "", regex=True)
        pools[name] = c
        in_train = c.drop_duplicates("player_id").pid_core.isin(train_players)
        rep[f"pool_{name}"] = {"games": int(len(c)), "players": int(c.player_id.nunique()), "players_whose_core_id_is_a_training_player": int(in_train.sum())}
        print(f"pool {name}: {len(c)} games, {c.player_id.nunique()} players; core ids found among training players: {int(in_train.sum())}", flush=True)
    # identification test questions' games too
    qgames = {}
    for name in ("public", "private"):
        f = R2 / "data/tests" / f"player_identification_test_{name}.csv"
        if f.exists():
            q = pd.read_csv(f, dtype=str, keep_default_na=False)
            for _, r in q.iterrows():
                for i in range(1, 21):
                    s = r.get(f"sgf_{i}", "")
                    if s:
                        qgames[h(s)] = r["question_id"]
    rep["pid_test_games"] = len(qgames)
    questions, _ = ps.load_questions(ps.DEFAULT_INPUTS); tasks, owner, _, _ = ps.build_tasks(questions)
    th = [h(sgf) for _, sgf, _ in tasks]
    allpool = pd.concat(pools.values(), ignore_index=True) if pools else pd.DataFrame(columns=["h", "player_id", "pid_core"])
    pool_map = allpool.drop_duplicates("h").set_index("h")
    hits = [x in pool_map.index for x in th]; hits_q = [x in qgames for x in th]
    rep["rank_test"] = {"games": len(th), "games_in_pid_pools": int(sum(hits)), "games_in_pid_test_questions": int(sum(hits_q))}
    print(f"rank-prediction test: {len(th)} games; {sum(hits)} appear in the identification candidate pools, {sum(hits_q)} in the identification test questions", flush=True)
    # per rank-test question: pool player(s) matched and their training label
    per_q = {}
    for (k, sgf, col), qi, x, hit in zip(tasks, owner, th, hits):
        qid = questions[qi]["question_id"]
        if hit:
            pid = pool_map.loc[x, "player_id"]; core = pool_map.loc[x, "pid_core"]
            per_q.setdefault(qid, []).append((pid, core, train_label.get(core, None)))
    summ = {"questions_with_pool_hits": len(per_q), "questions_all_games_one_pool_player": 0, "questions_with_training_label": 0}
    recs = []
    for qid, lst in per_q.items():
        pids = {p for p, _, _ in lst}; labs = {l for _, _, l in lst if l is not None}
        if len(pids) == 1:
            summ["questions_all_games_one_pool_player"] += 1
        if labs:
            summ["questions_with_training_label"] += 1
        recs.append({"question_id": qid, "games_hit": len(lst), "pool_players": "|".join(sorted(pids)), "training_label": "|".join(sorted(labs))})
    rep["rank_test_questions"] = summ
    pd.DataFrame(recs).to_csv(ROOT / "results/r_variants/cross_task_overlap_questions.csv", index=False)
    print(f"rank-test questions with pool hits: {summ['questions_with_pool_hits']}; all hits one pool player: {summ['questions_all_games_one_pool_player']}; with a training label via the pool id: {summ['questions_with_training_label']}", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str)); print("written", a.out)


if __name__ == "__main__":
    main()
