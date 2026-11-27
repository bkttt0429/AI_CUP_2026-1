"""Prepare label-blind R1 questions and score them with the frozen November 16 behavior policy."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
R1 = HERE.parents[1]
PILOT = Path("/work/u2507115/behavior_probe_20261116")
HEAD = Path("/home/u2507115/.cache/pidbehavior_wt/scripts")
CHOICE = Path("/home/u2507115/.cache/pidchoice_wt/scripts")
sys.path.insert(0, str(HEAD))
from behavior_data import digest, immutable_json


def read(path):
    return json.loads(path.read_text())


def select_questions(config, pilot):
    selections, inputs = [], {}
    old_owners = {g["owner"] for g in pilot["rows"]}
    for fold in (config["development_fold"], config["evaluation_fold"]):
        root = R1 / "results/trunk_seq"
        questions_path = root / f"r_val{fold}.json"
        val = read(questions_path)["val"]
        inputs[str(questions_path)] = digest(questions_path)
        probabilities = []
        for arm in config["baseline"]:
            for seed in config["baseline_seeds"]:
                path = root / "es" / f"val{fold}" / f"{arm}_s{seed}.npz"
                inputs[str(path)] = digest(path)
                with np.load(path, allow_pickle=False) as z:
                    probabilities.append(z["proba"].astype(float))
        proba = np.mean(probabilities, axis=0)
        assert proba.shape == (len(val["player_id"]), 10) and np.allclose(proba.sum(1), 1, atol=1e-5)
        seen, counts = set(), Counter()
        def order(i):
            key = f"conditional-rank-nov18|{fold}|{i}|{val['player_id'][i]}"
            return hashlib.sha256(key.encode()).digest()
        for i in sorted(range(len(proba)), key=order):
            owner = val["player_id"][i]
            if val["n_games"][i] != 5 or owner in seen:
                continue
            first, second = np.argsort(-proba[i], kind="stable")[:2]
            low = int(min(first, second))
            if abs(first - second) != 1 or proba[i, first] > 2 * proba[i, second] or counts[low] == 12:
                continue
            assert owner not in old_owners and len(set(val["game_keys"][i])) == 5
            selections.append(dict(fold=fold, index=i, owner=owner, low=low, p=proba[i].tolist(),
                                   keys=val["game_keys"][i]))
            seen.add(owner)
            counts[low] += 1
        assert set(counts) == set(range(9)), counts
    assert len({q["owner"] for q in selections}) == len(selections)
    return selections, inputs


def prepare():
    config, pilot = read(HERE / "protocol.json"), read(PILOT / "protocol.json")
    queries, inputs = select_questions(config, pilot)
    inputs[str(HERE / "protocol.json")] = digest(HERE / "protocol.json")
    inputs[str(PILOT / "selection.json")] = digest(PILOT / "selection.json")
    inputs[str(PILOT / "protocol.json")] = digest(PILOT / "protocol.json")
    inputs[str(Path(__file__).resolve())] = digest(Path(__file__))
    immutable_json(HERE / "selection.json", dict(queries=queries, inputs=inputs))
    print("selected", Counter(q["fold"] for q in queries), flush=True)


def load_queries(queries, pilot):
    wanted = {key for q in queries for key in q["keys"]}
    games = {}
    for path in sorted((R1 / "data/training").glob("train_*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                key = r["rank"] + ":" + r["game_id"]
                if key in wanted:
                    assert key not in games
                    games[key] = dict(sgf=r["sgf_content"], color=r["color"], owner=r["player_id"],
                                      sha=hashlib.sha256(r["sgf_content"].encode()).hexdigest())
    assert set(games) == wanted
    hashes = Counter(games[k]["sha"] for q in queries for k in q["keys"])
    old = {g["sgf_sha256"] for g in pilot["rows"]}
    for q in queries:
        assert all(games[k]["owner"] == q["owner"] for k in q["keys"])
    excluded = {k: "duplicate SGF" for k, g in games.items() if hashes[g["sha"]] != 1 or g["sha"] in old}
    return games, excluded


def checked_replay(item):
    from pid_behavior_extract import replay
    key, game = item
    try:
        return key, replay(game), None
    except (ValueError, AssertionError, IndexError) as err:
        return key, None, f"{type(err).__name__}: {err}"


def rank_bank(pilot, device):
    import torch
    from pid_behavior_train import references
    from behavior_model import BehaviorPolicy
    owners = pilot["split"]["train"]
    rows = pilot["rows"]
    config = read(HERE / "protocol.json")
    members = [[] for _ in config["levels"]]
    data, groups = {}, []
    complete = read(PILOT / "extraction_complete.json")
    assert complete["protocol_sha256"] == digest(PILOT / "protocol.json")
    for j, owner in enumerate(owners):
        by_owner = [(i, g) for i, g in enumerate(rows) if g["owner"] == owner and g["role"] == "reference"]
        labels = {g["original_key"].split(":")[0] for _, g in by_owner}
        assert len(by_owner) == 7 and len(labels) == 1
        members[config["levels"].index(labels.pop())].append(j)
        groups.append([g["key"] for _, g in by_owner])
        for i, g in by_owner:
            file = PILOT / "games" / f"{i:04d}.npz"
            expected = complete["records"][i]
            assert expected["key"] == g["key"] and digest(file) == expected["sha256"]
            with np.load(file, allow_pickle=False) as z:
                data[g["key"]] = {k: torch.as_tensor(z[k], device=device)
                                  for k in ("state", "history", "ref_value", "played")}
    assert min(map(len, members)) >= 1
    memory = references(data, groups)
    weights = torch.zeros((len(owners), len(members)), device=device)
    for j, indexes in enumerate(members):
        weights[indexes, j] = 1 / len(indexes)
    selected = read(PILOT / "selection.json")
    networks = []
    for seed in config["behavior_seeds"]:
        checkpoint = PILOT / "models" / f"contextual_{seed}.pt"
        assert digest(checkpoint) == selected["checkpoint_sha256"][checkpoint.name]
        net = BehaviorPolicy("contextual").to(device)
        net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        networks.append(net.eval())
    return memory, weights, networks


def features(katago, replay, projection):
    import torch
    from pid_choice_extract import forward_choices
    x = forward_choices(katago, replay, projection, batch=32)
    phi = x["psi"].float()
    lp = x["logp"].float().log_softmax(-1)
    entropy = -(lp.exp() * lp.masked_fill(~torch.isfinite(lp), 0)).sum(1)
    scalar = torch.stack([x["value"], torch.as_tensor(replay["t"] / 256, device=lp.device),
                          torch.as_tensor(replay["mover"], device=lp.device), entropy], -1).float()
    state = torch.cat([phi[:, :361, :384].mean(1), scalar], -1)
    # The pilot stored candidate features as fp16 before the head consumed them.
    return (phi.half().float(), lp, state, torch.as_tensor(replay["history"], device=lp.device),
            torch.as_tensor(replay["played"], device=lp.device, dtype=torch.long))


def score_batch(queries, networks, memory, weights):
    import torch
    phi, lp, state, history, played = [torch.cat(fields) for fields in zip(*queries)]
    lengths = [len(q[-1]) for q in queries]
    ix = torch.arange(len(played), device=lp.device)
    neutral = [x.mean().item() for x in lp[ix, played].split(lengths)]
    rank_scores = []
    for net in networks:
        for condition in (None, "permute"):
            pred = net(phi, lp, state, history, *memory, ablation=condition)
            observed = pred.gather(2, played[:, None, None].expand(-1, weights.shape[0], 1)).squeeze(-1)
            rank_scores.append(torch.stack([x.mean(0) for x in (observed @ weights).split(lengths)]).cpu().numpy())
    return np.stack(rank_scores, axis=1).reshape(len(queries), len(networks), 2, 10), neutral


def run():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run inference in the Slurm allocation")
    import torch
    from pid_behavior_extract import init_worker
    sys.path.insert(0, str(CHOICE))
    from pid_choice_extract import forward_choices  # sets up the extractor's existing package paths
    from aicup.humansl import load_humansl
    assert callable(forward_choices)
    torch.set_num_threads(1)
    pilot, manifest = read(PILOT / "protocol.json"), read(HERE / "selection.json")
    for path, expected in manifest["inputs"].items():
        assert digest(Path(path)) == expected, path
    assert all(digest(Path(k)) == v for k, v in read(PILOT / "training_protocol.json")["code"].items())
    assert all(digest(Path(k)) == v for k, v in pilot["sources"].items())
    checkpoint = pilot["checkpoint"]["path"]
    assert digest(checkpoint) == pilot["checkpoint"]["sha256"]
    config = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]
    katago, _ = load_humansl(checkpoint, "cuda")
    katago.eval()
    projection = np.eye(384, dtype=np.float32)
    games, rejected = load_queries(manifest["queries"], pilot)
    memory, weights, nets = rank_bank(pilot, "cuda")
    todo = [(k, {"sgf": g["sgf"], "color": g["color"]}) for k, g in games.items() if k not in rejected]
    complete_keys, ranks, base = [], [], []
    started = time.monotonic()
    with torch.inference_mode(), ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn"),
            initializer=init_worker, initargs=(config,)) as pool:
        # Check the streamed feature path against the original stored pilot tensor.
        first = pilot["rows"][0]
        _, replay, err = pool.submit(checked_replay, ("check", first)).result()
        assert err is None, err
        actual = features(katago, replay, projection)
        with np.load(PILOT / "games/0000.npz", allow_pickle=False) as z:
            for field, a in zip(("phi", "logp", "state", "history", "played"), actual):
                np.testing.assert_allclose(a.cpu().numpy(), z[field], atol=.025, rtol=.005, err_msg=field)
        for start in range(0, len(todo), 28):
            waiting, keys = [], []
            for key, replay, err in pool.map(checked_replay, todo[start:start + 28], chunksize=1):
                if err is not None:
                    rejected[key] = err
                    continue
                waiting.append(features(katago, replay, projection))
                keys.append(key)
            if waiting:
                for k in range(0, len(keys), 4):
                    scores, neutral = score_batch(waiting[k:k + 4], nets, memory, weights)
                    complete_keys.extend(keys[k:k + 4])
                    ranks.extend(scores)
                    base.extend(neutral)
            if start % 140 == 0 or start + 28 >= len(todo):
                print(f"processed {min(start + 28, len(todo))}/{len(todo)}, {len(rejected)} excluded, {time.monotonic() - started:.0f}s", flush=True)
    assert len(set(complete_keys)) == len(complete_keys) and set(complete_keys) | set(rejected) == set(games)
    assert np.isfinite(ranks).all() and np.isfinite(base).all()
    immutable_json(HERE / "exclusions.json", rejected)
    with (HERE / "game_scores.tmp").open("wb") as f:
        np.savez_compressed(f, keys=complete_keys, ranks=ranks, neutral=base,
                            selection_sha=digest(HERE / "selection.json"))
    (HERE / "game_scores.tmp").replace(HERE / "game_scores.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "score"))
    args = parser.parse_args()
    prepare() if args.mode == "select" else run()
