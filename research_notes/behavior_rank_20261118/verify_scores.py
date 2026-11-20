"""Compute held-out identification scores for the small adjacent-rank cohort."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np
from score_games import HERE, PILOT, R1, CHOICE, checked_replay, digest, features, load_queries, read


def order(tag, key):
    return hashlib.sha256((tag + "|" + key).encode()).digest()


def split(queries, fold):
    group = sorted((q for q in queries if q["fold"] == fold), key=lambda q: order("verify-enrollment-v1", q["owner"]))
    assert len(group) == 108 and len({q["owner"] for q in group}) == 108
    games = [sorted(q["keys"], key=lambda k: order("verify-roles-v1", k)) for q in group]
    refs = [g[:3] for g in games[:72]]
    query = [key for g in games for key in g[3:]]
    assert len(set(query)) == 216 and set(query).isdisjoint(k for g in refs for k in g)
    return group, refs, query


def retrieve(fold, rows, refs, query):
    """Run the original fold-excluded M218 network on the stored R1 full-game inputs."""
    import torch
    import m88_seqid as base
    by_key = {k: q["owner"] for q in rows for k in q["keys"]}
    wanted = set(query) | {k for g in refs for k in g}
    root = R1 / "results/trunk_vec" / f"fold{fold}"
    def subset(csv_path):
        with csv_path.open() as f:
            keys = [(i, r["game_key"], r["player_id"]) for i,r in enumerate(csv.DictReader(f)) if r["game_key"] in wanted]
        if not keys:
            return {}
        k, owners, *arrays = base.read_part(csv_path)
        found = {}
        for i, key, owner in keys:
            assert k[i] == key and owners[i] == owner == by_key[key]
            found[key] = [a[i].copy() for a in arrays]
        return found
    data = {}
    with ThreadPoolExecutor(4) as pool:
        for part in pool.map(subset, sorted(root.glob("games_s*_p*.csv"))):
            assert not data.keys() & part.keys()
            data.update(part)
    assert set(data) == wanted
    checkpoint = Path("/work/u2507115/m218_ckpt/g25") / f"ckpt_{fold}_s30.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    assert cfg["ext_share"] == .25 and cfg["oe"] == 0
    base.D, base.FFN, base.LAYERS = cfg["width"], cfg["ffn"], cfg["layers"]
    net = base.build_model().cuda()
    net.load_state_dict(saved["best_state"])
    net.eval()
    keys, vectors = sorted(wanted), {}
    for i in range(0, len(keys), 64):
        batch = keys[i:i + 64]
        gl, pos, mov, length, subj = [torch.as_tensor(np.stack(field), device="cuda") for field in zip(*(data[k] for k in batch))]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = net(gl, pos.long(), mov.long(), length.long(), subj.long()).float().cpu().numpy()
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1, atol=2e-5)
        vectors.update(zip(batch, e))
    prototypes = np.array([sum(vectors[k] for k in ref) for ref in refs])
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    return np.stack([vectors[k] for k in query]) @ prototypes.T, digest(checkpoint)


def condition(games, refs, query, katago, projection, nets, pool):
    import torch
    from pid_behavior_train import references
    def batch_replay(keys):
        items = [(k, {"sgf": games[k]["sgf"], "color": games[k]["color"]}) for k in keys]
        for k, replay, error in pool.map(checked_replay, items):
            if error:
                raise ValueError(f"{k}: {error}")
            yield k, features(katago, replay, projection)
    data = {}
    for k, (phi, lp, state, history, played) in batch_replay([k for g in refs for k in g]):
        ix = torch.arange(len(played), device="cuda")
        residual = phi[ix, played] - (lp.exp()[..., None] * phi).sum(1)
        data[k] = dict(state=state, history=history, played=played,
                       ref_value=torch.cat([residual, lp[ix, played, None]], -1))
    mem = references(data, refs)
    scores = []
    # Preserve all 72 individual identities for shortlist evaluation.
    for i in range(0, len(query), 8):
        items = list(batch_replay(query[i:i+8]))
        phi, lp, state, hist, played = [torch.cat(field) for field in zip(*(v for k,v in items))]
        lengths = [len(x[-1]) for _, x in items]
        all_modes = []
        for net in nets:
            pair = []
            for condition in (None, "permute"):
                pred = net(phi, lp, state, hist, *mem, ablation=condition)
                chosen = pred.gather(2, played[:,None,None].expand(-1,len(refs),1)).squeeze(-1)
                pair.append(torch.stack([x.mean(0) for x in chosen.split(lengths)]))
            all_modes.append(torch.stack(pair, dim=1))
        scores.extend(torch.stack(all_modes).mean(0).cpu().numpy())
    return np.stack(scores)


def main():
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("GPU job required")
    import torch
    from pid_behavior_extract import init_worker
    from behavior_model import BehaviorPolicy
    sys.path.insert(0, str(CHOICE))
    import pid_choice_extract
    from aicup.humansl import load_humansl
    assert callable(pid_choice_extract.forward_choices)
    os.chdir(R1)
    sys.path.insert(0, str(R1 / "logs/nano4"))
    torch.set_num_threads(1)
    protocol = read(PILOT / "protocol.json")
    queries = read(HERE / "selection.json")["queries"]
    games, rejected = load_queries(queries, protocol)
    assert not rejected
    checkpoint = protocol["checkpoint"]["path"]
    assert digest(checkpoint) == protocol["checkpoint"]["sha256"]
    katago, _ = load_humansl(checkpoint, "cuda")
    katago.eval()
    config = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]
    frozen = read(PILOT / "selection.json")["checkpoint_sha256"]
    nets = []
    for seed in protocol["seeds"]:
        file = PILOT / "models" / f"contextual_{seed}.pt"
        assert digest(file) == frozen[file.name]
        net = BehaviorPolicy("contextual").cuda()
        net.load_state_dict(torch.load(file, map_location="cuda", weights_only=True))
        nets.append(net.eval())
    projection = np.eye(384, dtype=np.float32)
    out = {"protocol_sha": digest(HERE / "verifier_protocol.json"), "code_sha": digest(Path(__file__)),
           "selection_sha": digest(HERE / "selection.json")}
    with torch.inference_mode(), ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn"),
            initializer=init_worker, initargs=(config,)) as pool:
        # Check exact reference construction against a game already used by the pilot.
        from pid_behavior_extract import replay
        r = pool.submit(replay, protocol["rows"][0]).result()
        phi, lp, state, history, played = features(katago, r, projection)
        ix = torch.arange(len(played), device="cuda")
        rv = torch.cat([phi[ix, played] - (lp.exp()[...,None]*phi).sum(1), lp[ix,played,None]], -1)
        with np.load(PILOT / "games/0000.npz") as z:
            np.testing.assert_allclose(rv.cpu().numpy(), z["ref_value"], atol=.025, rtol=.005)
        for fold in (1, 2):
            rows, refs, qkeys = split(queries, fold)
            cos, source = retrieve(fold, rows, refs, qkeys)
            print(f"fold {fold}: M218 retrieved", flush=True)
            scores = condition(games, refs, qkeys, katago, projection, nets, pool)
            assert cos.shape == (216,72) and scores.shape == (216,2,72) and np.isfinite(scores).all() and np.isfinite(cos).all()
            out.update({f"cos{fold}": cos, f"behavior{fold}": scores, f"keys{fold}": qkeys,
                        f"owners{fold}": [q["owner"] for q in rows], f"m218sha{fold}": source})
            print(f"fold {fold}: both policy conditions finished", flush=True)
    with (HERE / "verifier_scores.tmp").open("wb") as f:
        np.savez_compressed(f, **out)
    (HERE / "verifier_scores.tmp").replace(HERE / "verifier_scores.npz")


if __name__ == "__main__":
    main()
