#!/usr/bin/env python
"""M197 subject residual64 + scalars3, both sides for registered queries/tests.

Run --stage calibrate on one GPU before --stage extract on sharded GPUs.
Run --stage audit with the same --nshards after ALL extraction shards succeed.
All stages, including tests, run via Slurm. No silent replay skips.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4"),
                str(Path.home() / "BK_AICUP_2026-2/src")]
import trunk_extract as te
import m81_tactic_id as m81
from aicup import choice_residual as cr
from aicup.humansl import load_humansl
from aicup.l1b import _GraphExtraOutputs, neutral_rows
from aicup.sgf import setup_and_moves, board_size

T_MAX = 256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 2**20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def replay(task):
    key, sgf, color, both = task
    if color not in ("B", "W") or board_size(sgf) != 19:
        raise ValueError(f"unsupported game {key}")
    stones, moves = setup_and_moves(sgf)
    gs = te.start_state(stones, te.RULES)
    bins, globs, pos, mover, ply = [], [], [], [], []
    for t, (c, x, y) in enumerate(moves[:T_MAX]):
        pla = te.Board.BLACK if c == "B" else te.Board.WHITE
        loc = te.Board.PASS_LOC if x is None else gs.board.loc(x, y)
        if both or c == color:
            b, g = gs.get_input_features(te._FEATURES)
            bins.append(b[0].astype(np.uint8)); globs.append(g[0].astype(np.float32))
            pos.append(361 if x is None else te._FEATURES.loc_to_tensor_pos(loc, gs.board))
            mover.append(0 if c == "B" else 1); ply.append(t)
        gs.play(pla, loc)
    return dict(key=key, LEN=min(len(moves), T_MAX), SUBJ=int(color == "W"), BOTH=both,
                bins=np.stack(bins) if bins else np.empty((0, 0, 19, 19), np.uint8),
                globs=np.stack(globs) if globs else np.empty((0, 0), np.float32),
                POS=np.array(pos, np.int16), MOV=np.array(mover, np.int8), T=np.array(ply, np.int16))


def load_tasks():
    rows = m81.load_rows()
    if rows.game_key.duplicated().any():
        raise ValueError("training keys are not unique")
    with np.load(ROOT / "results/m84_cache/queries.npz") as q:
        both = set(q["qkey"][q["qtest"] < 0].tolist())
    missing = both - set(rows.game_key)
    if missing:
        raise ValueError(f"registered queries absent from training inputs: {sorted(missing)[:10]}")
    tasks = [(k, s, c, k in both) for k, s, c in zip(rows.game_key, rows.sgf_content, rows.color)]
    tasks.sort(key=lambda x: x[0])
    # Stable task numbering is the same order as existing m81/test trunk extraction.
    tests = [(f"test:{i}", s, c, True) for i, (_, (s, c)) in enumerate(m81.load_tests())]
    return tasks + tests, rows


def provenance():
    files = sorted((ROOT / "data/training").glob("train_*.csv")) + sorted((ROOT / "data/tests").glob("rank_prediction_test_*.csv"))
    sources = [Path(__file__), ROOT / "src/aicup/choice_residual.py", ROOT / "scripts/trunk_extract.py",
               *[ROOT / "src/aicup" / f"{m}.py" for m in ("sgf", "pilot", "humansl", "l1b")],
               ROOT / "logs/nano4/m81_tactic_id.py", ROOT / "logs/nano4/ident_pool.py"]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sources}
    kata_root = Path(sys.modules["katago.game.board"].__file__).resolve().parents[1]
    hashes.update({"katago/" + str(p.relative_to(kata_root)): file_hash(p) for p in sorted(kata_root.rglob("*.py"))})
    return dict(schema="m197-r64-scalars3-v1", policy="occupancy-plus-pass (probe-compatible; not full legality)",
                t_max=T_MAX, projection_seed=te.PROJ_SEED, model_sha256=file_hash(te.CKPT),
                code_sha256=hashes, libraries={"numpy": np.__version__, "torch": torch.__version__},
                input_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in files},
                queries_sha256=file_hash(ROOT / "results/m84_cache/queries.npz"))


class Extractor:
    def __init__(self, workers, batch):
        config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
        self.pool = mp.get_context("fork").Pool(workers, initializer=te.init_worker, initargs=(config,))
        self.model, _ = load_humansl(str(te.CKPT), "cuda")
        self.model.eval()
        self.rows = torch.as_tensor(neutral_rows(), dtype=torch.float32, device="cuda")
        self.proj = None; self.batch = batch

    def close(self):
        self.pool.close(); self.pool.join()

    def extract(self, tasks):
        records = self.pool.map(replay, tasks, chunksize=2)
        active = [d for d in records if len(d["T"])]
        count = sum(len(d["T"]) for d in records)
        x = np.empty((count, 67), np.float32)
        if active:
            bins = np.concatenate([d["bins"] for d in active]); globs = np.concatenate([d["globs"] for d in active])
            pos = np.concatenate([d["POS"] for d in active]); mov = np.concatenate([d["MOV"] for d in active])
            with torch.no_grad():
                for i in range(0, count, self.batch):
                    j = min(count, i + self.batch)
                    b = torch.as_tensor(bins[i:j], device="cuda").float()
                    g = torch.as_tensor(globs[i:j], device="cuda").float()
                    c = torch.as_tensor(mov[i:j], device="cuda").long()
                    extra = _GraphExtraOutputs(["trunkfinal"])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = self.model(b, g, input_meta=self.rows[c], extra_outputs=extra)
                    trunk = extra.returned["trunkfinal"].float().flatten(2)
                    logits = self.model.postprocess_output(output)[0][0][:, 0, :].float()
                    if self.proj is None:
                        self.proj = te.projections(trunk.shape[1], trunk.device)[1]
                    occupied = (b[:, 1] + b[:, 2]).flatten(1) > 0
                    played = torch.as_tensor(pos[i:j], device="cuda").long()
                    x[i:j] = cr.features(trunk, logits, occupied, played, self.proj).cpu().numpy()
        data = {k: np.array([d[k] for d in records], dtype=dt) for k, dt in
                [("key", str), ("LEN", np.int16), ("SUBJ", np.int8), ("BOTH", bool)]}
        data["OFF"] = np.r_[0, np.cumsum([len(d["T"]) for d in records])].astype(np.int64)
        for k in ("T", "MOV", "POS"):
            data[k] = np.concatenate([d[k] for d in records])
        return data, x


def check_alignment(data, tasks):
    """Independent SGF-only coverage check, including original ply and owner."""
    for i, (key, sgf, color, both) in enumerate(tasks):
        _, moves = setup_and_moves(sgf); moves = moves[:T_MAX]
        keep = [(t, c, x, y) for t, (c, x, y) in enumerate(moves) if both or c == color]
        sl = slice(data["OFF"][i], data["OFF"][i + 1])
        if data["key"][i] != key or data["LEN"][i] != len(moves) or data["SUBJ"][i] != int(color == "W") or data["BOTH"][i] != both:
            raise ValueError(f"game metadata mismatch: {key}")
        for field, expected in [("T", [r[0] for r in keep]), ("MOV", [int(r[1] == "W") for r in keep]),
                                ("POS", [361 if r[2] is None else r[3] * 19 + r[2] for r in keep])]:
            if not np.array_equal(data[field][sl], expected):
                raise ValueError(f"incomplete/misaligned {field}: {key}")


def calibrate(a, tasks, rows, info):
    folds = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str})
    players = set(folds.loc[folds.fold == 1, "player_id"])
    held = set(json.loads((ROOT / "results/m82/held_keys.json").read_text()))
    selected = rows[rows.player_id.isin(players) & ~rows.game_key.isin(held)].sample(a.calibration_games, random_state=197)
    sample = [(k, s, c, False) for k, s, c in zip(selected.game_key, selected.sgf_content, selected.color)]
    ex = Extractor(a.workers, a.batch); values = []; offsets = [0]; start = time.time(); plies = 0
    try:
        for i in range(0, len(sample), a.part):
            data, x = ex.extract(sample[i:i + a.part]); check_alignment(data, sample[i:i + a.part])
            values.append(x); offsets.extend((data["OFF"][1:] + offsets[-1]).tolist()); plies += int(data["LEN"].sum())
            print(f"calibration {i + len(data['key'])}/{len(sample)} games; {plies} replay plies; {time.time()-start:.1f}s", flush=True)
    finally:
        ex.close()
    extraction_seconds = time.time() - start
    x = np.concatenate(values); cut = offsets[len(sample) * 3 // 4]
    fit, val = x[:cut], x[cut:]
    mu, sd = fit.mean(0), np.maximum(fit.std(0), 1e-6)
    trials = []; scale = None
    for qscale in [20., 16., 12., 10.]:
        q = cr.quantize(val, mu, sd, qscale); err = (cr.dequantize(q, mu, sd, qscale) - val) / sd
        report = dict(scale=qscale, standardized_rmse=float(np.sqrt(np.mean(err**2))),
                      clipping=float((np.abs((val-mu)/sd*qscale) > 127).mean()),
                      per_dimension_rmse=np.sqrt(np.mean(err**2, axis=0)).tolist(),
                      per_dimension_clipping=(np.abs((val-mu)/sd*qscale) > 127).mean(0).tolist())
        trials.append(report)
        if report["standardized_rmse"] <= .035 and report["clipping"] <= .001:
            scale = qscale; break
    if scale is None:
        raise ValueError(f"quantization fidelity gate failed: {trials}")
    # Exact SGF-only storage forecast, not an extrapolation from sample lengths.
    forecast_start = time.time()
    total_plies = 0; selected_plies = 0
    for _, sgf, color, both in tasks:
        _, moves = setup_and_moves(sgf); moves = moves[:T_MAX]
        total_plies += len(moves); selected_plies += sum(both or c == color for c, _, _ in moves)
    forecast = int(selected_plies * 72 + len(tasks) * 256 + (len(tasks)//a.part + 1) * 8192)
    free = shutil.disk_usage(a.out).free
    if forecast + a.reserve_gib * 2**30 > free:
        raise OSError(f"storage forecast {forecast} + reserve exceeds free {free}")
    stats = a.out / "stats.npz"
    with stats.with_suffix(".tmp").open("wb") as f:
        np.savez(f, mu=mu, sd=sd, scale=scale, keys=np.array([t[0] for t in sample]))
    os.replace(stats.with_suffix(".tmp"), stats)
    elapsed = time.time()-start
    result = dict(status="calibrated", provenance=info, stats_sha256=file_hash(stats),
                  layout={"nshards": a.layout_shards, "part": a.part},
                  calibration_games=len(sample), fit_games=len(sample)*3//4, trials=trials,
                  expected_games=len(tasks), replay_plies=total_plies, stored_plies=selected_plies,
                  forecast_bytes=forecast, free_bytes=free, reserve_gib=a.reserve_gib,
                  measured_replay_plies_per_second=plies/extraction_seconds,
                  measured_stored_plies_per_second=len(x)/extraction_seconds,
                  extraction_seconds=extraction_seconds, forecast_seconds=time.time()-forecast_start,
                  calibration_seconds=elapsed,
                  estimate_8gpu_hours=selected_plies/(len(x)/extraction_seconds)/8/3600)
    result["run_id"] = digest({"provenance": info, "stats_sha256": result["stats_sha256"]})
    atomic_json(a.out / "manifest.json", result)
    print(json.dumps(result, indent=2), flush=True)


def check_layout(out, manifest, part, nshards):
    if manifest["layout"] != {"nshards": nshards, "part": part}:
        raise ValueError("partition layout changed; resume requires the calibrated shard count and part size")
    n = manifest["expected_games"]
    expected = set()
    for s in range(nshards):
        count = len(range(s, n, nshards))
        expected.update(f"part_s{s}of{nshards}_p{k:05d}.npz" for k in range((count + part - 1)//part))
    extra = {p.name for p in Path(out).glob("part_*.npz")} - expected
    if extra:
        raise ValueError(f"unexpected parts from another layout: {sorted(extra)[:10]}")


def run_parts(a, tasks, manifest):
    check_layout(a.out, manifest, a.part, a.nshards)
    with np.load(a.out / "stats.npz") as z:
        mu, sd, scale = z["mu"], z["sd"], float(z["scale"])
    ex = None; start = time.time(); done = 0; replayed = 0; stored = 0; total_bytes = 0
    shards = range(a.nshards) if a.stage == "audit" else [a.shard]
    try:
        for shard in shards:
            mine = tasks[shard::a.nshards]
            for k, i in enumerate(range(0, len(mine), a.part)):
                batch = mine[i:i+a.part]; keys = [t[0] for t in batch]; input_id = digest(batch)
                path = a.out / f"part_s{shard}of{a.nshards}_p{k:05d}.npz"
                if path.exists():
                    if not cr.valid_part(path, keys, manifest["run_id"], input_id):
                        raise ValueError(f"stale/corrupt part; refusing overwrite: {path}")
                    with np.load(path) as z:
                        data = {f: z[f] for f in z.files}
                    check_alignment(data, batch)
                else:
                    if a.stage == "audit":
                        raise FileNotFoundError(path)
                    if ex is None:
                        ex = Extractor(a.workers, a.batch)
                    data, x = ex.extract(batch); check_alignment(data, batch)
                    data["XQ"] = cr.quantize(x, mu, sd, scale)
                    cr.save_part(path, data, manifest["run_id"], input_id, int(a.reserve_gib*2**30))
                    replayed += int(data["LEN"].sum())
                done += len(keys); stored += len(data["T"]); total_bytes += path.stat().st_size
                if k % 20 == 0:
                    print(f"shard {shard}: {i+len(batch)}/{len(mine)} games, new replay {replayed}, {replayed/max(time.time()-start,1):.0f} plies/s, {time.time()-start:.0f}s", flush=True)
        if a.stage == "audit" and (done != manifest["expected_games"] or stored != manifest["stored_plies"]):
            raise ValueError("total game/ply coverage does not match manifest")
        result = dict(status="complete_verified", stage=a.stage, shard=a.shard, nshards=a.nshards,
                      part_games=a.part, games=done, stored_plies=stored, bytes=total_bytes,
                      seconds=time.time()-start, run_id=manifest["run_id"])
        atomic_json(a.out / ("audit.json" if a.stage == "audit" else f"complete_s{a.shard}of{a.nshards}.json"), result)
        print(json.dumps(result), flush=True)
    finally:
        if ex is not None:
            ex.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["calibrate", "extract", "audit"], required=True)
    ap.add_argument("--out", type=Path, default=Path("/work/u2507115/m197_choice64"))
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--part", type=int, default=64); ap.add_argument("--calibration-games", type=int, default=2000)
    ap.add_argument("--layout-shards", type=int, default=8, help="immutable extraction layout recorded by calibration")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--reserve-gib", type=float, default=3.)
    a = ap.parse_args()
    if not 0 <= a.shard < a.nshards or min(a.workers, a.batch, a.part, a.calibration_games, a.layout_shards) <= 0 or a.reserve_gib < 0:
        ap.error("invalid dimensions or reserve")
    a.out.mkdir(parents=True, exist_ok=True)
    tasks, rows = load_tasks(); info = provenance()
    print(f"inputs {len(tasks)} games, {sum(t[3] for t in tasks)} both-side games", flush=True)
    if a.stage == "calibrate":
        if (a.out / "manifest.json").exists() or list(a.out.glob("part_*.npz")):
            raise ValueError("calibration already exists; refusing to replace it")
        calibrate(a, tasks, rows, info)
    else:
        del rows
        manifest = json.loads((a.out / "manifest.json").read_text())
        if manifest["provenance"] != info or manifest["stats_sha256"] != file_hash(a.out / "stats.npz"):
            raise ValueError("input/code/model/calibration provenance changed")
        run_parts(a, tasks, manifest)


if __name__ == "__main__":
    main()
