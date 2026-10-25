"""Saving and loading trained set models (aicup.settrain.Fitted) for the submission pipeline (item 2).

A saved model is two files with one stem: <stem>.pt, the torch state_dict as trained (float32 weights,
buffers included), and <stem>.json, the sidecar with everything needed to rebuild and audit it: `rung`
(a key of aicup.ladder.SET_RUNGS, whose make(d_in) rebuilds the module) and its vector `rep`, `seed`,
`d_in`, `d_glob`, the TrainCfg, the standardisation statistics of the Fitted (tok_mean / tok_std /
glob_mean / glob_std as float64 lists -- exactly the ones fit_set_model used, i.e. of the training
questions without the inner players), the inner early-stopping record (inner_ce, best_eval, best_epoch,
stopped_early, n_inner_players, n_inner_questions) and the per-epoch losses, the parameter count, the
profiles and feature names of the pipeline, the sha256 of the .pt bytes, the provenance of the code
that trained it (aicup.ladder.environment: git_head, git_dirty, code_sha256, code_files, torch, device)
and any `extra` fields the trainer adds (training folds, per_player, the questions' sha, the paired base
model and its sha256).

load_set_model rebuilds the Fitted on a device and returns it with the sidecar and a list of warnings;
its predict_set_model output equals the trained model's bit for bit on the same device. It refuses a
sidecar of another format, an unknown rung, other profiles or features than this pipeline computes, or a
.pt whose bytes do not match the recorded sha256; a code_sha256 that differs from the running code is a
warning (the training modules changed since training), not an error.
"""
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from aicup.humansl import PROFILES
from aicup.qfeatures import FEATURE_NAMES
from aicup.settrain import Fitted, TrainCfg

FORMAT = 1
STATS = ("tok_mean", "tok_std", "glob_mean", "glob_std")
INNER = ("inner_ce", "best_eval", "best_epoch", "stopped_early", "n_inner_players", "n_inner_questions")


def sidecar_path(pt_path):
    """<stem>.json next to <stem>.pt."""
    return Path(pt_path).with_suffix(".json")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_set_model(fitted, rung, seed, pt_path, extra=None):
    """Write <pt_path> (state_dict) and its sidecar for the Fitted of `rung` / `seed`; returns the sidecar dict.
    `extra` fields are added to the sidecar (and may not shadow its own keys)."""
    from aicup.aggregators import n_params
    from aicup.ladder import SET_RUNGS, environment, to_json
    if rung not in SET_RUNGS:
        raise ValueError(f"unknown rung {rung!r}; known: {list(SET_RUNGS)}")
    pt_path = Path(pt_path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in fitted.model.state_dict().items()}, pt_path)
    meta = {"format": FORMAT, "rung": rung, "rep": SET_RUNGS[rung][0], "seed": int(seed), "d_in": int(len(fitted.tok_mean)),
            "d_glob": int(len(fitted.glob_mean)), "train_cfg": asdict(fitted.cfg),
            **{k: np.asarray(getattr(fitted, k), dtype=np.float64).tolist() for k in STATS},
            **{k: getattr(fitted, k) for k in INNER}, "losses": list(fitted.losses), "n_params": n_params(fitted.model),
            "profiles": list(PROFILES), "features": list(FEATURE_NAMES), "state_sha256": file_sha256(pt_path),
            **environment(fitted.cfg.device), "written": datetime.now().isoformat(timespec="seconds")}
    clash = sorted(set(extra or {}) & set(meta))
    if clash:
        raise ValueError(f"extra sidecar fields shadow the model's own: {clash}")
    meta.update(extra or {})
    sidecar_path(pt_path).write_text(to_json(meta, indent=1) + "\n")
    return meta


def load_set_model(pt_path, device="cpu"):
    """(Fitted on `device`, sidecar dict, warnings) from <pt_path> and its sidecar; ValueError on a file that
    cannot be trusted (see the module docstring)."""
    from aicup.ladder import SET_RUNGS, code_version
    pt_path = Path(pt_path)
    side = sidecar_path(pt_path)
    if not pt_path.exists() or not side.exists():
        raise FileNotFoundError(f"{pt_path} and {side} are both needed")
    meta = json.loads(side.read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{side}: format {meta.get('format')!r}, this code reads {FORMAT}")
    if meta["rung"] not in SET_RUNGS:
        raise ValueError(f"{side}: unknown rung {meta['rung']!r}; known: {list(SET_RUNGS)}")
    if meta["profiles"] != list(PROFILES) or meta["features"] != list(FEATURE_NAMES):
        raise ValueError(f"{side}: the model was trained with other profiles or features than this pipeline computes")
    sha = file_sha256(pt_path)
    if sha != meta["state_sha256"]:
        raise ValueError(f"{pt_path}: sha256 {sha[:12]} does not match the sidecar's {meta['state_sha256'][:12]}")
    rep, make, _ = SET_RUNGS[meta["rung"]]
    if meta.get("rep", rep) != rep:
        raise ValueError(f"{side}: rep {meta['rep']!r} but the rung {meta['rung']} uses {rep!r}")
    cfg = TrainCfg(**{**meta["train_cfg"], "device": device})
    with torch.random.fork_rng(devices=[]):  # built as fit_set_model builds it; the weights are then replaced
        torch.manual_seed(cfg.seed)
        model = make(meta["d_in"])
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    fitted = Fitted(model, *(np.asarray(meta[k], dtype=np.float64) for k in STATS), cfg, losses=list(meta["losses"]),
                    inner_ce=list(meta["inner_ce"]), best_eval=meta["best_eval"], best_epoch=meta["best_epoch"],
                    stopped_early=meta["stopped_early"], n_inner_players=meta["n_inner_players"],
                    n_inner_questions=meta["n_inner_questions"])
    if len(fitted.tok_mean) != meta["d_in"] or len(fitted.glob_mean) != meta["d_glob"]:
        raise ValueError(f"{side}: statistics of {len(fitted.tok_mean)} / {len(fitted.glob_mean)} columns, d_in / d_glob say "
                         f"{meta['d_in']} / {meta['d_glob']}")
    warnings = []
    now = code_version()["code_sha256"]
    if meta.get("code_sha256") != now:
        warnings.append(f"{side.name}: trained by other training code (code_sha256 {str(meta.get('code_sha256'))[:12]}, "
                        f"now {now[:12]}); the model is used as saved")
    return fitted, meta, warnings


def summary(meta):
    """The sidecar fields worth a report line: rung, seed, d_in, best_epoch, train_folds, git_head, code_sha256, base model."""
    keys = ("rung", "seed", "d_in", "n_params", "best_epoch", "stopped_early", "train_folds", "per_player", "git_head",
            "code_sha256", "base_model", "base_model_sha256")
    return {k: meta.get(k) for k in keys}
