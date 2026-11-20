"""M197 feature convention and atomic, provenance-checked compact parts.

The policy mask deliberately matches the probe: occupied board points excluded,
pass included. It is NOT a full legality mask (ko/suicide are not filtered).
"""
import os
import shutil
from pathlib import Path

import numpy as np
import torch


def features(trunk, logits, occupied, played, projection):
    """Return residual projection and (-log p, entropy, log1p rank)."""
    points = trunk.shape[-1]
    legal = torch.cat([~occupied, torch.ones(len(trunk), 1, dtype=torch.bool, device=trunk.device)], 1)
    lp = torch.log_softmax(logits.float().masked_fill(~legal, -torch.inf), dim=-1)
    p = lp.exp()
    mean = torch.einsum("bcp,bp->bc", trunk.float(), p[:, :points])
    local = torch.zeros_like(mean)
    on = played < points
    local[on] = trunk[on.nonzero()[:, 0], :, played[on]].float()
    chosen = lp.gather(1, played[:, None])[:, 0]
    entropy = -(p * lp.masked_fill(~legal, 0.)).sum(1)
    rank = (lp > chosen[:, None]).sum(1).float()
    return torch.cat([(local - mean) @ projection,
                      torch.stack([-chosen.clamp(min=-20.), entropy, torch.log1p(rank)], 1)], 1)


def quantize(x, mu, sd, scale=20.):
    if not np.isfinite(x).all() or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("nonfinite feature or invalid calibration")
    return np.clip(np.rint((x - mu) / sd * scale), -127, 127).astype(np.int8)


def dequantize(q, mu, sd, scale=20.):
    return q.astype(np.float32) * (sd / scale) + mu


def check_data(d):
    n, m = len(d["key"]), len(d["T"])
    if len(set(d["key"].tolist())) != n:
        raise ValueError("duplicate game key")
    if any(len(d[k]) != n for k in ("LEN", "SUBJ", "BOTH")):
        raise ValueError("game metadata length mismatch")
    off = d["OFF"]
    if len(off) != n + 1 or off[0] != 0 or off[-1] != m or (np.diff(off) < 0).any():
        raise ValueError("invalid offsets")
    if d["XQ"].shape != (m, 67) or d["XQ"].dtype != np.int8:
        raise ValueError("invalid quantized features")
    if any(len(d[k]) != m for k in ("MOV", "POS")):
        raise ValueError("ply metadata length mismatch")
    if ((d["LEN"] < 0) | (d["LEN"] > 256)).any() or not np.isin(d["SUBJ"], [0, 1]).all():
        raise ValueError("invalid game length or subject")
    if not np.isin(d["MOV"], [0, 1]).all() or ((d["POS"] < 0) | (d["POS"] > 361)).any():
        raise ValueError("invalid mover or position")
    for i in range(n):
        sl = slice(off[i], off[i + 1]); t = d["T"][sl]
        if ((t < 0) | (t >= d["LEN"][i])).any() or (np.diff(t) <= 0).any():
            raise ValueError("invalid original ply alignment")
        if d["BOTH"][i]:
            if not np.array_equal(t, np.arange(d["LEN"][i])):
                raise ValueError("incomplete both-side game")
        elif (d["MOV"][sl] != d["SUBJ"][i]).any():
            raise ValueError("opponent ply in subject-only game")


def valid_part(path, keys, run_id, input_id):
    try:
        with np.load(path, allow_pickle=False) as d:
            if str(d["run_id"]) != run_id or str(d["input_id"]) != input_id or not np.array_equal(d["key"], keys):
                return False
            check_data({k: d[k] for k in d.files})
        return True
    except (OSError, ValueError, KeyError, EOFError):
        return False


def save_part(path, data, run_id, input_id, reserve_bytes=3 * 2**30):
    check_data(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need = sum(v.nbytes for v in data.values()) + 65536
    if shutil.disk_usage(path.parent).free < reserve_bytes + need:
        raise OSError("quota free space below reserve; extraction stopped without deleting files")
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            np.savez(f, **data, run_id=np.array(run_id), input_id=np.array(input_id))
            f.flush(); os.fsync(f.fileno())
        if not valid_part(tmp, data["key"], run_id, input_id):
            raise ValueError("written part failed validation")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
