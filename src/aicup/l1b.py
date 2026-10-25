"""L1b stage 1: a frozen KataGo trunk under rank-unknown metadata, plus one policy head per rank.

The trunk is humanv0's (or any KataGo `Model`), run once per position with the metadata row of
`neutral_meta()`: the same human / KGS / byo-yomi profile as `humansl.make_profile_meta`, but both
ranks unknown (flags 4 and 5 set, the rank thermometers empty), which humanv0 saw in training.
`RankHeads` holds `n_ranks` deep copies of the model's `PolicyHead`; head k is trained only on the
moves of rank-k players, so its masked log-softmax is log p(a | H, k). At step 0 every head equals
humanv0's own policy under the neutral metadata (the known starting point of any improvement).

`score_positions` has the shape of `humansl.score_positions`: the log p of each target move under
every head, [N, n_ranks]. `save_heads` / `load_heads` store the heads' state_dict as .pt with a JSON
sidecar (format, n_ranks, levels, sha256 of the .pt, provenance, caller metadata) and refuse a file
whose sha256 or n_ranks do not match.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from katago.train.model_pytorch import ExtraOutputs, PolicyHead

from aicup.humansl import make_profile_meta
from aicup.ladder import environment, to_json
from aicup.metric import LEVELS
from aicup.setmodel_io import file_sha256, sidecar_path

FORMAT = 1
N_MOVES = 362  # 361 points + pass, the policy output length on 19x19


def neutral_meta():
    """humansl.make_profile_meta's profile with both ranks unknown: rank-unknown flags set, thermometers empty."""
    meta = make_profile_meta("1d")
    meta.inverseBRank = 0
    meta.inverseWRank = 0
    meta.bRankIsUnknown = True
    meta.wRankIsUnknown = True
    return meta


def neutral_rows():
    """Metadata rows [2, 192] float32 for Black to move (row 0) and White to move (row 1)."""
    meta = neutral_meta()
    return np.stack([meta.get_metadata_row(nextPlayer=c, boardArea=361) for c in ("b", "w")]).astype(np.float32)


class RankHeads(torch.nn.Module):
    """n_ranks independent deep copies of a KataGo PolicyHead; forward gives policy channel 0 of each."""

    def __init__(self, policy_head: PolicyHead, n_ranks: int = len(LEVELS)):
        super().__init__()
        self.n_ranks = int(n_ranks)
        self.heads = torch.nn.ModuleList([copy.deepcopy(policy_head) for _ in range(self.n_ranks)])
        self.requires_grad_(True)  # trainable even when copied from a frozen model

    def forward(self, trunk, mask, mask_sum_hw, mask_sum):
        """[B, n_ranks, 362] policy logits (channel 0 of every head), computed in fp32 as KataGo does."""
        outs = [head(trunk, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)[:, 0, :]
                for head in self.heads]
        return torch.stack(outs, dim=1)


def trunk_features(model, bins, globs, colors, rows=None):
    """(trunkfinal [B, C, 19, 19] float32, mask [B, 1, 19, 19], mask_sum_hw [B, 1, 1, 1], mask_sum) of a batch.

    `rows` are the metadata rows [2, 192] indexed by `colors` (0 Black, 1 White to move); None means
    neutral_rows(). A model without a metadata encoder ignores them. The mask quantities are computed
    as Model.forward computes them (plane 0 = on-board). The caller chooses no_grad / autocast.
    """
    device = bins.device
    bins, globs = bins.float(), globs.float()
    input_meta = None
    if model.get_has_metadata_encoder():
        rows_t = torch.as_tensor(neutral_rows() if rows is None else rows, dtype=torch.float32, device=device)
        input_meta = rows_t[torch.as_tensor(colors, device=device).long()]
    extra = ExtraOutputs(["trunkfinal"])
    model(bins, globs, input_meta=input_meta, extra_outputs=extra)
    mask = bins[:, 0:1, :, :].contiguous()
    mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)
    mask_sum = torch.sum(mask)
    return extra.returned["trunkfinal"].float(), mask, mask_sum_hw, mask_sum


def masked_logp(logits, legal):
    """log-softmax over the last axis with illegal points (legal False) at -inf; legal is [B, 362] bool."""
    legal_t = torch.as_tensor(legal, device=logits.device).bool()
    return torch.log_softmax(logits.masked_fill(~legal_t[:, None, :], float("-inf")), dim=-1)


def score_positions(model, heads, bins, globs, legal, target, colors, batch=64):
    """log p of each target move under every head: [N, n_ranks] float32 (humansl.score_positions' shape).

    The trunk runs under no_grad and, on CUDA, bf16 autocast; the heads run in fp32.
    """
    device = next(heads.parameters()).device
    n = len(target)
    out = np.empty((n, heads.n_ranks), dtype=np.float32)
    rows = neutral_rows()
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(n, i + batch)
            b = torch.as_tensor(np.asarray(bins[i:j]), device=device)
            g = torch.as_tensor(np.asarray(globs[i:j]), device=device)
            c = torch.as_tensor(np.asarray(colors[i:j]), device=device).long()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                trunk, mask, mask_sum_hw, mask_sum = trunk_features(model, b, g, c, rows)
            logits = heads(trunk.float(), mask.float(), mask_sum_hw.float(), mask_sum.float())
            logp = masked_logp(logits, torch.as_tensor(np.asarray(legal[i:j]), device=device))
            tgt = torch.as_tensor(np.asarray(target[i:j]), device=device).long()[:, None, None].expand(-1, heads.n_ranks, 1)
            out[i:j] = logp.gather(2, tgt)[:, :, 0].float().cpu().numpy()
    return out


def save_heads(heads, path, meta):
    """state_dict -> path (.pt) and the sidecar path.with_suffix('.json'); returns the sidecar dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), path)
    device = next(heads.parameters()).device
    side = {"format": FORMAT, "n_ranks": heads.n_ranks, "levels": list(LEVELS[:heads.n_ranks]),
            "n_params": int(sum(p.numel() for p in heads.parameters())), "state_sha256": file_sha256(path),
            **environment(str(device)), **meta}
    sidecar_path(path).write_text(to_json(side, indent=1) + "\n")
    return json.loads(sidecar_path(path).read_text())


def load_heads(path, model, device="cpu"):
    """(RankHeads, sidecar, warnings) restored from path onto model.policy_head's architecture.

    ValueError when the sidecar is missing, its format or n_ranks does not fit, or the .pt sha256 differs.
    A code_sha256 other than the current code's is a warning (the heads were trained by other code).
    """
    path = Path(path)
    side_path = sidecar_path(path)
    if not side_path.exists():
        raise ValueError(f"{path}: sidecar {side_path.name} is missing")
    meta = json.loads(side_path.read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: sidecar format {meta.get('format')!r}, this code reads {FORMAT}")
    sha = file_sha256(path)
    if meta.get("state_sha256") != sha:
        raise ValueError(f"{path}: sha256 {sha[:12]} differs from the sidecar's {str(meta.get('state_sha256'))[:12]}")
    n_ranks = meta.get("n_ranks")
    if not isinstance(n_ranks, int) or n_ranks < 1 or meta.get("levels") != list(LEVELS[:n_ranks]):
        raise ValueError(f"{path}: n_ranks {n_ranks!r} / levels {meta.get('levels')!r} do not fit this code's LEVELS")
    heads = RankHeads(model.policy_head, n_ranks)
    state = torch.load(path, map_location=device, weights_only=True)
    try:
        heads.load_state_dict(state)
    except RuntimeError as e:
        raise ValueError(f"{path}: state_dict does not fit model.policy_head's architecture ({e})") from e
    heads.to(device).eval()
    warnings = []
    current = environment("cpu")["code_sha256"]
    if meta.get("code_sha256") != current:
        warnings.append(f"{path}: trained by other training code (code_sha256 {str(meta.get('code_sha256'))[:12]} vs "
                        f"{current[:12]}); predictions may differ from the training run's")
    return heads, meta, warnings
