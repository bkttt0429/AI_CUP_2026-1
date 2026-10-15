"""Official Task-2 scoring: 1 for the exact rank, e^-1 for an adjacent rank, 0 otherwise."""
import math

import numpy as np

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
RANK_TO_IDX = {r: i for i, r in enumerate(LEVELS)}
ADJACENT = math.exp(-1)


def score(pred_idx, true_idx):
    d = abs(int(pred_idx) - int(true_idx))
    return 1.0 if d == 0 else ADJACENT if d == 1 else 0.0


def mean_score(preds, trues):
    preds, trues = list(preds), list(trues)
    if len(preds) != len(trues) or not preds:
        raise ValueError("preds and trues must be non-empty and the same length")
    return sum(score(p, t) for p, t in zip(preds, trues)) / len(preds)


def expected_scores(probs):
    """Expected official score of predicting each class k: p_k + e^-1 (p_{k-1} + p_{k+1})."""
    p = np.asarray(probs, dtype=float)
    padded = np.concatenate([[0.0], p, [0.0]])
    return p + ADJACENT * (padded[:-2] + padded[2:])


def expected_score_decision(probs):
    return int(np.argmax(expected_scores(probs)))
