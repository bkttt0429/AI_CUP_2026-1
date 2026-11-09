"""Losses aligned with the official score (architecture v2 section 4F).

S(k, y) is the official score of predicting class k when the truth is y: 1, e^-1 for a neighbour, 0 otherwise
(aicup.metric). Two losses are built on it and shared by the trees, the MLP and the C / T arms:

  kernel targets   t(k | y) = S(k, y) / sum_j S(j, y); the loss is the cross-entropy against this soft label. A proper
                   loss: the minimiser is the class posterior blurred by the kernel, so the matching decision is the plain
                   argmax (the expected-score decision would blur twice).
  expected score   l(z, y) = -sum_k softmax(z)_k S(k, y), the negative expected official score. A linear scoring rule
                   whose minimiser is a one-hot on the expected-score decision, so it serves only as a second stage that
                   starts from a cross-entropy solution and stops on held-out players.

The numpy functions give XGBoost custom objectives their gradient and hessian (hessian 2 p (1 - p) as in XGBoost's own
softmax objective, floored, times the sample weight); the torch functions go into the training loops.
"""
import numpy as np
import torch

from aicup.metric import ADJACENT

N_CLASSES = 10
HESS_FLOOR = 1e-6


def score_matrix(n_classes=N_CLASSES):
    """S[k, y]: the official score of predicting k when the truth is y."""
    d = np.abs(np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :])
    return np.where(d == 0, 1.0, np.where(d == 1, ADJACENT, 0.0))


def kernel_targets(y, n_classes=N_CLASSES):
    """[N, n_classes] soft targets t(k | y) = S(k, y) / sum_j S(j, y): interior classes 0.576 / 0.212 / 0.212, the two
    edge classes 0.731 / 0.269."""
    T = score_matrix(n_classes)[:, np.asarray(y, dtype=int)].T
    return T / T.sum(axis=1, keepdims=True)


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _weighted(g, h, weight):
    if weight is None:
        return g, h
    w = np.asarray(weight, dtype=float)[:, None]
    return g * w, h * w


def kernel_grad_hess(margin, y, weight=None):
    """Gradient p - t and hessian 2 p (1 - p) of the kernel-target cross-entropy at raw margins [N, K]."""
    p = softmax(margin)
    return _weighted(p - kernel_targets(y, p.shape[1]), np.maximum(2.0 * p * (1.0 - p), HESS_FLOOR), weight)


def expected_score_np(proba, y):
    """Per-sample expected official score sum_k p_k S(k, y)."""
    p = np.asarray(proba, dtype=float)
    return (p * score_matrix(p.shape[1])[:, np.asarray(y, dtype=int)].T).sum(axis=1)


def expected_score_grad_hess(margin, y, weight=None):
    """Gradient -p_j (S(j, y) - sum_k p_k S(k, y)) of the negative expected score at raw margins [N, K]; the hessian is the
    softmax surrogate 2 p (1 - p) (the exact second derivative is indefinite)."""
    p = softmax(margin)
    S = score_matrix(p.shape[1])[:, np.asarray(y, dtype=int)].T
    g = -p * (S - (p * S).sum(axis=1, keepdims=True))
    return _weighted(g, np.maximum(2.0 * p * (1.0 - p), HESS_FLOOR), weight)


def kernel_ce_torch(logits, targets):
    """Per-sample cross-entropy against soft targets [N, K] (kernel_targets as a tensor)."""
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1)


def expected_score_loss_torch(logits, y, S=None):
    """Per-sample negative expected official score; S is score_matrix as a tensor on logits' device (built when None)."""
    if S is None:
        S = torch.as_tensor(score_matrix(logits.shape[1]), dtype=logits.dtype, device=logits.device)
    return -(torch.softmax(logits, dim=1) * S[:, y].T).sum(dim=1)
