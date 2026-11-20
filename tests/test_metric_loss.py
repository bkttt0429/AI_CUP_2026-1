import math

import numpy as np
import pytest
import torch

from aicup.metric import expected_scores
from aicup.metric_loss import (expected_score_grad_hess, expected_score_loss_torch, expected_score_np, kernel_ce_torch,
                               kernel_grad_hess, kernel_targets, score_matrix, softmax)

E1 = math.exp(-1)


def test_score_matrix_and_kernel_targets():
    S = score_matrix()
    assert S.shape == (10, 10) and S[3, 3] == 1.0 and S[3, 5] == 0.0
    assert S[3, 4] == pytest.approx(E1) and S[4, 3] == pytest.approx(E1)
    T = kernel_targets([0, 5, 9])
    assert np.allclose(T.sum(axis=1), 1.0)
    assert T[1, 5] == pytest.approx(1 / (1 + 2 * E1)) and T[1, 4] == pytest.approx(E1 / (1 + 2 * E1)) and T[1, 7] == 0.0
    assert T[0, 0] == pytest.approx(1 / (1 + E1)) and T[0, 1] == pytest.approx(E1 / (1 + E1))
    assert T[2, 9] == pytest.approx(1 / (1 + E1)) and T[2, 8] == pytest.approx(E1 / (1 + E1))


def test_expected_score_matches_metric():
    rng = np.random.RandomState(0)
    p = softmax(rng.normal(size=(20, 10)))
    y = rng.randint(0, 10, 20)
    assert np.allclose(expected_score_np(p, y), [expected_scores(pi)[yi] for pi, yi in zip(p, y)])
    lt = expected_score_loss_torch(torch.tensor(np.log(p)), torch.tensor(y))
    assert np.allclose(lt.numpy(), -expected_score_np(p, y))


def test_gradients_match_autograd_and_hessian_is_positive():
    rng = np.random.RandomState(1)
    z, y, w = rng.normal(size=(12, 10)), rng.randint(0, 10, 12), rng.uniform(0.5, 2.0, 12)
    zt = torch.tensor(z, requires_grad=True)
    (expected_score_loss_torch(zt, torch.tensor(y)) * torch.tensor(w)).sum().backward()
    g, h = expected_score_grad_hess(z, y, w)
    assert np.allclose(g, zt.grad.numpy(), atol=1e-10)
    zt = torch.tensor(z, requires_grad=True)
    (kernel_ce_torch(zt, torch.tensor(kernel_targets(y))) * torch.tensor(w)).sum().backward()
    g2, h2 = kernel_grad_hess(z, y, w)
    assert np.allclose(g2, zt.grad.numpy(), atol=1e-10)
    p = softmax(z)
    assert np.allclose(h, np.maximum(2 * p * (1 - p), 1e-6) * w[:, None]) and np.allclose(h, h2) and (h > 0).all()
    g0, _ = kernel_grad_hess(z, y)
    assert np.allclose(g0 * w[:, None], g2)


def test_kernel_ce_optimum_is_the_blurred_label():
    # with the true class fixed, the soft target itself has the lowest kernel cross-entropy, and its argmax is that class
    y = np.array([4])
    t = torch.tensor(kernel_targets(y))
    logits_of = lambda d: torch.log(d.clamp_min(1e-12))  # noqa: E731 -- zero-probability classes get a finite, very negative logit
    best = kernel_ce_torch(logits_of(t), t)
    other = kernel_ce_torch(logits_of(torch.tensor(kernel_targets([5]))), t)
    assert torch.isfinite(best) and float(best) < float(other) and int(t.argmax()) == 4
