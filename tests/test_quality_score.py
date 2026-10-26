"""scripts/quality_score.py: per-move point / win-rate loss from the side-to-move values, split by side."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import quality_score as qs  # noqa: E402


def test_losses_and_sides():
    # moves 0..4: B plays 0, 2, 4; W plays 1, 3. lead / win are from the side to move at each position.
    tb, tw = np.array([0, 2, 4]), np.array([1, 3])
    vb = np.array([[0.6, 2.0], [0.5, 1.0], [0.4, -3.0]])  # B to move at t = 0, 2, 4
    vw = np.array([[0.45, -1.5], [0.5, 0.5]])  # W to move at t = 1, 3
    row = qs.game_quality(tb, vb, tw, vw, "B")
    # B's moves: t=0: 2.0 + (-1.5) = 0.5; t=2: 1.0 + 0.5 = 1.5; t=4 is the last position -> skipped
    assert row["t_n"] == 2 and np.isclose(row["t_loss_mean"], 1.0) and np.isclose(row["t_gt1"], 0.5)
    # W's moves: t=1: -1.5 + 1.0 = -0.5; t=3: 0.5 + (-3.0) = -2.5
    assert row["o_n"] == 2 and np.isclose(row["o_loss_mean"], -1.5)
    # win-rate loss of B at t=0: 0.6 - (1 - 0.45) = 0.05
    assert np.isclose(qs.game_quality(tb[:1], vb[:1], tw[:1], vw[:1], "B")["t_wloss_mean"], 0.05)
    # the last position (t=4) is B's: the target's end lead is -3.0
    assert np.isclose(row["t_end_lead"], -3.0) and row["n_positions"] == 5
    w = qs.game_quality(tb, vb, tw, vw, "W")
    assert np.isclose(w["t_loss_mean"], -1.5) and np.isclose(w["o_loss_mean"], 1.0) and np.isclose(w["t_end_lead"], 3.0)


def test_empty_side():
    row = qs.game_quality(np.array([0]), np.array([[0.5, 0.0]]), np.zeros(0, int), np.zeros((0, 2)), "W")
    assert row["t_n"] == 0 and np.isnan(row["t_loss_mean"])
