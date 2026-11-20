"""Section 4M M36: the cross-fit choice of aicup.attnfeat and its question columns."""
import numpy as np
import pandas as pd

from aicup.attnfeat import load_scores, question_features


def test_cross_fit_choice_and_question_features(tmp_path):
    keys = np.array(["g0", "g1", "g3", "g4"])
    a = np.log(np.full((4, 10), 0.1, np.float32)); b = a.copy(); a[:, 0] += 1.0; b[:, 1] += 1.0
    np.savez(tmp_path / "part_000of001_0000.npz", keys=keys, logp0=a, logp1=b)
    pd.DataFrame({"game_key": keys, "fold": [0, 1, 3, 4]}).to_csv(tmp_path / "games.csv", index=False)
    sc = load_scores(tmp_path, tmp_path / "games.csv")
    assert np.allclose(sc["g1"], b[1]) and np.allclose(sc["g3"], a[2]) and np.allclose(sc["g4"], a[3]) and np.allclose(sc["g0"], 0.5 * (a[0] + b[0]))
    F = question_features([["g1", "g3"], ["g0"]], sc)
    assert F.shape == (2, 20) and np.allclose(F[0, :10], 0.5 * (b[1] + a[2])) and np.allclose(F[1, 10:], 0.0)
    try:
        question_features([["missing"]], sc)
        raise AssertionError("an unscored game must fail")
    except KeyError:
        pass
