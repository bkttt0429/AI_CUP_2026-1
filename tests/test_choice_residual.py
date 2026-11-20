"""Catch wrong pass baselines, owner/ply alignment and unsafe resume."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def implementation():
    assert importlib.util.find_spec("aicup.choice_residual"), "choice residual implementation missing"
    from aicup import choice_residual
    return choice_residual


def test_policy_expectation_includes_pass_and_excludes_occupied():
    c = implementation()
    # One channel, two board points; point 1 occupied despite a large logit.
    trunk = torch.tensor([[[2., 100.]], [[2., 100.]]])
    logits = torch.tensor([[0., 20., 0.], [0., 20., 0.]])
    occupied = torch.tensor([[False, True], [False, True]])
    got = c.features(trunk, logits, occupied, torch.tensor([0, 2]), torch.ones(1, 1))
    np.testing.assert_allclose(got[:, 0], [1., -1.], atol=1e-6)
    np.testing.assert_allclose(got[:, 1:], [[np.log(2), np.log(2), 0]] * 2, atol=1e-6)


def test_quantization_round_trip_and_clipping():
    c = implementation()
    x = np.array([[1., 2.], [1.1, 2.2], [100., -100.]], np.float32)
    mu, sd = np.array([1., 2.]), np.array([2., 4.])
    q = c.quantize(x, mu, sd)
    np.testing.assert_array_equal(q, [[0, 0], [1, 1], [127, -127]])
    np.testing.assert_allclose(c.dequantize(q[:2], mu, sd), x[:2], atol=1e-6)
    with pytest.raises(ValueError):
        c.quantize(np.array([[np.nan, 0]]), mu, sd)


def test_part_rejects_changed_provenance_and_incomplete_ply_alignment(tmp_path):
    c = implementation()
    path = tmp_path / "part.npz"
    data = dict(key=np.array(["g1", "g2"]), LEN=np.array([4, 3], np.int16),
                SUBJ=np.array([0, 1], np.int8), BOTH=np.array([False, True]),
                OFF=np.array([0, 2, 5], np.int64), T=np.array([0, 2, 0, 1, 2], np.int16),
                MOV=np.array([0, 0, 0, 1, 0], np.int8), POS=np.array([0, 361, 2, 3, 4], np.int16),
                XQ=np.zeros((5, 67), np.int8))
    c.save_part(path, data, "run-a", "input-a", reserve_bytes=0)
    assert c.valid_part(path, ["g1", "g2"], "run-a", "input-a")
    assert not c.valid_part(path, ["g1", "g2"], "run-b", "input-a")
    assert not c.valid_part(path, ["g1", "g2"], "run-a", "input-b")
    assert not c.valid_part(path, ["g2", "g1"], "run-a", "input-a")
    data["T"] = np.array([0, 2, 0, 1, 1], np.int16)
    with pytest.raises(ValueError):
        c.save_part(tmp_path / "bad.npz", data, "run-a", "input-a", reserve_bytes=0)


def test_selective_replay_matches_existing_for_handicap_pass_and_empty():
    implementation()
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import choice_residual_extract as e
    import trunk_extract as te
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    te.init_worker(config)
    for sgf, color in [("(;SZ[19]HD[0];B[aa];W[bb];B[];W[cc])", "B"),
                       ("(;SZ[19]HD[2];B[aa];W[bb];B[];W[cc])", "W")]:
        old = te.replay(("g", sgf, color))
        for both in [False, True]:
            got = e.replay(("g", sgf, color, both))
            ix = np.arange(len(old[3])) if both else np.flatnonzero(old[5])
            np.testing.assert_array_equal(got["T"], ix)
            for field, column in [("bins", 1), ("globs", 2), ("POS", 3), ("MOV", 4)]:
                np.testing.assert_array_equal(got[field], old[column][ix])
    empty = e.replay(("empty", "(;SZ[19]HD[0])", "W", False))
    assert empty["LEN"] == 0 and len(empty["T"]) == 0 and empty["SUBJ"] == 1
    long_sgf = "(;SZ[19]HD[0]" + ";B[];W[]" * 129 + ")"
    boundary = e.replay(("boundary", long_sgf, "W", False))
    assert boundary["LEN"] == 256
    np.testing.assert_array_equal(boundary["T"], np.arange(1, 256, 2))


def test_alignment_audit_rejects_missing_subject_ply():
    import choice_residual_extract as e
    task = ("g", "(;SZ[19]HD[0];B[aa];W[bb];B[])", "B", False)
    data = dict(key=np.array(["g"]), LEN=np.array([3]), SUBJ=np.array([0]),
                BOTH=np.array([False]), OFF=np.array([0, 1]), T=np.array([0]),
                MOV=np.array([0]), POS=np.array([0]))
    with pytest.raises(ValueError, match="incomplete/misaligned T"):
        e.check_alignment(data, [task])


def test_resume_rejects_changed_partition_and_extra_parts(tmp_path):
    import choice_residual_extract as e
    assert hasattr(e, "check_layout"), "resume layout guard missing"
    manifest = {"layout": {"nshards": 2, "part": 2}, "expected_games": 5}
    e.check_layout(tmp_path, manifest, 2, 2)
    with pytest.raises(ValueError):
        e.check_layout(tmp_path, manifest, 2, 1)
    with pytest.raises(ValueError):
        e.check_layout(tmp_path, manifest, 3, 2)
    (tmp_path / "part_s0of2_p00000.npz").touch()
    e.check_layout(tmp_path, manifest, 2, 2)
    (tmp_path / "part_s0of1_p00000.npz").touch()
    with pytest.raises(ValueError):
        e.check_layout(tmp_path, manifest, 2, 2)
