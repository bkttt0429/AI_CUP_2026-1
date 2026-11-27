import subprocess
from pathlib import Path

import numpy as np
import pytest

from aicup.sgf import parse_moves

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "tools" / "katago" / "katago"
MODEL = ROOT / "models" / "katago" / "b18c384nbt-humanv0.bin.gz"


def _engine_runs():
    """The binary needs libzip and OpenSSL 1.1 (tools/katago-libs on lab01); other hosts may lack them."""
    if not (BIN.exists() and MODEL.exists()):
        return False
    try:
        return subprocess.run([str(ROOT / "experiments" / "katago.sh"), "version"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _engine_runs(), reason="KataGo engine not present or cannot start on this host")
def test_engine_target_logps_on_three_move_game():
    from katago_engine_logp import engine_target_logps

    moves = parse_moves("(;SZ[19];B[pd];W[dp];B[pq])")
    res = engine_target_logps(moves, "B", "rank_3d")
    assert [r["t"] for r in res] == [0, 2]
    assert [r["gtp"] for r in res] == ["Q16", "Q3"]
    for r in res:
        assert np.log(0.01) < r["logp"] < np.log(0.9)
        finite = r["policy_logp"][np.isfinite(r["policy_logp"])]
        assert np.exp(finite).sum() == pytest.approx(1.0, abs=1e-3)
    assert np.isfinite(res[1]["policy_logp"]).sum() == np.isfinite(res[0]["policy_logp"]).sum() - 2
