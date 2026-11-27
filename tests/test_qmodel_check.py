"""experiments/qmodel_check.py: the regularised question-level model ladder with the TSFEL-16 and noise-16 robustness blocks (issue #41)."""
import json

import numpy as np
import pytest

import qmodel_check as qc
from tests.test_fused_scripts import _folds_csv, _opponent_scores
from tests.test_l1b_scripts import _synthetic_scores


def _run(tmp_path, capsys, extra):
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab")
    opp_dir = _opponent_scores(tmp_path, name="opp_ab")
    folds_csv = _folds_csv(tmp_path)
    out = tmp_path / "qmodel_check.json"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--per-player", "2", "--per-player-more", "3", "--n-boot", "50",
            "--folds-csv", str(folds_csv), "--out", str(out), *extra]
    code = qc.main(args)
    return code, out, capsys.readouterr(), args


def test_qmodel_check_cli_without_tsfel(tmp_path, capsys):
    code, out, cap, args = _run(tmp_path, capsys, ["--no-tsfel"])
    assert code == 0
    res = json.loads(out.read_text())
    assert res["sets"] == ["base", "base_noise16"] and res["models"] == list(qc.MODELS) and res["tsfel_cache"] is None
    keys = set(res["scores"])
    assert keys == {f"{m}@base" for m in qc.MODELS} | {f"{m}@base_noise16" for m in qc.MODELS if m != "mlp_pp64"}
    assert res["scores"]["mlp@base"]["n_features"] == 59 and res["scores"]["mlp@base_noise16"]["n_features"] == 75 and res["scores"]["mlp_pp64@base"]["n_features"] == 59
    assert set(res["comparisons"]["pre_specified"]) == {"mlp_l1@base vs mlp@base"}
    assert set(res["comparisons"]["report_only"]) == {f"{m}@base vs mlp@base" for m in ("mlp_es", "mlp_l2", "mlp_h16", "softmax", "mlp_pp64")}
    assert set(res["comparisons"]["robustness"]) == {f"{m}@base_noise16 vs {m}@base" for m in qc.MODELS if m != "mlp_pp64"}
    c = res["comparisons"]["pre_specified"]["mlp_l1@base vs mlp@base"]
    assert set(c["conditions"]) == {"1", "3", "4"} and len(c["per_fold_diff"]) == 4 and isinstance(c["pass"], bool) and res["min_diff"] == 0.005
    for m in ("mlp_es", "mlp_l1"):
        assert all(e is not None and 10 <= e <= 300 and e % 10 == 0 for e in res["scores"][f"{m}@base"]["best_epoch"])
    assert all(e is None for e in res["scores"]["mlp@base"]["best_epoch"]) and len(res["inner_questions"]) == 4 and all(k > 0 for k in res["inner_questions"])
    assert "[pre_specified] mlp_l1@base vs mlp@base" in cap.out and "[robustness] mlp@base_noise16 vs mlp@base" in cap.out and "best epochs" in cap.out and "wrote" in cap.out
    # the noise block is seeded: the same command reproduces the scores exactly
    assert qc.main(args) == 0 and json.loads(out.read_text())["scores"]["mlp@base_noise16"]["per_fold"] == res["scores"]["mlp@base_noise16"]["per_fold"]
    with pytest.raises(SystemExit):
        qc.main(args + ["--folds", "0", "1"])
    # without the cache and without --no-tsfel the run is refused
    assert qc.main([x for x in args if x != "--no-tsfel"] + ["--tsfel-cache", str(tmp_path / "missing")]) == 2 and "run scripts/tsfel_check.py" in capsys.readouterr().err


def test_qmodel_check_cli_with_tsfel_cache(tmp_path, capsys):
    pytest.importorskip("tsfel")
    from aicup.seqtsfel import fold_tsfel_table, save_tsfel_table
    from tests.test_tsfel_check import _opponent_scores_long
    l1b_dir, hsl_dir, _ = _synthetic_scores(tmp_path, folds=(0, 1, 2, 3, 4), l1b_name="l1b_ab", moves=20, players_per_rank=2)
    opp_dir = _opponent_scores_long(tmp_path, "opp_ab", moves=20, players_per_rank=2)
    folds_csv = _folds_csv(tmp_path, players_per_rank=2)
    cache = tmp_path / "tsfel"
    cache.mkdir()
    for f in (1, 2, 3, 4):
        save_tsfel_table(cache / f"fold{f}.npz", fold_tsfel_table(l1b_dir / f"fold{f}", workers=2))
    out = tmp_path / "qmodel_check.json"
    args = ["--l1b", str(l1b_dir), "--opp", str(opp_dir), "--hsl", str(hsl_dir), "--per-player", "2", "--per-player-more", "3", "--n-boot", "50",
            "--folds-csv", str(folds_csv), "--tsfel-cache", str(cache), "--out", str(out)]
    assert qc.main(args) == 0
    res = json.loads(out.read_text())
    assert res["sets"] == ["base", "base_tsfel16", "base_noise16"] and res["tsfel_cache"] == str(cache)
    assert 59 < res["scores"]["mlp@base_tsfel16"]["n_features"] <= 75 and res["scores"]["softmax@base_tsfel16"]["n_features"] == res["scores"]["mlp@base_tsfel16"]["n_features"]
    assert {k for k in res["comparisons"]["robustness"] if "tsfel16" in k} == {f"{m}@base_tsfel16 vs {m}@base" for m in qc.MODELS if m != "mlp_pp64"}
    assert np.isfinite([h["balanced_mean_over_n"] for h in res["scores"].values()]).all()
