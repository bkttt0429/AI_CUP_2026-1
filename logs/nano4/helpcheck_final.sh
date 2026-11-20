#!/usr/bin/env bash
cd /home/u2507115/aicup1_final
PY=/home/u2507115/BK_AICUP_2026-1/.venv/bin/python
fail=0
for f in scripts/make_splits.py scripts/score_folds.py scripts/hd_table.py scripts/patch_scored.py scripts/l1b_build_cache.py scripts/l1b_train.py scripts/l1b_score.py scripts/trunk_extract.py scripts/trunk_seq_check.py scripts/predict_submission.py scripts/trunk_predict.py logs/nano4/x40_keys.py logs/nano4/m53_write_probe.py logs/nano4/m58_decode.py; do
  out=$($PY -I "$f" --help 2>&1 >/dev/null | tail -2)
  if [ $? -ne 0 ] || echo "$out" | grep -q "Error\|Traceback"; then echo "FAIL $f :: $out"; fail=1; else echo "ok   $f"; fi
done
$PY -I -c "
import sys, importlib, pathlib; sys.path[:0] = ['src', 'scripts', 'logs/nano4']
mods = [p.stem for p in pathlib.Path('src/aicup').glob('*.py') if p.stem != '__init__']
bad = []
for m in mods:
    try: importlib.import_module('aicup.' + m)
    except Exception as e: bad.append((m, repr(e)[:120]))
import aicup; print('aicup from', pathlib.Path(aicup.__file__).resolve())
print('module import failures:', bad if bad else 'none')"
exit $fail
