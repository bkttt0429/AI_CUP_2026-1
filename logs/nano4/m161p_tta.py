#!/usr/bin/env python
"""Section 4M M161: orientation-averaged queries (m124_tta_queries.py, unchanged) for the networks of m161p_prod.py.

  python -I logs/nano4/m161p_tta.py --seeds 20 --name S6t      (reads results/m161p and /work/u2507115/m161p_ckpt, parts in
                                                                results/m161p_tta, writes results/m84_cache/var_S6t.npz)
"""
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m124_tta_queries as t  # noqa: E402

t.OUT, t.CK, t.TMP = ROOT / "results/m161p", Path("/work/u2507115/m161p_ckpt"), ROOT / "results/m161p_tta"

if __name__ == "__main__":
    t.main()
