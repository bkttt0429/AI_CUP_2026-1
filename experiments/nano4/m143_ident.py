#!/usr/bin/env python
"""Section 4M M143 step 2 (2026-11-15): identifiers that include the 40,000-step symmetry-augmented networks (design doc M143).

For every --tag NAME:var[,var...] (sources fused with the cached block; every member of each variant file is used):
  m84_variant.stage_eval_multi  -> results/r_variants/m84_<NAME>_{val,test}.npz, m84_<NAME>_eval.json (weights, held-out top-1)
  m127_conf_family.fuse         -> results/r_variants/m127_<NAME>.npz (top-20 fused values)
  m127_conf_family.quick_write  -> results/r_variants/m84_<NAME>~m15_{val,test}.npz
m84_variant.finalize is disabled here: it rewrites upload_next/README.md from a fixed template (the note about the final file
at the top of that README would be lost). One GPU.

  python -I logs/nano4/m143_ident.py --tag SP4L:P,S4Lt --tag SPAL:P,S2y,S4ft,S4Lt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m127_conf_family as m127  # noqa: E402
import m84_variant as mv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--tag", action="append", required=True); ap.add_argument("--stat", default="m15"); a_ = ap.parse_args()
    mv.finalize = lambda: print("(finalize skipped: upload_next/README.md is not touched)", flush=True)
    for spec in a_.tag:
        name, vars_ = spec.split(":"); src = []
        assert name not in mv.TAGS and name not in mv.MULTI, f"{name} is an existing tag"
        for v in vars_.split(","):
            n = int(np.load(mv.CACHE / f"var_{v}.npz", mmap_mode="r")["protos"].shape[0]); src.append((v, list(range(n))))
        mv.TAGS[name] = (src[-1][0], src[-1][1], None); mv.MULTI[name] = src
        print(f"== identifier {name}: block + " + " + ".join(f"{v} ({len(m)} members)" for v, m in src), flush=True)
        mv.stage_eval_multi(name); m127.fuse(name); m127.quick_write([name], a_.stat)


if __name__ == "__main__":
    main()
