#!/usr/bin/env bash
# M101: merge the two arms (results/m101 = with external players -> S3, results/m101c = control -> S3c; seed 3) into the matcher
# cache, build the averaged sources S2x (S2e's three seeds + S3) and S2y (S2e + S3 + S3c) and evaluate the identifiers.
set -uo pipefail
python -I scripts/m_series/m88_seqid.py --stage merge --full --seeds 3 --out m101:S3 || exit 1
python -I scripts/m_series/m88_seqid.py --stage merge --full --seeds 3 --out m101c:S3c || exit 1
python -I - <<'PY' || exit 1
import numpy as np
from pathlib import Path
C = Path("results/m84_cache"); a, b, c = (np.load(C / f"var_{k}.npz") for k in ("S2e", "S3", "S3c")); assert a["protos"].shape[0] == 3 and b["protos"].shape[0] == 1 and c["protos"].shape[0] == 1
np.savez(C / "var_S2x.npz", protos=np.concatenate([a["protos"], b["protos"]]), qvec=np.concatenate([a["qvec"], b["qvec"]]))
np.savez(C / "var_S2y.npz", protos=np.concatenate([a["protos"], b["protos"], c["protos"]]), qvec=np.concatenate([a["qvec"], b["qvec"], c["qvec"]])); print("var_S2x (4 members) and var_S2y (5 members) written", flush=True)
PY
for t in S3 S3c S2x SP2x S2y SP2y; do python -I scripts/m_series/m84_variant.py --stage eval --tag "$t"; done
