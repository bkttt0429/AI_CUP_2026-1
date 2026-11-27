#!/usr/bin/env bash
# File-access audit, second version (falls back to a plain run on nodes without strace; an empty list never replaces the stored one): run the chain that rebuilds the final files and reads the daily probes under strace and
# list every file it opens. Used to decide what is NOT needed any more (nothing is deleted or moved here).
#   chain: identifier refit of SQ4's composition (must equal SQ4) -> paired pool comparison -> registered pool test + file
#   build (must equal ct112 byte for byte) -> ct115 builder's reproduction check -> probe decoder -> ledger build ->
#   recognition-share report -> anchor diagnostic.
main() {
  local A=/dev/shm/audit_$$ O=reports/backup_20261118 rc=0 m
  mkdir -p $A $O || return 1; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  rm -f submission_rank_ctaudit.csv results/r_variants/*ctaudit* results/r_variants/*AU1* results/r_variants/audit_pool.json results/r_variants/m163_pool_audit.json
  local tr=1; command -v strace > /dev/null || { tr=0; echo "== no strace on $(hostname): the chain is run without the file list (checks only)"; }
  run() { local n=$1 r; shift; if [ $tr = 1 ]; then strace -f -qq -e trace=openat,open -o $A/$n.strace "$@" > $A/$n.out 2>&1; r=$?; else "$@" > $A/$n.out 2>&1; r=$?; : > $A/$n.strace; fi; echo "== $n: exit $r ($(wc -l < $A/$n.strace) open calls)"; [ $r = 0 ] || { tail -4 $A/$n.out | cut -c1-300; rc=1; }; }
  run ident python -I logs/nano4/m164_ident.py --prior fit --tag AU1:S6at,S7t,S7tc
  run pool165 python -I logs/nano4/m165_pool.py --sets SQ4 SPALm --json audit_pool.json
  run pool163 python -I logs/nano4/m163_pool.py --new SQ4 --control SPALm --name ctaudit --json m163_pool_audit.json
  run build207 python -I logs/nano4/m207_build.py
  run decode python -I logs/nano4/m206_decode.py upload_probe/probe_C2B.csv 0.8417364 --ref submission_rank_ct116.csv --ref-score 0.8394955
  run diag120 python -I logs/nano4/m120_diag_mass.py SPALm~m15 SQ4~m15
  run diag224 python -I logs/nano4/m224_anchor_diag.py SQ4 SZ3
  cat > $A/ledger.py <<'PY'
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led
import ens_compare as ec
z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); prior = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}
qs, cand, obs = led.build({**led.KNOWN_SCORES, **json.load(open(led.LEDGER))}, prior); print(f"ledger build: {len(obs)} uploads, {len(qs)} constrained Public questions")
for v in ec.FOLDS:
    y, n, pid, q = ec.load_fold(v); print(f"fold {v}: {len(y)} questions")
PY
  run ledger python -I $A/ledger.py
  grep -h "minus\|REGISTERED" $A/pool165.out $A/pool163.out | cut -c1-160
  m=$(md5sum submission_rank_ctaudit.csv 2>/dev/null | cut -c1-12); [ "$m" = 7d99d6f2c5b7 ] && echo "== the rebuilt file equals ct112 byte for byte (md5 $m)" || { echo "== the rebuilt file DIFFERS from ct112: '$m'"; rc=1; }
  python -I - <<'PY' || rc=1
import numpy as np, sys
bad = 0
for part in ("val", "test"):
    a, b = (np.load(f"results/r_variants/m84_{i}~m15_{part}.npz", allow_pickle=True) for i in ("SQ4", "AU1"))
    for k in a.files:
        x, y = a[k], b[k]; same = x.shape == y.shape and (np.array_equal(x, y) if x.dtype.kind in "USbi" else np.allclose(x, y, rtol=0, atol=1e-5, equal_nan=True))
        if not same:
            bad += 1; print(f"  {part}/{k}: differs" + (f" (max abs diff {np.nanmax(np.abs(x.astype(float) - y.astype(float))):.2e}; unequal {np.mean(x != y):.4f})" if x.shape == y.shape and x.dtype.kind in "fi" else f" (unequal share {np.mean(x != y):.4f})" if x.shape == y.shape else ""))
print("== the refitted identifier equals SQ4" if not bad else f"== the refitted identifier differs from SQ4 in {bad} array(s)"); sys.exit(1 if bad else 0)
PY
  python -I - $A $O <<'PY'
import os, re, sys
from collections import defaultdict
from pathlib import Path
A, O = Path(sys.argv[1]), Path(sys.argv[2]); ROOT = Path.cwd().resolve(); rd, wr = defaultdict(set), set(); pat = re.compile(r'open(?:at)?\((?:AT_FDCWD, )?"([^"]+)", ([A-Z_|0-9]+)[^)]*\)\s+= (-?\d+)')
for f in sorted(A.glob("*.strace")):
    for line in open(f, errors="ignore"):
        m = pat.search(line)
        if not m or m.group(3).startswith("-"):
            continue
        p = Path(m.group(1)); p = p if p.is_absolute() else ROOT / p
        try:
            r = p.resolve()
        except OSError:
            continue
        s = str(r)
        if not (s.startswith(str(ROOT) + "/") or s.startswith("/work/u2507115/")) or "/.venv/" in s or "/__pycache__/" in s or "/.git/" in s or r.is_dir():
            continue
        (wr.add(s) if ("O_WRONLY" in m.group(2) or "O_RDWR" in m.group(2) or "O_CREAT" in m.group(2)) else rd[s].add(f.stem))
(O / ("audit_chain_files.txt" if rd else "audit_chain_files.empty")).write_text("".join(f"{s}\t{','.join(sorted(v))}\n" for s, v in sorted(rd.items()))); bk = defaultdict(list)
for s in rd:
    rel = s[len(str(ROOT)) + 1:] if s.startswith(str(ROOT) + "/") else s; parts = rel.split("/"); key = "/".join(parts[:2]) if parts[0] == "results" and len(parts) > 2 else "/".join(parts[:5]) if rel.startswith("/work/u2507115/aicup1_moved/") else "/".join(parts[:4]) if rel.startswith("/work/") else parts[0] if len(parts) > 1 else "(root files)"; bk[key].append(s)
print(f"== {len(rd)} distinct files read ({sum(os.path.getsize(s) for s in rd if os.path.exists(s)) / 2**30:.1f} GB), {len(wr)} written; by folder (files, GB):")
for k in sorted(bk):
    v = bk[k]; print(f"  {k}: {len(v)} files, {sum(os.path.getsize(s) for s in v if os.path.exists(s)) / 2**30:.2f} GB" + (("  [" + " ".join(sorted(os.path.basename(s) for s in v))[:700] + "]") if len(v) <= 60 else ""))
PY
  rm -f submission_rank_ctaudit.csv results/r_variants/*ctaudit* results/r_variants/*AU1* results/r_variants/audit_pool.json results/r_variants/m163_pool_audit.json; rm -rf $A
  echo "== $(date '+%F %T') audit finished with rc=$rc; full list in $O/audit_chain_files.txt"
  return $rc
}
main "$@"; exit $?
