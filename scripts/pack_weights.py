#!/usr/bin/env python
"""Pack the submission's weights into one tar with a SHA256SUMS manifest, for a GitHub Release asset (issue #9, item 5).

The fine-tuned L1b trunks (heads3_{A,B}.trunk.pt, 108 MB each) exceed GitHub's 100 MB file limit, so the
weights stay out of git (results/l1b/**/*.pt is ignored) and are archived here: every --l1b-heads file with
its sidecar and, when the sidecar names one, its trunk file; every --models document (rank_model.json,
rank_model_fused.json, ...); every file of --set-dir (the ens_top3 set models). Before packing, each heads
and trunk file is checked against the sha256 its sidecar records, and the fused document's l1b_models
against the same files; a mismatch is refused (exit 2). SHA256SUMS is written inside the tar and next to it.
Release assets may be up to 2 GB each; no quota, no precision change. Upload: GitHub -> Releases -> new
release -> attach the tar.

  python -I scripts/pack_weights.py --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt \\
      --models results/models/rank_model.json results/models/rank_model_fused.json results/models/rank_model_fused_opp.json \\
      --set-dir results/models/set --out release/weights-2026-11-24.tar
"""
import argparse
import hashlib
import json
import sys
import tarfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.setmodel_io import file_sha256, sidecar_path  # noqa: E402


def collect(l1b_heads, models, set_dir):
    """[(archive name, path)] of everything to pack, after the sha256 checks; ValueError on a mismatch."""
    files, seen = [], {}

    def add(path, name=None):
        path = Path(path)
        if not path.exists():
            raise ValueError(f"{path} does not exist")
        name = name or path.name
        if name in seen and seen[name] != path:
            raise ValueError(f"two files would be archived as {name}: {seen[name]} and {path}")
        seen[name] = path
        files.append((name, path))

    heads_sha = {}
    for pt in l1b_heads:
        pt = Path(pt)
        side = sidecar_path(pt)
        if not side.exists():
            raise ValueError(f"{pt}: sidecar {side.name} is missing")
        meta = json.loads(side.read_text())
        sha = file_sha256(pt)
        if meta.get("state_sha256") != sha:
            raise ValueError(f"{pt}: sha256 {sha[:12]} differs from its sidecar's {str(meta.get('state_sha256'))[:12]}")
        add(pt)
        add(side)
        heads_sha[pt.name] = (sha, None)
        if meta.get("trunk_file"):
            trunk = pt.with_name(meta["trunk_file"])
            tsha = file_sha256(trunk) if trunk.exists() else None
            if tsha != meta.get("trunk_sha256"):
                raise ValueError(f"{trunk}: sha256 {str(tsha)[:12]} differs from the sidecar's {str(meta.get('trunk_sha256'))[:12]}")
            add(trunk)
            heads_sha[pt.name] = (sha, tsha)
    for doc_path in models:
        add(doc_path)
        doc = json.loads(Path(doc_path).read_text())
        for m in doc.get("l1b_models") or []:
            got = heads_sha.get(m.get("heads_file"))
            if got is None:
                raise ValueError(f"{doc_path}: L1b model {m.get('name')} ({m.get('heads_file')}) is not among --l1b-heads")
            if got != (m.get("state_sha256"), m.get("trunk_sha256")):
                raise ValueError(f"{doc_path}: L1b model {m.get('name')}'s sha256s differ from the files being packed")
    if set_dir:
        for p in sorted(Path(set_dir).iterdir()):
            if p.suffix in (".pt", ".json"):
                add(p, f"set/{p.name}")
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b-heads", nargs="*", default=[], metavar="PT")
    ap.add_argument("--models", nargs="*", default=[], metavar="JSON", help="rank_model*.json documents")
    ap.add_argument("--set-dir", default=None, help="directory of the set models (.pt + .json)")
    ap.add_argument("--out", default=str(ROOT / "release" / f"weights-{date.today().isoformat()}.tar"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        print(f"{out} exists; pass --overwrite", file=sys.stderr)
        return 2
    try:
        files = collect(a.l1b_heads, a.models, a.set_dir)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if not files:
        print("nothing to pack", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{file_sha256(p)}  {name}" for name, p in files]
    sums = out.with_name("SHA256SUMS")
    sums.write_text("\n".join(lines) + "\n")
    tmp = out.with_name(out.name + ".tmp")
    with tarfile.open(tmp, "w") as tar:
        for name, p in files:
            tar.add(p, arcname=name)
        tar.add(sums, arcname="SHA256SUMS")
    tmp.replace(out)
    total = sum(p.stat().st_size for _, p in files)
    print(f"wrote {out}: {len(files)} files, {total / 1e6:.1f} MB of weights, {out.stat().st_size} bytes; manifest {sums}")
    for line in lines:
        print("  " + line)
    print("sha256 of the tar: " + hashlib.sha256(out.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
