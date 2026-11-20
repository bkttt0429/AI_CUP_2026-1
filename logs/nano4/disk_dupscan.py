"""Disk inventory helper: files of at least 20 MB under the given roots with identical content (same size, same MD5), largest waste first.
Symbolic links are not followed; hard links to the same inode count once."""
import hashlib
import os
import sys
from collections import defaultdict

roots = sys.argv[1:]; by_size = defaultdict(list); seen = set()
for root in roots:
    for d, dirs, files in os.walk(root, followlinks=False):
        for f in files:
            p = os.path.join(d, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if os.path.islink(p) or st.st_size < 20 * 2**20 or (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino)); by_size[st.st_size].append(p)


def digest(p, quick):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        if quick:
            h.update(fh.read(2**22)); fh.seek(max(0, os.path.getsize(p) - 2**22)); h.update(fh.read(2**22))
        else:
            for b in iter(lambda: fh.read(2**24), b""):
                h.update(b)
    return h.hexdigest()


groups = []
for size, ps in by_size.items():
    if len(ps) < 2:
        continue
    q = defaultdict(list)
    for p in ps:
        q[digest(p, True)].append(p)
    for cand in q.values():
        if len(cand) < 2:
            continue
        full = defaultdict(list)
        for p in cand:
            full[digest(p, False)].append(p)
        groups += [(size * (len(g) - 1), size, sorted(g)) for g in full.values() if len(g) > 1]
groups.sort(reverse=True); total = sum(g[0] for g in groups)
print(f"{len(groups)} groups of identical files, {total / 2**30:.1f} GB could be freed by keeping one copy of each", flush=True)
for waste, size, g in groups[:80]:
    print(f"\n{size / 2**20:.0f} MB x {len(g)} copies (extra {waste / 2**20:.0f} MB)")
    for p in g:
        print("   " + p)
