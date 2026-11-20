"""Read-only storage audit: do not follow symlinks, never count hardlinks twice.

Only stable regular files >=1 MiB are content-hashed. No deletions or rewrites.
"""
import collections
import hashlib
import json
import os
import stat
import time
from pathlib import Path

roots = [Path('/home/u2507115/BK_AICUP_2026-1'),
         Path('/home/u2507115/BK_AICUP_2026-2'), Path('/work/u2507115')]
skip = {'.git', '.venv', '__pycache__', 'node_modules',
        'o26_parallel_run_20261116', 'm161r_ckpt', 'm197_choice64_v2'}
start = time.time()
files, links, errors, inodes = [], [], [], {}

def signature(p):
    s = p.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns

def hash_file(item, sample=False):
    p = Path(item['path'])
    before = signature(p)
    assert list(before) == item['signature'], f'changed before read: {p}'
    h = hashlib.sha256()
    with p.open('rb') as f:
        if sample:
            h.update(f.read(131072))
            f.seek(max(0, before[2] - 131072))
            h.update(f.read(131072))
        else:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                h.update(block)
    assert signature(p) == before, f'changed during read: {p}'
    return h.hexdigest()

for root in roots:
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in list(dirs) + names:
            p = Path(directory) / name
            try:
                s = p.lstat()
                if stat.S_ISLNK(s.st_mode):
                    links.append({'path': str(p), 'target': os.readlink(p),
                                  'resolved': str(p.resolve()), 'exists': p.exists()})
                    if name in dirs:
                        dirs.remove(name)
                elif stat.S_ISREG(s.st_mode) and s.st_size >= 1024*1024:
                    identity = (s.st_dev, s.st_ino)
                    if identity in inodes:
                        inodes[identity]['hardlink_aliases'].append(str(p))
                        continue
                    item = {'path': str(p), 'bytes': s.st_size,
                            'allocated_bytes': s.st_blocks*512,
                            'signature': list(signature(p)), 'hardlink_aliases': [],
                            'recent': s.st_mtime > start-3600}
                    files.append(item)
                    inodes[identity] = item
            except (OSError, AssertionError) as e:
                errors.append(str(e))
    print(f'inventoried {root}: {len(files)} unique large files; {len(links)} symlinks', flush=True)

sizes = collections.defaultdict(list)
for f in files:
    if not f['recent']:
        sizes[f['bytes']].append(f)
samples = collections.defaultdict(list)
for group in sizes.values():
    if len(group) < 2:
        continue
    for f in group:
        try:
            samples[(f['bytes'], hash_file(f, True))].append(f)
        except (OSError, AssertionError) as e:
            errors.append(str(e))
full = collections.defaultdict(list)
for group in samples.values():
    if len(group) < 2:
        continue
    for f in group:
        try:
            full[(f['bytes'], hash_file(f))].append(f)
        except (OSError, AssertionError) as e:
            errors.append(str(e))
dupes = [{'bytes_each': size, 'sha256': sha, 'copies': group,
          'logical_reclaimable_bytes': size*(len(group)-1)}
         for (size, sha), group in full.items() if len(group) > 1]
dupes.sort(key=lambda x: x['logical_reclaimable_bytes'], reverse=True)
report = {'started': start, 'finished': time.time(), 'roots': list(map(str, roots)),
          'excluded_directory_names': sorted(skip), 'minimum_bytes': 1024*1024,
          'recent_cutoff_seconds': 3600, 'files': files, 'symlinks': links,
          'duplicates': dupes, 'errors': errors,
          'logical_reclaimable_bytes': sum(g['logical_reclaimable_bytes'] for g in dupes)}
out = Path('logs/nano4/storage_audit_20261116.json')
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(f"DONE: {len(dupes)} duplicate groups; {report['logical_reclaimable_bytes']/2**30:.3f} GiB potentially reclaimable; {len(errors)} read/change errors", flush=True)
for g in dupes[:15]:
    print(f"{g['logical_reclaimable_bytes']/2**30:.3f} GiB: " + ' | '.join(f['path'] for f in g['copies']), flush=True)
