"""Give each of this batch's eight engine/feeder pairs a disjoint set of CPUs in its own allocation."""
import os
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
assert os.environ.get("SLURM_JOB_ID")=="804628"
workers={}
for proc in Path("/proc").glob("[0-9]*"):
    try:
        if proc.stat().st_uid!=os.getuid() or (proc/"comm").read_text().strip()!="katago":continue
        parent=int(next(l.split()[1] for l in (proc/"status").read_text().splitlines() if l.startswith("PPid:")))
        p=Path("/proc")/str(parent)
        if p.joinpath("cwd").resolve()!=here:continue
        args=p.joinpath("cmdline").read_bytes().split(b"\0")
        if b"search.py" not in args:continue
        shard=int(args[args.index(b"search.py")+1])
        workers[shard]=(int(proc.name),parent)
    except (FileNotFoundError,PermissionError):continue
assert set(workers)==set(range(8)),workers
mask=set().union(*(os.sched_getaffinity(engine) for engine,_ in workers.values()))
assert len(mask)==48
cores={}
for cpu in mask:
    t=Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    key=(int((t/"physical_package_id").read_text()),int((t/"core_id").read_text()))
    cores.setdefault(key,set()).add(cpu)
groups=list(sorted(cores));assert len(groups) in (24,48)
per_worker=len(groups)//8
for shard,(engine,parent) in sorted(workers.items()):
    target=mask if "restore" in sys.argv else set().union(*(cores[k] for k in groups[shard*per_worker:(shard+1)*per_worker]))
    assert len(target) in (6,48)
    files=Path(f"/proc/{engine}/io").read_text().splitlines()
    for pid in (engine,parent):
        for tid in Path(f"/proc/{pid}/task").iterdir():
            os.sched_setaffinity(int(tid.name),target)
    print(shard,engine,sorted(target),[l for l in files if l.startswith(("wchar:","syscw:"))],flush=True)
