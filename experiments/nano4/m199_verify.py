"""Check reconstructed SQ4 against saved production top scores, plus exploratory CI."""
import json
from pathlib import Path
import numpy as np

root=Path.cwd();rv=root/"results/r_variants"
q=np.load(root/"results/m84_cache/queries.npz");Q,T=int(q["Q"]),int(q["T"])
old=np.load(rv/"m127_SQ4.npz");pred=np.load(rv/"m198_tail_style_predictions.npz")
rep=json.loads((rv/"m199_length_check.json").read_text())
assert np.array_equal(old["hit"],pred["baseline_hit"]), "baseline predictions differ from saved production"
tv=old["TV"][2*Q:2*Q+2*T];m=tv[:,0]-tv[:,1:5].mean(1)
threshold=float(np.quantile(m[T:],.95))
net=float((m[:T]>=threshold).mean()-(m[T:]>=threshold).mean())
assert abs(threshold-rep["readouts"]["SQ4"]["null_q95"])<1e-3
assert abs(net-rep["readouts"]["SQ4"]["test_recognised_net_q95"])<2e-4
mask=pred["confirm"];truth=pred["truth"][mask];w=pred["weight"][mask]
delta=pred["control_hit"][mask].astype(float)-pred["baseline_hit"][mask]
owners,ix=np.unique(truth,return_inverse=True)
n=np.bincount(ix,weights=w*delta);d=np.bincount(ix,weights=w)
rng=np.random.default_rng(199);bs=[]
for _ in range(5000):
    take=rng.integers(0,len(owners),len(owners));bs.append(n[take].sum()/d[take].sum())
out=dict(baseline_verified=True,saved_null_q95=threshold,saved_test_net=net,
         exploratory_length_confirmation_diff=float(np.average(delta,weights=w)),
         exploratory_length_confirmation_ci95=np.quantile(bs,[.025,.975]).tolist())
(rv/"m199_verification.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
