"""Numerical diagnosis: quadratic reference fit versus exact convex MAP.

8 calibration players, existing 7/3 game split, fixed features and ridge.
No official pool data and no production writes. This is not efficacy validation.
"""
import os
os.environ.setdefault('AICUP1_ROOT', '/home/u2507115/BK_AICUP_2026-1')
from pathlib import Path
import sys
STUDY = Path('/home/u2507115/.cache/pidchoice_wt')
sys.path[:0] = [str(STUDY / 'scripts'), str(STUDY / 'src')]
import hashlib
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scipy.optimize import minimize
from pid_choice_extract import forward_choices, te
from pid_choice_data_audit import replay_choices_checked
from pid_choice_manifest import read_manifest, validate_projection, verify_source
from aicup.humansl import load_humansl

RUN = Path('/work/u2507115/pid_v2/choice_probe_20261116/2993db0b3ed8')
PREFIX = Path(__file__).with_suffix('')
CONFIGS = [('A1', 1.), ('A2', 1.), ('A2', 10.)]
N_PLAYERS = 8


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selected_rows(m):
    rows = [g for g in m['games'] if g['pool'] == 'calibration']
    owners = sorted({g['owner'] for g in rows}, key=lambda p: hashlib.sha256(('choice-exactfit-diag-v1|' + p).encode()).hexdigest())[:N_PLAYERS]
    return owners, [g for g in rows if g['owner'] in owners]


def prepare():
    # SGFs are embedded and hash-checked by read_manifest. Verify only consumed
    # external assets below, not unrelated original CSVs and baseline models.
    m = read_manifest(RUN, verify=False)
    owners, rows = selected_rows(m)
    p = {'scope': 'Numerical mechanism probe; selected calibration players are not fresh validation. No official-pool retuning.',
         'run_id': m['run_id'], 'owners': owners, 'keys': [g['key'] for g in rows],
         'configs': CONFIGS, 'reference_games': 7, 'heldout_games': 3, 'max_iterations': 100,
         'comparison': 'Same fixed q, phi, scale and lambda; quadratic MAP versus exact convex MAP; both scored with exact normalizer.',
         'checks': 'Exact objective must not increase; infinity norm of exact gradient <=1e-5; report every config, no selection.',
         'resource_limit': 'One GPU, 8-minute restart after cancelling the 32-player run for replay cost; cumulative allocation target <=15 GPU-minutes. Feature maps transient.',
         'revision_basis': 'Before any heldout results: 32-player run replay throughput exceeds budget. Take first 8 of identical preordered owners; same configurations and tests.',
         'manifest_sha256': sha(RUN / 'manifest.json'), 'normalizer_sha256': sha(RUN / 'normalizer.npz'),
         'script_sha256': sha(__file__),
         'source_sha256': {str(STUDY / x): sha(STUDY / x) for x in ['scripts/pid_choice_extract.py', 'scripts/pid_choice_data_audit.py', 'src/aicup_pid/choice_signal.py']}}
    path = PREFIX.with_suffix('.protocol.json')
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(p)), 'protocol changed'
    else:
        path.write_text(json.dumps(p, indent=2) + '\n')
    print('Protocol ready:', path, flush=True)


def replay(g):
    return replay_choices_checked(g['key'], g['sgf'], g['color'])


def objective(lp, phi, played, ridge):
    actual = phi[torch.arange(len(phi), device=phi.device), played].mean(0)
    def fun(z):
        zz = torch.as_tensor(z, device=phi.device, dtype=torch.float64)
        logits = lp + phi @ zz
        probs = logits.softmax(1)
        loss = torch.logsumexp(logits, 1).mean() - actual @ zz + ridge * (zz @ zz) / 2
        grad = (probs[..., None] * phi).sum((0, 1)) / len(phi) - actual + ridge * zz
        return float(loss), grad.cpu().numpy()
    return fun


def fit(lp, phi, played, ridge):
    q = lp.softmax(1)
    mu = (q[..., None] * phi).sum(1)
    actual = phi[torch.arange(len(phi), device=phi.device), played]
    u = (actual - mu).mean(0)
    f = torch.einsum('ta,tad,tae->de', q, phi, phi) / len(phi) - mu.T @ mu / len(phi)
    z0 = torch.linalg.solve(f + ridge * torch.eye(phi.shape[-1], device=phi.device, dtype=phi.dtype), u).cpu().numpy()
    fun = objective(lp, phi, played, ridge)
    before, g0 = fun(z0)
    opt = minimize(fun, z0, jac=True, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-7, 'maxls': 30})
    after, g1 = fun(opt.x)
    assert np.isfinite(opt.x).all() and after <= before + 1e-9
    assert np.max(np.abs(g1)) <= 1e-5, (opt.message, np.max(np.abs(g1)))
    return z0, opt.x, {'objective_before': before, 'objective_after': after, 'objective_gain': before - after,
                        'gradient_inf_before': float(np.max(np.abs(g0))), 'gradient_inf_after': float(np.max(np.abs(g1))),
                        'iterations': int(opt.nit), 'optimizer_success': bool(opt.success),
                        'relative_parameter_change': float(np.linalg.norm(opt.x - z0) / max(np.linalg.norm(z0), 1e-12))}


def self_check():
    rng = np.random.default_rng(121)
    lp = torch.log_softmax(torch.tensor(rng.normal(size=(12, 5)), dtype=torch.float64), 1)
    phi = torch.tensor(rng.normal(size=(12, 5, 3)), dtype=torch.float64)
    played = torch.tensor(rng.integers(0, 5, 12))
    z, direction = rng.normal(size=3), rng.normal(size=3)
    fn = objective(lp, phi, played, 1.)
    delta = 1e-5
    fd = (fn(z + delta * direction)[0] - fn(z - delta * direction)[0]) / (2 * delta)
    assert abs(fd - fn(z)[1] @ direction) < 1e-8
    # Adding a state-dependent constant to every action must cancel.
    phi2 = phi + torch.tensor(rng.normal(size=(12, 1, 3)))
    fn2 = objective(lp, phi2, played, 1.)
    assert abs(fn(z)[0] - fn2(z)[0]) < 1e-12
    assert np.max(np.abs(fn(z)[1] - fn2(z)[1])) < 1e-12
    fit(lp, phi, played, 1.)
    print('Gradient, common-offset invariance and convex optimization checks passed', flush=True)


def features(model, ex, games, projection, scale):
    xs, ps = [], []
    for r in ex.map(replay, games, chunksize=1):
        x = forward_choices(model, r, projection)
        xs.append({k: v.double() for k, v in x.items()})
        ps.append(torch.tensor(r['played'], device='cuda', dtype=torch.long))
    lp = torch.cat([x['logp'] for x in xs])
    # Remove float32 normalization drift; shared by both estimators.
    lp = lp - torch.logsumexp(lp, 1, keepdim=True)
    psi = torch.cat([x['psi'] for x in xs]) / scale
    value = torch.cat([x['value'] for x in xs])
    return lp, psi, value, torch.cat(ps), [len(p) for p in ps]


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'use Slurm'
    start = time.monotonic()
    protocol = json.loads(PREFIX.with_suffix('.protocol.json').read_text())
    assert protocol['script_sha256'] == sha(__file__)
    assert protocol['manifest_sha256'] == sha(RUN / 'manifest.json')
    assert protocol['normalizer_sha256'] == sha(RUN / 'normalizer.npz')
    for p, h in protocol['source_sha256'].items():
        assert sha(p) == h
    self_check()
    m = read_manifest(RUN, verify=False)
    for k in ('checkpoint', 'projection'):
        verify_source(m['assets'][k])
    owners, rows = selected_rows(m)
    assert protocol['owners'] == owners and protocol['keys'] == [g['key'] for g in rows]
    assert all(g['fold'] in (3, 4) for g in rows)
    ref = {p: [g for g in rows if g['owner'] == p and g['role'] == 'reference'] for p in owners}
    held = {p: [g for g in rows if g['owner'] == p and g['role'] == 'heldout'] for p in owners}
    assert all(len(ref[p]) == 7 and len(held[p]) == 3 for p in owners)
    assert not {g['key'] for v in ref.values() for g in v} & {g['key'] for v in held.values() for g in v}
    projection = validate_projection(m['assets']['projection']['path'])
    checkpoint = m['assets']['checkpoint']['path']
    config = torch.load(checkpoint, map_location='cpu', weights_only=True)['config']
    torch.set_num_threads(1)
    model, _ = load_humansl(checkpoint, 'cuda')
    model.eval()
    scale = torch.tensor(np.load(RUN / 'normalizer.npz')['scale'], device='cuda', dtype=torch.float64)
    estimates, details = {str(c): [] for c in CONFIGS}, {str(c): [] for c in CONFIGS}
    scores = {str(c): {'quadratic_fit': [], 'exact_fit': []} for c in CONFIGS}
    counts = []
    with ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context('spawn'), initializer=te.init_worker, initargs=(config,)) as ex:
        for i, p in enumerate(owners):
            lp, psi, value, played, ns = features(model, ex, ref[p], projection, scale)
            for arm, ridge in CONFIGS:
                phi = psi if arm == 'A1' else torch.cat([psi, psi * value[:, None, None]], -1)
                z0, z1, info = fit(lp, phi, played, ridge)
                estimates[str((arm, ridge))].append((z0, z1))
                details[str((arm, ridge))].append({'owner': p, 'positions': sum(ns), **info})
            del lp, psi, value, played, phi
            PREFIX.with_suffix('.reference_progress.json').write_text(json.dumps({'scope': 'Reference-only interim; not efficacy results', 'fit_diagnostics': details}) + '\n')
            print(f'reference {i+1}/{len(owners)} elapsed={time.monotonic()-start:.1f}s', flush=True)
        for i, p in enumerate(owners):
            lp, psi, value, played, ns = features(model, ex, held[p], projection, scale)
            for arm, ridge in CONFIGS:
                key = str((arm, ridge))
                phi = psi if arm == 'A1' else torch.cat([psi, psi * value[:, None, None]], -1)
                for j, name in enumerate(('quadratic_fit', 'exact_fit')):
                    zs = torch.tensor(np.stack([zz[j] for zz in estimates[key]]), device='cuda')
                    # Batch candidate vectors to keep the score tensor small.
                    per_candidate = []
                    for zz in zs:
                        tilt = phi @ zz
                        score = tilt[torch.arange(len(phi), device='cuda'), played] - torch.logsumexp(lp + tilt, 1)
                        per_candidate.append(torch.stack([chunk.sum() for chunk in score.split(ns)]).cpu().numpy())
                    scores[key][name].append(np.stack(per_candidate, 1))
            counts.extend(ns)
            del lp, psi, value, played, phi
            print(f'heldout {i+1}/{len(owners)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    counts, truth = np.array(counts), np.repeat(np.arange(len(owners)), 3)
    result = {'scope': protocol['scope'], 'candidate_count': len(owners), 'protocol_sha256': sha(PREFIX.with_suffix('.protocol.json')), 'configs': {}, 'fit_diagnostics': details}
    for key, sc in scores.items():
        a, b = np.concatenate(sc['quadratic_fit']), np.concatenate(sc['exact_fit'])
        own = np.arange(len(truth)), truth
        per_owner_gain = ((b[own] - a[own]) / counts).reshape(len(owners), 3).mean(1)
        result['configs'][key] = {'quadratic_fit_top1': float((a.argmax(1) == truth).mean()),
                                 'exact_fit_top1': float((b.argmax(1) == truth).mean()),
                                 'own_heldout_llr_gain_nats_per_move': float(per_owner_gain.mean()),
                                 'owners_with_positive_gain': int((per_owner_gain > 0).sum()),
                                 'per_owner_heldout_llr_gain': per_owner_gain.tolist(),
                                 'recovered_top1': int(((a.argmax(1) != truth) & (b.argmax(1) == truth)).sum()),
                                 'broken_top1': int(((a.argmax(1) == truth) & (b.argmax(1) != truth)).sum())}
    result['elapsed_seconds'] = time.monotonic() - start
    PREFIX.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(PREFIX.with_suffix('.npz'), truth=truth, counts=counts, owners=np.array(owners),
                        **{key + ':' + name: np.concatenate(v) for key, sc in scores.items() for name, v in sc.items()})
    print(json.dumps(result['configs']), flush=True)


if __name__ == '__main__':
    if '--prepare' in sys.argv:
        prepare()
    else:
        main()
