"""Exploratory, read-only diagnosis of the completed choice pilot; no tuning.

Uses the original seeds, reference-only fits, and cached exact likelihoods.
Outputs are not independent confirmation: these pools have been examined before.
Run with project Python; cap BLAS threads. No GPU or production artifact writes.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

RUN = Path('/work/u2507115/pid_v2/choice_probe_20261116/2993db0b3ed8')
OUT = Path(__file__).with_suffix('.json')
CONFIGS = [(a, r) for a in ('A1', 'A2', 'A2perm') for r in (1., 10.)]
SOURCES = {}


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def verified(p):
    meta = json.loads(p.with_suffix('.json').read_text())
    sha = digest(p)
    assert sha == meta['sha256'], str(p)
    SOURCES[str(p)] = sha
    return meta


def ranks(s, y):
    own = s[np.arange(len(y)), y, None]
    return ((s > own) | ((s == own) & (np.arange(s.shape[1])[None] < y[:, None]))).sum(1)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def metrics(s, base, y, sizes):
    r, br = ranks(s, y), ranks(base, y)
    wrong = np.argmax(base, axis=1)
    # The strongest incorrect baseline candidate, also on baseline-correct cases.
    tmp = base.copy()
    tmp[np.arange(len(y)), y] = -np.inf
    rival = tmp.argmax(1)
    margin = s[np.arange(len(y)), y] - s[np.arange(len(y)), rival]
    out = {}
    for n in (1, 2, 3):
        ix = sizes == n
        err = ix & (br > 0)
        near = ix & (br >= 1) & (br <= 4)
        def rate(v, mask):
            return float(v[mask].mean()) if mask.any() else None
        out[str(n)] = {
            'questions': int(ix.sum()), 'baseline_top1': rate(br == 0, ix),
            'top1': rate(r == 0, ix), 'top5': rate(r < 5, ix),
            'median_true_rank_1based': float(np.median(r[ix] + 1)),
            'true_beats_all_wrong_fraction': float((599 - r[ix]).mean() / 599),
            'baseline_errors': int(err.sum()), 'baseline_near_errors': int(near.sum()),
            'true_beats_baseline_rival_on_errors': rate(margin > 0, err),
            'true_beats_baseline_rival_on_near_errors': rate(margin > 0, near),
            'true_beats_baseline_rival_on_correct': rate(margin > 0, ix & (br == 0)),
            'standalone_top1_on_baseline_errors': rate(r == 0, err),
            'top1_agreement_with_baseline': rate(s.argmax(1) == wrong, ix),
        }
    return out


def score_comparison(exact, quad, base, truth, sizes, z):
    """Isolate score approximation with the SAME reference parameter estimate."""
    b = base.copy()
    b[np.arange(len(truth)), truth] = -np.inf
    rival = b.argmax(1)
    em = exact[np.arange(len(truth)), truth] - exact[np.arange(len(truth)), rival]
    qm = quad[np.arange(len(truth)), truth] - quad[np.arange(len(truth)), rival]
    out = {'quadratic_metrics': metrics(quad, base, truth, sizes), 'by_size': {}}
    for n in (1, 2, 3):
        ix = sizes == n
        freq = np.bincount(exact[ix].argmax(1), minlength=len(z))
        prob = freq / freq.sum()
        means = exact[ix].mean(0)
        out['by_size'][str(n)] = {
            'exact_quad_top1_agreement': float((exact[ix].argmax(1) == quad[ix].argmax(1)).mean()),
            'exact_quad_rival_margin_sign_agreement': float((np.sign(em[ix]) == np.sign(qm[ix])).mean()),
            'rival_margin_correlation': float(np.corrcoef(em[ix], qm[ix])[0, 1]),
            'number_candidates_ever_selected': int((freq > 0).sum()),
            'top10_candidates_share_of_predictions': float(np.sort(freq)[-10:].sum() / freq.sum()),
            'prediction_effective_candidate_count': float(1 / np.square(prob).sum()),
            'candidate_average_score_norm_correlation': float(np.corrcoef(means, np.linalg.norm(z, axis=1))[0, 1]),
        }
    return out


def stability(a, b):
    ac, bc = a - a.mean(0), b - b.mean(0)
    denom = np.sqrt(np.square(ac).sum() * np.square(bc).sum())
    cross = (ac * bc).sum()
    # Same person, disjoint games; trace correlation in the fixed calibrated metric.
    between = (ac.T @ bc + bc.T @ ac) / (2 * (len(a) - 1))
    eig = np.linalg.eigvalsh(between)
    positive = eig[eig > 0]
    return {
        'centered_trace_correlation': float(cross / denom),
        'shuffled_person_trace_correlation': float((ac * bc[np.random.default_rng(20261116).permutation(len(bc))]).sum() / denom),
        'reference_mean_energy_fraction': float(len(a) * np.square(a.mean(0)).sum() / np.square(a).sum()),
        'query_mean_energy_fraction': float(len(b) * np.square(b.mean(0)).sum() / np.square(b).sum()),
        'positive_cross_cov_effective_rank': float(positive.sum() ** 2 / np.square(positive).sum()),
        'cross_cov_eigenvalues': eig.tolist(),
        'caution': 'Trace correlation is metric-dependent. Positive eigenspectrum is descriptive, not an identified signal rank.',
    }


def main():
    start = time.monotonic()
    manifest = json.loads((RUN / 'manifest.json').read_text())
    protocol = json.loads((RUN / 'exact_protocol.json').read_text())
    for name in ('manifest.json', 'normalizer.npz', 'exact_protocol.json'):
        SOURCES[str(RUN / name)] = digest(RUN / name)
    scale = np.load(RUN / 'normalizer.npz')['scale']
    result = {'scope': 'Exploratory diagnostics, not new efficacy evidence. Same candidates and overlapping games across seeds and question sizes.',
              'run_id': manifest['run_id'], 'seeds': [2, 3], 'pools': {}, 'script_sha256': digest(Path(__file__))}
    for pool in ('pub2', 'priv2'):
        rows = [g for g in manifest['games'] if g['pool'] == pool]
        by_owner = {}
        for g in rows:
            by_owner.setdefault(g['owner'], []).append(g['key'])
        owners = sorted(by_owner)
        assert len(owners) == 600 and all(len(v) == 10 for v in by_owner.values())
        moments = {}
        for path in sorted((RUN / 'moments').glob(f'{pool}_p*.npz')):
            verified(path)
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in ('U0', 'F0', 'U2', 'F2', 'n')}
                for i, k in enumerate(z['keys'].tolist()):
                    assert k not in moments
                    moments[k] = {name: value[i] for name, value in arrays.items()}
        assert set(moments) == {g['key'] for g in rows}
        result['pools'][pool] = {}
        for seed in (2, 3):
            rng = np.random.default_rng(4100 + seed)
            held = {p: [by_owner[p][i] for i in rng.permutation(10)[:3]] for p in owners}
            ref = {p: [k for k in by_owner[p] if k not in held[p]] for p in owners}
            qs = [q for p in owners for q in ([[k] for k in held[p]] + [held[p][:2], held[p][:3]])]
            truth, sizes = np.repeat(np.arange(600), 5), np.tile([1, 1, 1, 2, 3], 600)
            path = RUN / f'baseline_{pool}_s{seed}.npz'
            verified(path)
            with np.load(path, allow_pickle=False) as z:
                assert z['owners'].tolist() == owners and np.array_equal(truth, z['truth']) and np.array_equal(sizes, z['sizes'])
                base = z['scores']
            evidence = {}
            audit = json.loads((RUN / f'exact_arithmetic_{pool}_s2_3.json').read_text())
            assert audit['verdict'] == 'passed' and audit['protocol_id'] == protocol['protocol_id']
            for path in sorted((RUN / 'exact_scores').glob(f'{pool}_s2_3_p*.npz')):
                meta = verified(path)['metadata']
                assert meta['owners'] == owners and meta['configs'] == [list(c) for c in CONFIGS]
                assert meta['protocol_id'] == protocol['protocol_id'] and meta['pool'] == pool
                with np.load(path, allow_pickle=False) as z:
                    for k, s in zip(z[f'keys_s{seed}'].tolist(), z[f'scores_s{seed}']):
                        assert k not in evidence and np.isfinite(s).all()
                        evidence[k] = s
            assert set(evidence) == {k for ks in held.values() for k in ks}
            out = {'exact': {}, 'stability': {}, 'score_approximation': {}}
            for i, (arm, ridge) in enumerate(CONFIGS):
                scores = np.stack([sum(evidence[k][i] for k in q) for q in qs])
                out['exact'][f'{arm}_ridge{ridge:g}'] = metrics(scores, base, truth, sizes)
            def sum_moment(groups, field):
                return np.stack([sum(moments[k][field].astype(np.float64) for k in ks) for ks in groups])
            refs, helds = [ref[p] for p in owners], [held[p] for p in owners]
            nr, nq = sum_moment(refs, 'n'), sum_moment(qs, 'n')
            for suffix in ('0', '2'):
                d = scale if suffix == '0' else np.tile(scale, 2)
                ur, uq = sum_moment(refs, 'U' + suffix) / d, sum_moment(qs, 'U' + suffix) / d
                uh, nh = sum_moment(helds, 'U' + suffix) / d, sum_moment(helds, 'n')
                out['stability']['U' + suffix] = stability(ur / nr[:, None], uh / nh[:, None])
                if suffix == '0':
                    scores = unit(uq) @ unit(ur).T
                    out['residual_cosine'] = metrics(scores, base, truth, sizes)
                fr = sum_moment(refs, 'F' + suffix) / np.outer(d, d)
                fh = sum_moment(helds, 'F' + suffix) / np.outer(d, d)
                fq = sum_moment(qs, 'F' + suffix) / np.outer(d, d)
                for ridge in (1., 10.):
                    zr = np.linalg.solve(fr + ridge * nr[:, None, None] * np.eye(len(d)), ur[..., None])[..., 0]
                    zh = np.linalg.solve(fh + ridge * nh[:, None, None] * np.eye(len(d)), uh[..., None])[..., 0]
                    out['stability'][f'z{suffix}_ridge{ridge:g}'] = stability(zr, zh)
                    arm = 'A1' if suffix == '0' else 'A2'
                    i = CONFIGS.index((arm, ridge))
                    exact = np.stack([sum(evidence[k][i] for k in q) for q in qs])
                    quad = uq @ zr.T - 0.5 * np.einsum('cd,qde,ce->qc', zr, fq, zr, optimize=True)
                    out['score_approximation'][f'{arm}_ridge{ridge:g}'] = score_comparison(exact, quad, base, truth, sizes, zr)
            result['pools'][pool][str(seed)] = out
            print(pool, seed, {a: {n: round(v[n]['top1'], 4) for n in ('1', '2', '3')} for a, v in out['exact'].items()}, flush=True)
        del moments
    result['elapsed_seconds'] = time.monotonic() - start
    result['sources_sha256'] = SOURCES
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Saved', OUT, 'seconds', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
