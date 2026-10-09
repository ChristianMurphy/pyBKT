"""bkt_lean.fit vs pyBKT EM_fit (C++ E-step, serial), 20 iterations with tol=-1, plus
convergence behaviour with the default tol (int-truncated loglike in compat mode)."""
import numpy as np, copy, common, bkt_lean
import pyBKT.fit.EM_fit as EM
from pyBKT.fit import E_step as CPP

def model_from(m):
    m = copy.deepcopy(m); l, f = m['learns'], m['forgets']
    m['As'] = np.stack([np.array([[1 - l[i], f[i]], [l[i], 1 - f[i]]]) for i in range(len(l))])
    return m

class PerIndex:  # C++ E-step with fixed=-1 arrays -> per-resource / per-subpart parameters
    @staticmethod
    def run(d, m, n, p, fixed):
        R, K = len(m['learns']), len(m['guesses'])
        return CPP.run(d, m, n, p, dict(learn=-np.ones(R), forget=-np.ones(R), guess=-np.ones(K), slip=-np.ones(K)))

def compare(tag, d, m, rust_kw, cpp_mod=CPP, maxiter=20, tol=-1):
    EM.E_step = cpp_mod
    pm, plls = EM.EM_fit(model_from(m), d, tol=tol, maxiter=maxiter, parallel=False)
    EM.E_step = CPP
    r = bkt_lean.fit(d['data'], d['resources'], d['starts'], d['lengths'], m['prior'], m['learns'], m['forgets'], m['guesses'], m['slips'],
                     max_iter=maxiter, tol=tol, **rust_kw)
    R, K = len(m['learns']), len(m['guesses'])
    rows = []
    for k, rv in [('prior', r['prior']), ('learns', r['learns']), ('forgets', r['forgets']), ('guesses', r['guesses']), ('slips', r['slips']),
                  ('As', np.array(r['As']).reshape(R, 2, 2)), ('emissions', np.array(r['emissions']).reshape(K, 2, 2)), ('pi_0', np.array(r['pi_0']).reshape(2, 1))]:
        a, b = np.asarray(pm[k], float), np.asarray(rv, float)
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))
        rows.append(f'{k}:{"bit" if np.array_equal(a, b) else f"rel {rel:.1e}"}')
    lr = np.array(r['log_likelihoods']); lp = plls.ravel()
    print(f'{tag}: iterations pyBKT={len(lp)} rust={len(lr)}; ' + ' '.join(rows) +
          f'; ll (pyBKT int) last={lp[-1]:.0f} rust last={lr[-1]!r}')

f = common.load('as_all'); d, m = common.dm(f)
compare('as_all K=1 R=1, 20 it, tol=-1, rust serial', d, m, dict(threads=0))
compare('as_all K=1 R=1, 20 it, tol=-1, rust 4 threads (chunked)', d, m, dict(threads=4))
compare('as_all K=1 R=1, 20 it, tol=-1, rust lanes=4 4 threads', d, m, dict(threads=4, lanes=4))
compare('as_all default tol=1e-3 maxiter=100, rust compat_int_loglike=True', d, m, dict(threads=0, compat_int_loglike=True), maxiter=100, tol=1e-3)
compare('as_all default tol=1e-3 maxiter=100, rust float loglike', d, m, dict(threads=0), maxiter=100, tol=1e-3)
# generic K>1, R>1 synthetic
rng = np.random.default_rng(3); K, R, S = 3, 4, 400
lengths = rng.integers(5, 80, S).astype(np.int64); starts = (np.concatenate([[0], np.cumsum(lengths)[:-1]]) + 1).astype(np.int64); N = int(lengths.sum())
data = np.zeros((K, N), np.int32); which = rng.integers(0, K, N); data[which, np.arange(N)] = rng.integers(1, 3, N)
d2 = dict(data=data, resources=rng.integers(1, R + 1, N).astype(np.int64), starts=starts, lengths=lengths)
m2 = dict(prior=0.3, learns=rng.random(R) * 0.3, forgets=rng.random(R) * 0.05, guesses=rng.random(K) * 0.3, slips=rng.random(K) * 0.2)
compare('synthetic K=3 R=4 (C++ with per-index fixed=-1), 20 it, rust serial', d2, m2, dict(threads=0), cpp_mod=PerIndex)
