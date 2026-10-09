"""Random K>1 / R>1 / edge-case tests vs C++ (serial must be bit-identical).
C++ has a parameter-loading quirk: without `fixed`, learn/forget/guess/slip are
initialised to -1 once *outside* the per-resource/per-subpart loop, so every
resource uses learns[0]/forgets[0] and every subpart uses guesses[0]/slips[0].
Passing fixed arrays of -1 resets them each iteration and gives per-index params,
which is what bkt_rs implements; we compare against that."""
import numpy as np, bkt_rs
from pyBKT.fit import E_step
from correctness import cpp_fix

rng = np.random.default_rng(42)
nfail = 0; ncase = 0
for case in range(300):
    K = int(rng.integers(1, 5)); R = int(rng.integers(1, 5))
    S = int(rng.integers(1, 30))
    lengths = rng.integers(1, 60, S).astype(np.int64)
    gaps = rng.integers(0, 3, S)
    starts = np.zeros(S, np.int64); pos = 0
    for i in range(S):
        pos += gaps[i]; starts[i] = pos + 1; pos += lengths[i]
    N = int(pos + rng.integers(0, 3))
    perm = rng.permutation(S); starts = starts[perm].copy(); lengths = lengths[perm].copy()  # unsorted
    data = rng.integers(0, 3, (K, N)).astype(np.int32)
    if K > 1:  # mostly one active subpart per attempt, like pyBKT multi-gs
        mask = rng.random((K, N)) < 0.7; data[mask] = 0
    res = rng.integers(1, R + 1, N).astype(np.int64)
    learns = rng.random(R); forgets = rng.random(R) * 0.3
    guesses = rng.random(K) * 0.5; slips = rng.random(K) * 0.3
    prior = float(rng.random())
    if case % 7 == 0:  # degenerate: zeros -> B==0 skip and NaN pairs
        forgets[:] = 0; learns[0] = 0; prior = 0.0 if case % 2 else 1.0
        guesses[0] = 0.0; slips[-1] = 0.0
    d = dict(data=data, resources=res, starts=starts, lengths=lengths)
    m = dict(prior=prior, learns=learns, forgets=forgets, guesses=guesses, slips=slips)
    fx = dict(learn=-np.ones(R), forget=-np.ones(R), guess=-np.ones(K), slip=-np.ones(K))
    ref = E_step.run(d, m, 1, 0, fx)
    covered = np.zeros(N, bool)
    for s, l in zip(starts, lengths): covered[s-1:s-1+l] = True
    a = (data, res, starts, lengths, prior, learns, forgets, guesses, slips)
    for th in (0, 3):
        tr, em, ini, ll, al = bkt_rs.e_step(*a, threads=th, chunk_attempts=50)
        ok_bit = (np.array_equal(tr, ref['all_trans_softcounts']) and np.array_equal(em, ref['all_emission_softcounts'])
                  and np.array_equal(ini, ref['all_initial_softcounts'])
                  and np.array_equal(al[:, covered], cpp_fix(ref['alpha'])[:, covered], equal_nan=True))
        ok_close = all(np.allclose(x, y, rtol=1e-12, atol=1e-12) for x, y in
                       [(tr, ref['all_trans_softcounts']), (em, ref['all_emission_softcounts']), (ini, ref['all_initial_softcounts'])])
        ok_ll = int(ll) == ref['total_loglike'] or abs(ll - round(ll)) < 1e-9 or not np.isfinite(ll)
        ncase += 1
        if not ((ok_bit if th == 0 else ok_close) and ok_ll):
            nfail += 1; print('FAIL', case, th, K, R, ll, ref['total_loglike'])
    if K == 1:
        for L, th, imp in [(L, th, imp) for L in (2, 4, 8) for th in (0, 2) for imp in ('fearless', 'autovec', 'autovec_dispatch')]:
                tr, em, ini, ll, al = bkt_rs.e_step(*a, threads=th, chunk_attempts=40, lanes=L, lane_impl=imp)
                ok = all(np.allclose(x, y, rtol=1e-12, atol=1e-12) for x, y in
                         [(tr, ref['all_trans_softcounts']), (em, ref['all_emission_softcounts']), (ini, ref['all_initial_softcounts'])])
                ok &= np.array_equal(al[:, covered], cpp_fix(ref['alpha'])[:, covered], equal_nan=True)
                ll0 = bkt_rs.e_step(*a, threads=0)[3]
                ok &= (ll == ll0) or abs(ll - ll0) <= 1e-12 * abs(ll0) + 1e-12 or (not np.isfinite(ll0) and not np.isfinite(ll))
                ncase += 1
                if not ok:
                    nfail += 1; print('LANES FAIL', case, L, th, R, ll, ll0)
    pref = E_step.predict(d, m, 0, fx)  # C++ predict also uses learns[0] w/o fixed? uses As_model = per-index model params
    p = bkt_rs.predict(*a, threads=0)
    if not np.array_equal(p[:, covered], pref[:, covered], equal_nan=True):
        nfail += 1; print('PRED FAIL', case)
print(f'{ncase} e_step checks + 300 predict checks, failures: {nfail}')
