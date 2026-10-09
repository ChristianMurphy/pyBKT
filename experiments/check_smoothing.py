import numpy as np
from bkt_np import *
rng = np.random.default_rng(3)
seqs = [rng.choice([-1, 0, 1], size=rng.integers(1, 40), p=[.05, .4, .55]) for _ in range(300)]
Y, L = pad(seqs)
p = dict(prior=.3, learn=.12, forget=.03, guess=.22, slip=.09)
S_fb, ll_fb = fb_counts(Y, L, p)
A, B = A_of(p), B_of(p)
S_on, ll_on = np.zeros(D), 0.0
for s in seqs:
    st = StudentStream()
    for y in s:
        ll_on += st.update(int(y), A, B, p["prior"])
    S_on += st.stats()
print("forward-only vs forward-backward: max rel diff", np.max(np.abs(S_on - S_fb) / np.maximum(1, np.abs(S_fb))), "loglik diff", ll_on - ll_fb)
# cross-check fb_counts against pyBKT's C++ E-step
from pyBKT.fit import E_step
N = int(L.sum()); starts = np.concatenate([[1], 1 + np.cumsum(L[:-1])])
d = {"data": (np.concatenate(seqs) + 1).astype(np.int32)[None, :], "resources": np.ones(N, np.int64), "starts": starts.astype(np.int64), "lengths": L.astype(np.int64)}
m = {"prior": p["prior"], "learns": np.array([p["learn"]]), "forgets": np.array([p["forget"]]), "guesses": np.array([p["guess"]]), "slips": np.array([p["slip"]])}
r = E_step.run(d, m, 1, 0, {})
# pyBKT layout: trans[0] as returned, viewed raw -> [[from0to0? ...]]; compare via raw buffer interpretation used by EM_fit (transpose)
T = r["all_trans_softcounts"][0].T; Em = r["all_emission_softcounts"][0].T; I = r["all_initial_softcounts"].ravel()
print("pyBKT trans(after EM_fit transpose)\n", T, "\nmine trans [from,to]\n", S_fb[2:6].reshape(2, 2))
print("pyBKT emit\n", Em, "\nmine emit [state,obs]\n", S_fb[6:10].reshape(2, 2), "\ninit", I, S_fb[:2])
