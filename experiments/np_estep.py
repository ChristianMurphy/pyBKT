"""Vectorized NumPy E-step over all students at once (single template/resource), students sorted by
length so step t touches only the n_t students still active. Validated against the C++ reference counts
stored in the fixtures; timed against the C++ kernel and pyBKT's pure-Python E-step."""
import sys, time, numpy as np

def estep(data, starts, lengths, prior, learn, forget, guess, slip):
    order = np.argsort(-lengths, kind="stable")
    Ls = lengths[order]; st = starts[order] - 1
    S, Tmax = len(Ls), int(Ls[0])
    n_active = np.searchsorted(-Ls, -np.arange(1, Tmax + 1), side="right")  # students with length > t
    A = np.array([[1 - learn, learn], [forget, 1 - forget]])
    B = np.array([[1 - guess, guess], [slip, 1 - slip]])
    Bx = np.vstack([np.ones(2), B.T])  # index by code: 0 missing ->1, 1 incorrect, 2 correct  (Bx[code] -> (2,))
    Bx = np.array([[1.0, 1.0], [B[0, 0], B[1, 0]], [B[0, 1], B[1, 1]]])
    alpha = np.empty((Tmax, S, 2)); cs = np.empty((Tmax, S)); codes = np.empty((Tmax, S), np.int8)
    for t in range(Tmax):
        n = n_active[t]
        c = data[st[:n] + t]; codes[t, :n] = c
        e = Bx[c]
        a = (np.array([1 - prior, prior]) * e) if t == 0 else (alpha[t - 1, :n] @ A) * e
        s = a.sum(1); cs[t, :n] = s; alpha[t, :n] = a / s[:, None]
    # Backward pass in posterior form (see bkt_np.fb_counts): gamma stays <= 1, no overflow.
    trans = np.zeros((2, 2)); emit = np.zeros((3, 2))
    gamma = np.empty((S, 2))                     # gamma of each student at its current step
    for t in range(Tmax - 1, -1, -1):
        n = n_active[t]
        ending = Ls[:n] == t + 1                 # students whose last step is t start here
        if t + 1 < Tmax:
            cont = ~ending
            pred = alpha[t, :n] @ A
            with np.errstate(invalid="ignore", divide="ignore"):
                xi = alpha[t, :n, :, None] * A[None] * (gamma[:n] / pred)[:, None, :]
            xi = np.nan_to_num(xi, nan=0.0)
            trans += xi[cont].sum(0)
            g = np.where(cont[:, None], xi.sum(2), alpha[t, :n])
        else:
            g = alpha[t, :n].copy()
        gamma[:n] = g
        np.add.at(emit, codes[t, :n], g)
        if t == 0:
            init = g.sum(0)
    ll = np.log(cs[np.arange(Tmax)[:, None] < 0] if False else 1).sum()
    ll = sum(np.log(cs[t, :n_active[t]]).sum() for t in range(Tmax))
    return trans, emit[1:], init, ll

if __name__ == "__main__":
    z = np.load(sys.argv[1])
    data = z["data"][0].astype(np.int8)
    args = (data, z["starts"], z["lengths"], float(z["prior"]), z["learns"][0], z["forgets"][0], z["guesses"][0], z["slips"][0])
    trans, emit, init, ll = estep(*args)
    # C++ raw buffers: ref_trans[0] viewed (2,2) is [to? from?]; compare via EM_fit's transpose convention
    rt, re_, ri = z["ref_trans"][0].T, z["ref_emission"][0].T, z["ref_init"].ravel()
    # mine: trans[from, to]; pyBKT after transpose: [to, from] -> compare rt.T
    print("max rel err trans", np.max(np.abs(trans - rt.T) / np.abs(rt.T)), "emit", np.max(np.abs(emit.T - re_) / np.abs(re_)), "init", np.max(np.abs(init - ri) / ri))
    best = 1e9
    for _ in range(3):
        t = time.perf_counter(); estep(*args); best = min(best, time.perf_counter() - t)
    N = len(data)
    print(f"numpy vectorized: {best*1e3:.0f} ms = {best/N*1e9:.0f} ns/attempt (N={N}, students={len(z['lengths'])}, max len={z['lengths'].max()})")
    from pyBKT.fit import E_step
    d = {k: z[k] for k in ("data", "resources", "starts", "lengths")}
    m = {"prior": float(z["prior"]), "learns": z["learns"], "forgets": z["forgets"], "guesses": z["guesses"], "slips": z["slips"]}
    for par in (0, 1):
        b = 1e9
        for _ in range(5):
            t = time.perf_counter(); E_step.run(d, m, 1, par, {}); b = min(b, time.perf_counter() - t)
        print(f"C++ parallel={par}: {b*1e3:.1f} ms = {b/N*1e9:.1f} ns/attempt")
