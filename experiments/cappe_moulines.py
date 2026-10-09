"""Cappé & Moulines (2009) online EM applied to BKT with one *student sequence* as one observation
(their eq. 15), with the paper's recommendations: gamma_n = gamma0 * n^-alpha, alpha in (1/2, 1),
Polyak-Ruppert averaging of theta (eq. 29), and no M-step during a burn-in (Assumption 1(c), Sec. 4).
Students are fed in the order their sequences complete. Exact per-sequence E-step (forward-backward).
Compared with batch EM on the same data, against known truth. Several random starts."""
import sys, numpy as np, pandas as pd
from bkt_np import *
from compare_em import init_params
from compare_online import simulate_stream

KEYS = ("prior", "learn", "guess", "slip")
truth = dict(prior=0.3, learn=0.15, forget=0.0, guess=0.2, slip=0.1)
ev = simulate_stream(int(sys.argv[1]) if len(sys.argv) > 1 else 40000, 12, truth)
seqs = {}
for u, y in ev:
    seqs.setdefault(u, []).append(y)
order = list(seqs)                      # roughly completion order (simulate_stream sorts by time)
S_all = [np.array(seqs[u]) for u in order]
Yall, Lall = pad(S_all)
per_student, _ = fb_counts(Yall, Lall, truth, per_student=True)  # only used for shapes
err = lambda p: max(abs(p[k] - truth[k]) for k in KEYS)

def online(p0, alpha, gamma0=1.0, burn=20, batch=1, avg_from=0.5):
    p, s = dict(p0), None
    thetas = []
    for b in range(0, len(S_all), batch):
        Y, L = pad(S_all[b:b + batch])
        sb, _ = fb_counts(Y, L, p)
        n = b // batch + 1
        g = min(1.0, gamma0 * n ** -alpha)
        s = sb if s is None else s + g * (sb - s)
        if n * batch >= burn:
            p = m_step(s, p_old=p)
            thetas.append([p[k] for k in KEYS])
    th = np.array(thetas)
    pr = th[int(len(th) * avg_from):].mean(0)          # Polyak-Ruppert average over the second half
    return p, dict(zip(KEYS, pr))

rows = []
for seed in range(3):
    p0 = init_params(seed)
    pb, hist = batch_em(Yall, Lall, p0, iters=500, tol=1e-6)
    rows.append(dict(seed=seed, method="batch EM (%d passes)" % len(hist), err=err(pb)))
    for alpha in (0.6, 0.75, 1.0):
        for batch in (1, 50):
            last, avg = online(p0, alpha, batch=batch)
            rows.append(dict(seed=seed, method=f"online a={alpha} batch={batch}", err=err(last)))
            if alpha < 1:
                rows.append(dict(seed=seed, method=f"online a={alpha} batch={batch} +PR avg", err=err(avg)))
    print("seed", seed, file=sys.stderr)
o = pd.DataFrame(rows)
o.to_csv("/tmp/claude-0/exp/cappe_moulines.csv", index=False)
print(f"{len(S_all)} students, {len(ev)} attempts; max abs param error vs truth (mean over 3 starts, worst start)")
print(o.groupby("method")["err"].agg(["mean", "max"]).sort_values("mean").round(4).to_string())
