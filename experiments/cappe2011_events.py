"""Event-level online EM with Cappé (2011)'s recommended step sizes, adapted to many students.

Per student: exact forward smoothing, Cappé (2011) Proposition 1 eqs. (6)-(7) (bkt_np.StudentStream).
Global: each event n yields Delta_n = (student's smoothed statistics after the event) - (before), i.e.
the event's expected statistics, including how it revises that student's earlier answers. The global
normalised statistic follows the stochastic-approximation form of Cappé & Moulines (2009) eq. (15) /
Cappé (2011) eq. (11):
    S_n = S_{n-1} + gamma_n (Delta_n - S_{n-1}),   theta_n = theta_bar(S_n)   (M-step every k events, after n_min)
  gamma_n = 1/n          reproduces our earlier undiscounted stream exactly (Cappé 2011 Sec. 3.1: avoid for HMMs)
  gamma_n = n^-alpha     alpha in (0.5, 0.8) as Cappé (2011) Sec. 3.1 recommends, optional Polyak-Ruppert averaging
  gamma_n = eta (const)  Mongillo & Denève (2008) eq. (2.11) tracking regime, window about 1/eta
The multi-student combination (Delta per event, one global step size) is our adaptation; Cappé (2011)
treats one long sequence."""
import sys, time, numpy as np, pandas as pd
from bkt_np import *
from compare_em import init_params
from compare_online import simulate_stream

KEYS = ("prior", "learn", "guess", "slip")
truth = dict(prior=0.3, learn=0.15, forget=0.0, guess=0.2, slip=0.1)
events = simulate_stream(40000, 12, truth)
err = lambda p: max(abs(p[k] - truth[k]) for k in KEYS)

def run(p0, schedule, every=100, n_min=20, avg_from=None, project=True):
    p = dict(p0); A, B = A_of(p), B_of(p)
    st, contrib = {}, {}
    S, avg, navg = None, None, 0
    for n, (u, y) in enumerate(events, 1):
        s = st.get(u)
        old = contrib.get(u, 0.0)
        if s is None:
            s = st[u] = StudentStream()
        s.update(int(y), A, B, p["prior"])
        new = s.stats(); contrib[u] = new
        delta = new - old
        g = schedule(n)
        S = delta.copy() if S is None else S + g * (delta - S)
        if project:
            # keep S in the set where the M-step is defined (C&M 2009 Sec. 3.1: truncate to a compact
            # subset); the per-event deltas include revisions and can push counts below zero
            S = np.maximum(S, 1e-6)
        if n >= n_min and n % every == 0:
            p = m_step(S, p_old=p); A, B = A_of(p), B_of(p)
            if avg_from is not None and n >= avg_from * len(events):
                v = np.array([p[k] for k in KEYS]); navg += 1
                avg = v if avg is None else avg + (v - avg) / navg
    out = {"last": p}
    if avg is not None:
        out["PR avg"] = dict(p, **dict(zip(KEYS, avg)))
    return out

rows = []
for seed in range(3):
    p0 = init_params(seed)
    for name, sch, avg in [("gamma=1/n (= earlier stream)", lambda n: 1.0 / n, None),
                           ("gamma=n^-0.6", lambda n: n ** -0.6, 0.5),
                           ("gamma=n^-0.7", lambda n: n ** -0.7, 0.5),
                           ("gamma=n^-0.8", lambda n: n ** -0.8, 0.5),
                           ("gamma=3e-5 const (M&D)", lambda n: max(1.0 / n, 3e-5), None),
                           ("gamma=1e-4 const (M&D)", lambda n: max(1.0 / n, 1e-4), None)]:
        t = time.perf_counter()
        for variant, p in run(p0, sch, avg_from=avg).items():
            rows.append(dict(seed=seed, method=f"{name} [{variant}]", err=err(p), seconds=time.perf_counter() - t))
    print("seed", seed, file=sys.stderr)
o = pd.DataFrame(rows); o.to_csv("/tmp/claude-0/exp/cappe2011_events_projected.csv", index=False)
print(f"{len(events)} events, 40k students; max abs param error vs truth over 3 random starts")
print(o.groupby("method")["err"].agg(["mean", "max"]).sort_values("mean").round(4).to_string())
