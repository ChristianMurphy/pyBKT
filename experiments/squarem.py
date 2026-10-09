"""EM acceleration for BKT: plain EM vs SQUAREM (Varadhan & Roland 2008, SqS3 with monotone fallback)
and a simple over-relaxed step. Counts E-steps (the expensive part) to reach the same fixed point.

Parameters are mapped to logit space for extrapolation so they stay in (0, 1).
"""
import sys, time
import numpy as np, pandas as pd
from bkt_np import *
from compare_em import init_params

KEYS = ("prior", "learn", "guess", "slip")
logit = lambda x: np.log(x) - np.log1p(-x)
expit = lambda z: 1 / (1 + np.exp(-z))
EPS = 1e-9


def to_vec(p):
    return logit(np.clip(np.array([p[k] for k in KEYS]), EPS, 1 - EPS))


def to_p(v):
    x = expit(v)
    return dict(prior=x[0], learn=x[1], forget=0.0, guess=x[2], slip=x[3])


class Counter:
    def __init__(self, Y, L):
        self.Y, self.L, self.n = Y, L, 0

    def em(self, p):
        """One EM map F(p) and the log-likelihood at p."""
        S, ll = fb_counts(self.Y, self.L, p)
        self.n += 1
        return m_step(S, p_old=p), ll


def plain(c, p, tol, maxe=2000):
    prev = -np.inf
    while c.n < maxe:
        q, ll = c.em(p)
        if abs(ll - prev) < tol:
            return p, ll
        p, prev = q, ll
    return p, ll


def squarem(c, p, tol, maxe=2000, step_max0=1.0, mstep=4.0):
    """SqS3 with SQUAREM's adaptive step cap and a monotone safeguard.

    A cycle is 2 EM maps, an extrapolation capped at |alpha| <= step_max, and 1 stabilising
    EM map. If the stabilised point has lower likelihood than the cycle start, the cycle is
    rejected (keep the plain double EM step) and the cap is reset; accepted maximal steps
    grow the cap by mstep.
    """
    v0 = to_vec(p)
    step_max = step_max0
    q, ll0 = c.em(to_p(v0))
    while c.n < maxe:
        p1 = q
        p2, ll1 = c.em(p1)
        v1, v2 = to_vec(p1), to_vec(p2)
        r, v = v1 - v0, v2 - 2 * v1 + v0
        nr, nv = np.linalg.norm(r), np.linalg.norm(v)
        alpha = -nr / nv if nv > 0 else -1.0
        alpha = max(min(alpha, -1.0), -step_max)
        vnew = v0 - 2 * alpha * r + alpha ** 2 * v
        pn, _ = c.em(to_p(vnew))
        vn = to_vec(pn)
        qn, lln = c.em(to_p(vn))        # ll at the candidate, and the next cycle's first EM map
        if lln < ll1 - 1e-9:
            # reject: continue from the plain double step
            step_max = step_max0
            vn = v2
            qn, lln = c.em(to_p(vn))
        elif alpha == -step_max:
            step_max *= mstep
        if abs(lln - ll0) < tol:
            return to_p(vn), lln
        v0, q, ll0 = vn, qn, lln
    return to_p(v0), ll0


def main(n_skills):
    df = pd.read_csv("/tmp/claude-0/data/as.csv", encoding="latin", low_memory=False)
    df = df[df["original"] == 1].dropna(subset=["skill_name"]).sort_values("order_id")
    vc = df["skill_name"].value_counts()
    skills = list(vc[vc >= 1000].index[:n_skills])
    rows = []
    for skill in skills:
        d = df[df["skill_name"] == skill]
        Y, L = pad([g["correct"].to_numpy() for _, g in d.groupby("user_id", sort=False)])
        for seed in range(3):
            p0 = init_params(seed)
            for name, fn in (("plain_em", plain), ("squarem", squarem)):
                c = Counter(Y, L)
                t = time.perf_counter()
                p, ll = fn(c, p0, tol=1e-4)
                rows.append(dict(skill=skill, seed=seed, method=name, esteps=c.n, ll=ll, seconds=time.perf_counter() - t,
                                 **{k: p[k] for k in KEYS}))
        print(skill, file=sys.stderr)
    out = pd.DataFrame(rows)
    out.to_csv("/tmp/claude-0/exp/squarem.csv", index=False)
    w = out.pivot_table(index=["skill", "seed"], columns="method", values=["esteps", "ll"])
    w["speedup"] = w[("esteps", "plain_em")] / w[("esteps", "squarem")]
    w["ll_gain"] = w[("ll", "squarem")] - w[("ll", "plain_em")]
    pd.set_option("display.width", 250)
    print(w.round(3).to_string())
    print("median E-step speedup", w["speedup"].median().round(2), "mean", w["speedup"].mean().round(2),
          "| ll gain (squarem - plain): min", w["ll_gain"].min().round(3), "median", w["ll_gain"].median().round(3))
    # same fixed point? compare params where both reached similar ll
    pv = out.pivot_table(index=["skill", "seed"], columns="method", values=list(KEYS))
    dif = max((pv[(k, "squarem")] - pv[(k, "plain_em")]).abs().median() for k in KEYS)
    print("median |param diff| (worst param):", round(dif, 5))


if __name__ == "__main__":
    main(int(sys.argv[1]))
