"""Beck & Chang (2007): Beta (2-category Dirichlet) priors seed the expected counts, so the M-step is
   param = (count_pos + a) / (count_pos + count_neg + a + b)       ("seed the CPT, then add observations")
Compare plain MLE EM against MAP-EM with weak and stronger priors on ASSISTments skills with >= 1000
answers: held-out log-likelihood/AUC (80/20 split by student), how often fits are implausible or stuck
at a boundary, and how much 3 random restarts disagree."""
import sys, numpy as np, pandas as pd
from bkt_np import *
from compare_em import init_params, score

P = ["prior", "learn", "guess", "slip"]
# (a, b) = pseudo-counts for "positive" / "negative". Means: prior .5, learn .2, guess .2, slip .1.
PRIORS = {
    "MLE": None,
    "MAP weak (10 counts each)": dict(prior=(5, 5), learn=(2, 8), guess=(2, 8), slip=(1, 9)),
    "MAP strong (50 counts each)": dict(prior=(25, 25), learn=(10, 40), guess=(10, 40), slip=(5, 45)),
}

def m_step_map(S, pri, p_old):
    if pri is None:
        return m_step(S, p_old=p_old)
    init, trans, emit = S[0:2], S[2:6].reshape(2, 2), S[6:10].reshape(2, 2)
    r = lambda pos, neg, ab: (pos + ab[0]) / (pos + neg + ab[0] + ab[1])
    return dict(prior=r(init[1], init[0], pri["prior"]), learn=r(trans[0, 1], trans[0, 0], pri["learn"]), forget=0.0,
                guess=r(emit[0, 1], emit[0, 0], pri["guess"]), slip=r(emit[1, 0], emit[1, 1], pri["slip"]))

def fit(Y, L, p, pri, iters=300, tol=1e-6):
    prev = -np.inf
    for i in range(iters):
        S, ll = fb_counts(Y, L, p)
        if abs(ll - prev) < tol:
            break
        p, prev = m_step_map(S, pri, p), ll
    return p, i

df = pd.read_csv("/tmp/claude-0/data/as.csv", encoding="latin", low_memory=False)
df = df[df["original"] == 1].dropna(subset=["skill_name"]).sort_values("order_id")
vc = df.skill_name.value_counts(); skills = list(vc[vc >= int(sys.argv[1]) if len(sys.argv) > 1 else 1000].index)
if len(sys.argv) > 2: skills = [sys.argv[2]]
rows = []
for skill in skills:
    d = df[df.skill_name == skill]; users = d.user_id.unique()
    test = set(users[np.random.RandomState(42).rand(len(users)) < 0.2])
    Ytr, Ltr = pad([g.correct.to_numpy() for _, g in d[~d.user_id.isin(test)].groupby("user_id", sort=False)])
    Yt, Lt = pad([g.correct.to_numpy() for _, g in d[d.user_id.isin(test)].groupby("user_id", sort=False)])
    for name, pri in PRIORS.items():
        for seed in range(3):
            p, it = fit(Ytr, Ltr, init_params(seed), pri)
            if not all(np.isfinite(p[k]) for k in P):
                rows.append(dict(skill=skill, n=len(d), method=name, seed=seed, iters=it, failed=True, **{k: p[k] for k in P}))
                continue
            rows.append(dict(skill=skill, n=len(d), method=name, seed=seed, iters=it, **{k: p[k] for k in P}, **score(Yt, Lt, p)))
    print(skill, file=sys.stderr)
    pd.DataFrame(rows).to_csv("/tmp/claude-0/exp/map_em_partial%s.csv" % ("_" + str(abs(hash(sys.argv[2]))) if len(sys.argv) > 2 else ""), index=False)
o = pd.DataFrame(rows)
print("fits with non-finite parameters:", int(o.get("failed", pd.Series(dtype=bool)).fillna(False).sum()))
o = o[~o.get("failed", pd.Series(False, index=o.index)).fillna(False).astype(bool)]
o["implausible"] = (o.guess > 0.5) | (o.slip > 0.5)
o["boundary"] = (o[P].lt(1e-3) | o[P].gt(1 - 1e-3)).any(axis=1)
o["never_learns"] = o.learn < 0.02
o.to_csv("/tmp/claude-0/exp/map_em.csv", index=False)
base = o[o.method == "MLE"].set_index(["skill", "seed"])[["ll", "auc"]]
o = o.join(base, on=["skill", "seed"], rsuffix="_mle")
o["d_ll"], o["d_auc"] = o.ll - o.ll_mle, o.auc - o.auc_mle
spread = o.groupby(["method", "skill"])[P].agg(lambda s: s.max() - s.min()).max(axis=1).gt(0.1).groupby("method").mean()
s = o.groupby("method").agg(heldout_ll=("ll", "mean"), d_ll_vs_mle=("d_ll", "mean"), worst_d_ll=("d_ll", "min"), d_auc=("d_auc", "mean"),
                           implausible=("implausible", "mean"), boundary=("boundary", "mean"), never_learns=("never_learns", "mean"), iters=("iters", "mean"))
s["restarts_disagree"] = spread
pd.set_option("display.width", 250)
print(f"{o.skill.nunique()} skills, 3 random starts each")
print(s.round(4).to_string())
