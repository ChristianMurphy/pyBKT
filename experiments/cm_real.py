"""Cappé–Moulines student-level online EM on ASSISTments skills vs batch EM, scored on held-out students."""
import sys, numpy as np, pandas as pd
from bkt_np import *
from compare_em import init_params, score
KEYS = ("prior", "learn", "guess", "slip")

def online(seqs, p0, alpha=0.75, batch=50, burn=20, passes=1, avg=False):
    p, s, n, th = dict(p0), None, 0, []
    for _ in range(passes):
        for b in range(0, len(seqs), batch):
            Y, L = pad(seqs[b:b + batch]); sb, _ = fb_counts(Y, L, p)
            sb = sb / len(seqs[b:b + batch])        # per-student scale
            n += 1; g = min(1.0, n ** -alpha)
            s = sb if s is None else s + g * (sb - s)
            if n * batch >= burn:
                p = m_step(s, p_old=p); th.append([p[k] for k in KEYS])
    if avg:
        a = np.array(th)[len(th) // 2:].mean(0); p = dict(p, **dict(zip(KEYS, a)))
    return p

df = pd.read_csv("/tmp/claude-0/data/as.csv", encoding="latin", low_memory=False)
df = df[df["original"] == 1].dropna(subset=["skill_name"]).sort_values("order_id")
vc = df["skill_name"].value_counts(); skills = list(vc[vc >= 3000].index[:int(sys.argv[1])])
rows = []
for skill in skills:
    d = df[df.skill_name == skill]; users = d.user_id.unique()
    test = set(users[np.random.RandomState(42).rand(len(users)) < 0.2])
    tr = [g.correct.to_numpy() for _, g in d[~d.user_id.isin(test)].groupby("user_id", sort=False)]
    Yt, Lt = pad([g.correct.to_numpy() for _, g in d[d.user_id.isin(test)].groupby("user_id", sort=False)])
    Ytr, Ltr = pad(tr)
    for seed in range(3):
        p0 = init_params(seed)
        pb, h = batch_em(Ytr, Ltr, p0, iters=500, tol=1e-6)
        res = {"batch": (pb, len(h))}
        for passes in (1, 3):
            for avg in (False, True):
                res[f"CM a=.75 b=50 x{passes}" + (" +PR" if avg else "")] = (online(tr, p0, passes=passes, avg=avg), passes)
        res["CM a=.75 b=10 x1"] = (online(tr, p0, batch=10), 1)
        for k, (p, passes) in res.items():
            rows.append(dict(skill=skill, seed=seed, method=k, passes=passes, **score(Yt, Lt, p), **{q: p[q] for q in KEYS}))
    print(skill, file=sys.stderr)
o = pd.DataFrame(rows)
b = o[o.method == "batch"].set_index(["skill", "seed"])
o = o.join(b[["ll", "auc"]], on=["skill", "seed"], rsuffix="_b")
o["d_ll"], o["d_auc"] = o.ll - o.ll_b, o.auc - o.auc_b
o.to_csv("/tmp/claude-0/exp/cm_real.csv", index=False)
print(o.groupby("method").agg(d_ll=("d_ll", "mean"), worst_d_ll=("d_ll", "min"), median_d_ll=("d_ll", "median"), d_auc=("d_auc", "mean"), passes=("passes", "mean")).sort_values("d_ll", ascending=False).round(4).to_string())
