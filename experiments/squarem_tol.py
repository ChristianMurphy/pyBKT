"""SQUAREM vs plain EM at several tolerances (the SQUAREM vignette reports 40x at tol 1e-8 on a Poisson
mixture). Also checks which basin each ends in: guess + slip < 1 is the plausible basin in Pardos &
Heffernan (2010); SQUAREM's extrapolation could in principle jump basins (the vignette's power-method
example converges to a different eigenvector with step.min0 = 0.5)."""
import sys, numpy as np, pandas as pd
from bkt_np import pad
from squarem import Counter, plain, squarem, KEYS
from compare_em import init_params

df = pd.read_csv("/tmp/claude-0/data/as.csv", encoding="latin", low_memory=False)
df = df[df["original"] == 1].dropna(subset=["skill_name"]).sort_values("order_id")
vc = df.skill_name.value_counts(); skills = list(vc[vc >= 1000].index[: int(sys.argv[1])])
rows = []
for skill in skills:
    d = df[df.skill_name == skill]
    Y, L = pad([g.correct.to_numpy() for _, g in d.groupby("user_id", sort=False)])
    for tol in (1e-4, 1e-6, 1e-8):
        for seed in range(2):
            p0 = init_params(seed)
            for name, fn in (("plain", plain), ("squarem", squarem)):
                c = Counter(Y, L); p, ll = fn(c, p0, tol=tol, maxe=5000)
                rows.append(dict(skill=skill, tol=tol, seed=seed, method=name, esteps=c.n, ll=ll, gs=p["guess"] + p["slip"], **{k: p[k] for k in KEYS}))
    print(skill, file=sys.stderr)
o = pd.DataFrame(rows); o.to_csv("/tmp/claude-0/exp/squarem_tol.csv", index=False)
w = o.pivot_table(index=["skill", "tol", "seed"], columns="method", values=["esteps", "ll", "gs"])
w["speedup"] = w[("esteps", "plain")] / w[("esteps", "squarem")]
w["dll"] = w[("ll", "squarem")] - w[("ll", "plain")]
w["basin_change"] = (w[("gs", "plain")] < 1) != (w[("gs", "squarem")] < 1)
g = w.groupby(level="tol")
print(pd.DataFrame({"runs": g.size(), "total_esteps_plain": g[("esteps", "plain")].sum(), "total_esteps_squarem": g[("esteps", "squarem")].sum(),
                    "median_speedup": g["speedup"].median(), "worse_by_gt_0.01": g["dll"].apply(lambda s: (s < -0.01).sum()),
                    "basin_changes": g["basin_change"].sum()}).round(2).to_string())
