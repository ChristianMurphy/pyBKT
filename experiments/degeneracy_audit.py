"""Pardos & Heffernan (2010)-style audit on real data: pyBKT's own EM (C++ E-step, serial) and its own
random_model_uni initialisation, 5 restarts per ASSISTments skill. Classifies each converged fit.
  degenerate : guess + slip >= 1  (a known student is no more likely to be right than an unknown one;
               P&H 2010's degenerate basin, Baker et al. 2008's model degeneracy)
  implausible: guess > 0.5 or slip > 0.5 (outside hmm-scalable's / Corbett-Anderson-style caps)
  boundary   : some parameter < 1e-3 or > 1 - 1e-3 (EM cannot leave it; P&H 2010 Sec. 3.1 note)
Also records which restart pyBKT would keep (best final log-likelihood; note the compiled build
returns it truncated to an integer, so ties go to the first restart)."""
import numpy as np, pandas as pd
from pyBKT.util import data_helper
from pyBKT.fit import EM_fit
from pyBKT.generate import random_model_uni

df = pd.read_csv("/tmp/claude-0/data/as.csv", low_memory=False, encoding="latin")
df = df[df["original"] == 1].dropna(subset=["skill_name"])
datas = data_helper.convert_data(df, ".*")
rows = []
for skill, d in datas.items():
    rand = np.random.RandomState(0)
    for r in range(5):
        m0 = random_model_uni.random_model_uni(1, 1, rand=rand)
        init = dict(prior=float(m0["prior"]), learn=float(m0["learns"][0]), guess=float(m0["guesses"][0]), slip=float(m0["slips"][0]))
        m, ll = EM_fit.EM_fit(m0, d, parallel=False)
        fin = dict(prior=float(m["pi_0"][1][0]), learn=float(m["As"][0, 1, 0]), guess=float(m["guesses"][0]), slip=float(m["slips"][0]))
        rows.append(dict(skill=skill, restart=r, n=d["data"].shape[1], ll_trunc=float(ll[-1][0]), iters=len(ll),
                         **{"init_" + k: v for k, v in init.items()}, **fin))
o = pd.DataFrame(rows)
P = ["prior", "learn", "guess", "slip"]
o["degenerate"] = o.guess + o.slip >= 1
o["implausible"] = (o.guess > 0.5) | (o.slip > 0.5)
o["boundary"] = (o[P].lt(1e-3) | o[P].gt(1 - 1e-3)).any(axis=1)
# pyBKT keeps the first restart with the maximal (truncated) ll
kept = o.loc[o.groupby("skill")["ll_trunc"].idxmax()]
o.to_csv("/tmp/claude-0/exp/degeneracy_audit.csv", index=False)
print(f"{o.skill.nunique()} skills x 5 restarts = {len(o)} fits; init guess+slip max {(o.init_guess+o.init_slip).max():.2f}")
for name, t in [("all restarts", o), ("restart pyBKT keeps", kept)]:
    print(f"{name:22s} degenerate {t.degenerate.mean():6.1%}  implausible {t.implausible.mean():6.1%}  boundary {t.boundary.mean():6.1%}")
spread = o.groupby("skill")[P].agg(lambda s: s.max() - s.min()).max(axis=1)
print(f"skills whose 5 restarts disagree by > 0.1 in some parameter: {(spread > 0.1).sum()} of {len(spread)}")
w = kept[kept.degenerate | kept.implausible]
print("kept fits that are degenerate/implausible (largest skills):")
print(w.sort_values("n", ascending=False)[["skill", "n"] + P].head(8).round(3).to_string(index=False))
