"""Build shared E-step fixtures (.npz) with reference outputs from the installed C++ pyBKT."""
import numpy as np, pandas as pd, sys
from pyBKT.util import data_helper
from pyBKT.fit import E_step
from pyBKT.generate import random_model_uni

def save(name, d, K, R):
    m = random_model_uni.random_model_uni(R, K, rand=np.random.RandomState(1))
    m["forgets"] = np.full(R, 0.02)
    r = E_step.run(d, m, 1, 0, {})
    np.savez(f"/tmp/claude-0/exp/{name}.npz", data=d["data"].astype(np.int32), resources=np.asarray(d["resources"], np.int64),
             starts=np.asarray(d["starts"], np.int64), lengths=np.asarray(d["lengths"], np.int64),
             prior=m["prior"], learns=m["learns"], forgets=m["forgets"], guesses=m["guesses"], slips=m["slips"],
             ref_trans=r["all_trans_softcounts"], ref_emission=r["all_emission_softcounts"], ref_init=r["all_initial_softcounts"],
             ref_alpha=r["alpha"])
    print(name, "attempts", d["data"].shape[1], "students", len(d["starts"]), "K", K, "R", R)

df = pd.read_csv("/tmp/claude-0/data/as.csv", encoding="latin", low_memory=False)
# all skills concatenated as one big "skill": realistic length distribution
df["skill_name"] = "all"
d = data_helper.convert_data(df, "all")["all"]
save("as_all", d, 1, 1)
d = data_helper.convert_data(pd.read_csv("/tmp/claude-0/data/as.csv", encoding="latin", low_memory=False).assign(skill_name="all"), "all", model_type=[1,0,0,1])["all"]
save("as_all_multi", d, len(d["gs_names"]), len(d["resource_names"]))
rng = np.random.default_rng(0); N, L = 5_000_000, 20; S = N // L
d = {"data": rng.integers(1, 3, size=(1, N)).astype(np.int32), "resources": np.ones(N, np.int64),
     "starts": np.arange(S, dtype=np.int64) * L + 1, "lengths": np.full(S, L, np.int64)}
save("synth5m", d, 1, 1)
