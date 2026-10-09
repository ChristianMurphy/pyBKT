"""End-to-end EM fit, 20 iterations (no early stop), same start: pyBKT pure-Python EM_fit vs the
vectorized NumPy E-step (np_estep.estep) with a closed-form M-step. Run under the NumPy<2 env with
PYTHONPATH=<repo>/source-py so pyBKT is the pure-Python build."""
import time, numpy as np, pandas as pd
from pyBKT.util import data_helper
from pyBKT.fit import EM_fit
from pyBKT.generate import random_model_uni
from np_estep import estep

df = pd.read_csv("/tmp/claude-0/data/as.csv", low_memory=False, encoding="latin")
df = df[df["original"] == 1].dropna(subset=["skill_name"])
datas = data_helper.convert_data(df, ".*")
sizes = sorted(((d["data"].shape[1], k) for k, d in datas.items()), reverse=True)
picks = [sizes[0], sizes[5], sizes[len(sizes) // 2], sizes[-10]]
tot_py = tot_np = 0.0
for n, skill in picks:
    d = datas[skill]
    m0 = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))
    p = dict(prior=float(m0["prior"]), learn=float(m0["learns"][0]), forget=0.0, guess=float(m0["guesses"][0]), slip=float(m0["slips"][0]))
    t = time.perf_counter(); fm, _ = EM_fit.EM_fit({k: (v.copy() if hasattr(v, "copy") else v) for k, v in m0.items()}, d, tol=-1, maxiter=20, parallel=False); t_py = time.perf_counter() - t
    data8 = d["data"][0].astype(np.int8)
    t = time.perf_counter()
    for _ in range(20):
        trans, emit, init, ll = estep(data8, d["starts"], d["lengths"], p["prior"], p["learn"], p["forget"], p["guess"], p["slip"])
        p = dict(prior=init[1] / init.sum(), learn=trans[0, 1] / trans[0].sum(), forget=trans[1, 0] / trans[1].sum(),
                 guess=emit[1, 0] / emit[:, 0].sum(), slip=emit[0, 1] / emit[:, 1].sum())
    t_np = time.perf_counter() - t
    ref = dict(prior=float(fm["prior"]), learn=float(fm["learns"][0]), guess=float(fm["guesses"][0]), slip=float(fm["slips"][0]))
    diff = max(abs(p[k] - ref[k]) for k in ref)
    tot_py += t_py; tot_np += t_np
    print(f"{skill[:34]:34s} N={n:6d} students={len(d['starts']):5d} maxlen={d['lengths'].max():5d}  pure-Python {t_py:7.2f}s  vectorized {t_np:6.3f}s  {t_py/t_np:6.1f}x  max|param diff| {diff:.1e}")
print(f"total: pure-Python {tot_py:.1f}s, vectorized {tot_np:.2f}s, {tot_py/tot_np:.0f}x")
