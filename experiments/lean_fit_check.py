"""Independent check of bkt_lean.fit against pyBKT's compiled EM_fit: 20 iterations, no early stop,
same start, four ASSISTments skills; serial, 4 threads and fearless lanes."""
import sys, numpy as np, pandas as pd, importlib
from pyBKT.util import data_helper
from pyBKT.fit import EM_fit
from pyBKT.generate import random_model_uni
lean = importlib.import_module("bkt_lean")
df = pd.read_csv("/tmp/claude-0/data/as.csv", low_memory=False, encoding="latin")
df = df[df["original"] == 1].dropna(subset=["skill_name"])
datas = data_helper.convert_data(df, ".*")
sizes = sorted(((d["data"].shape[1], k) for k, d in datas.items()), reverse=True)
out = lean.fit(*[datas[sizes[0][1]][k] for k in ("data", "resources", "starts", "lengths")], 0.3, np.array([0.1]), np.array([0.0]), np.array([0.2]), np.array([0.1]), max_iter=2, tol=-1.0)
print("fit() returns:", type(out).__name__, (list(out.keys()) if isinstance(out, dict) else len(out)))
for n, skill in [sizes[0], sizes[5], sizes[len(sizes) // 2], sizes[-10]]:
    d = datas[skill]
    m0 = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))
    args = [d[k] for k in ("data", "resources", "starts", "lengths")] + [float(m0["prior"]), m0["learns"].copy(), m0["forgets"].copy(), m0["guesses"].copy(), m0["slips"].copy()]
    ref, _ = EM_fit.EM_fit({k: (v.copy() if hasattr(v, "copy") else v) for k, v in m0.items()}, d, tol=-1, maxiter=20, parallel=False)
    refv = np.array([float(ref["prior"]), ref["learns"][0], ref["guesses"][0], ref["slips"][0]])
    row = [f"{skill[:30]:30s} N={n:6d}"]
    for label, kw in [("serial", {}), ("4 threads", dict(threads=4)), ("simd lanes", dict(threads=4, lanes=8, lane_impl="fearless"))]:
        try:
            r = lean.fit(*args, max_iter=20, tol=-1.0, **kw)
            g = r if isinstance(r, dict) else dict(zip(["prior", "learns", "forgets", "guesses", "slips"], r))
            v = np.array([float(np.ravel(g["prior"])[0]), np.ravel(g["learns"])[0], np.ravel(g["guesses"])[0], np.ravel(g["slips"])[0]])
            row.append(f"{label}: " + ("bit-identical" if np.array_equal(v, refv) else f"max rel {np.max(np.abs(v - refv) / np.abs(refv)):.1e}"))
        except Exception as e:
            row.append(f"{label}: {type(e).__name__}: {str(e)[:60]}")
    print(" | ".join(row))
