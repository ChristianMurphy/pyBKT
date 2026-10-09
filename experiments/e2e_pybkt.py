"""End-to-end baseline: pyBKT Model on a CSV, timing each stage in one process."""
import sys, time, json, resource
import numpy as np, pandas as pd
from pyBKT.models import Model
path, num_fits = sys.argv[1], int(sys.argv[2])
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
t = time.perf_counter(); df = pd.read_csv(path, low_memory=False, encoding="latin"); t_read = time.perf_counter() - t
if "original" in df.columns: df = df[df["original"] == 1]
df = df.dropna(subset=["skill_name"])
m = Model(seed=0, num_fits=num_fits, parallel=(sys.argv[3] == "1") if len(sys.argv) > 3 else True)
t = time.perf_counter(); m.fit(data=df.copy()); t_fit = time.perf_counter() - t
t = time.perf_counter(); p = m.predict(data=df.copy()); t_pred = time.perf_counter() - t
print(json.dumps(dict(file=path, rows=len(df), skills=len(m.fit_model), num_fits=num_fits, read_s=round(t_read, 1), fit_s=round(t_fit, 1), predict_s=round(t_pred, 1), peak_mb=rss())))
