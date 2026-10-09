import resource, time, pandas as pd
from pyBKT.util import data_helper
df = pd.read_csv("/tmp/claude-0/data/as.csv", low_memory=False, encoding="latin")
df = df[df["original"] == 1].dropna(subset=["skill_name"])
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t = time.perf_counter(); d = data_helper.convert_data(df.copy(), ".*"); el = time.perf_counter() - t
print(f"before convert {base:.0f} MB, peak after {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB, {el:.2f}s")
