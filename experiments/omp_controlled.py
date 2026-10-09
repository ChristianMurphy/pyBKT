"""Controlled re-measurement of the compiled E-step's per-call cost, serial vs OpenMP (planning round 6).

One fresh process per (size, mode), so upstream master's sticky omp_set_num_threads(1) can't leak between
measurements. Reports median, p90 and min of 200 calls after 20 warm-up calls.

  python omp_controlled.py <answers> <students> <parallel 0|1>
  ./omp_controlled.sh  (driver: idle, then with 2 and 4 busy background processes)
"""
import sys, time, numpy as np
from pyBKT.fit import E_step
from pyBKT.generate import random_model_uni

n, S, par = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(0)
L = np.full(S, n // S, np.int64); L[-1] += n - L.sum()
d = {"data": rng.integers(1, 3, (1, n)).astype(np.int32), "resources": np.ones(n, np.int64),
     "starts": np.concatenate([[1], 1 + np.cumsum(L[:-1])]).astype(np.int64), "lengths": L}
m = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))
for _ in range(20):
    E_step.run(d, m, 1, par, {})
t = []
for _ in range(200):
    t0 = time.perf_counter(); E_step.run(d, m, 1, par, {}); t.append(time.perf_counter() - t0)
t = np.array(t) * 1e6
print(f"{n},{S},{par},{np.median(t):.1f},{np.percentile(t, 90):.1f},{t.min():.1f}")
