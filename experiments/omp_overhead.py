import time, numpy as np
from pyBKT.fit import E_step
from pyBKT.generate import random_model_uni
rng = np.random.default_rng(0)
for n, S in [(5, 2), (1800, 230), (22900, 1115)]:
    L = np.full(S, n // S, np.int64); L[-1] += n - L.sum()
    d = {"data": rng.integers(1, 3, (1, n)).astype(np.int32), "resources": np.ones(n, np.int64),
         "starts": np.concatenate([[1], 1 + np.cumsum(L[:-1])]).astype(np.int64), "lengths": L}
    m = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))
    out = []
    for par in (1, 0):
        b = 9
        for _ in range(30):
            t = time.perf_counter(); E_step.run(d, m, 1, par, {}); b = min(b, time.perf_counter() - t)
        out.append(f"par={par}:{b*1e6:7.0f}us")
    print(f"N={n:6d}", *out)
