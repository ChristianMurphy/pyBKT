"""Old (scaled-beta) vs new (posterior-form) vectorized NumPy E-step, interleaved in one process."""
import sys, time, numpy as np
import np_estep_old as old, np_estep as new
for f in ("as_all.npz", "synth5m.npz"):
    z = np.load(f); data = z["data"][0].astype(np.int8)
    args = (data, z["starts"], z["lengths"], float(z["prior"]), z["learns"][0], z["forgets"][0], z["guesses"][0], z["slips"][0])
    best = {"old": 9e9, "new": 9e9}
    for _ in range(5):
        for name, mod in (("old", old), ("new", new)):
            t = time.perf_counter(); mod.estep(*args); best[name] = min(best[name], time.perf_counter() - t)
    N = len(data)
    print(f"{f:12s} old {best['old']/N*1e9:6.0f} ns/answer   new {best['new']/N*1e9:6.0f} ns/answer   new/old {best['new']/best['old']:.2f}x")
