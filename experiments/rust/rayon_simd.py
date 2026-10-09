"""Controlled check: do rayon (bkt_rs) / std scoped threads (bkt_lean) and SIMD lanes compose?
E-step without alpha output, ns/answer, best of N runs, all variants interleaved in one process per crate."""
import sys, time, importlib, numpy as np
crate = sys.argv[1]
m = importlib.import_module(crate)
for f in ("as_all", "synth5m"):
    z = np.load(f"/tmp/claude-0/exp/{f}.npz")
    a = (z["data"], z["resources"], z["starts"], z["lengths"], float(z["prior"]), z["learns"], z["forgets"], z["guesses"], z["slips"])
    kw0 = dict(write_alpha=False) if crate == "bkt_rs" else dict(alpha="none")
    variants = {}
    for th in (0, 2, 4):
        variants[f"exact  threads={th}"] = dict(threads=th)
        for L in (4, 8):
            impl = "fearless"
            variants[f"simd L={L} threads={th}"] = dict(threads=th, lanes=L, lane_impl=impl)
    best = {k: 9e9 for k in variants}
    for _ in range(7):
        for k, kw in variants.items():
            t = time.perf_counter(); m.e_step(*a, **kw0, **kw); best[k] = min(best[k], time.perf_counter() - t)
    N = z["data"].shape[1]
    print(f"== {crate} {f} ({N} answers)")
    for k in variants:
        base = best[k.split("threads=")[0] + "threads=0"]
        print(f"   {k:22s} {best[k]/N*1e9:6.2f} ns/answer   scaling vs 1 thread {base/best[k]:4.2f}x")
