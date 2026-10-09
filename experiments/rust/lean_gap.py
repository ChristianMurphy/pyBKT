"""Where does bkt_lean's SIMD + threads gap vs bkt_rs (rayon) come from?
Same process, interleaved, best of 9, no alpha output. Variants:
  rs           bkt_rs.e_step (numpy crate, zero-copy inputs, rayon)
  lean copy    bkt_lean.e_step (inputs copied to compact codes on every call, std scoped threads)
  lean ds      bkt_lean.e_step_ds on a Dataset converted once (no per-call input copy)"""
import time, numpy as np, bkt_rs, bkt_lean
for f in ("synth5m", "as_all"):
    z = np.load(f"/tmp/claude-0/exp/{f}.npz"); N = z["data"].shape[1]
    arr = (z["data"], z["resources"], z["starts"], z["lengths"])
    par = (float(z["prior"]), z["learns"], z["forgets"], z["guesses"], z["slips"])
    t = time.perf_counter(); ds = bkt_lean.Dataset(*arr); t_ds = time.perf_counter() - t
    V = {}
    for th in (0, 4):
        for chunk in (65536, 16384):
            for lanes in (0, 8):
                k = dict(threads=th, chunk_attempts=chunk, lanes=lanes, lane_impl="fearless")
                tag = f"{'simd L8' if lanes else 'exact  '} thr={th} chunk={chunk//1024}k"
                V[f"rs        {tag}"] = (lambda k=k: bkt_rs.e_step(*arr, *par, write_alpha=False, **k))
                V[f"lean copy {tag}"] = (lambda k=k: bkt_lean.e_step(*arr, *par, alpha="none", **k))
                V[f"lean ds   {tag}"] = (lambda k=k: bkt_lean.e_step_ds(ds, *par, alpha="none", **k))
    best = {k: 9e9 for k in V}
    for _ in range(9):
        for k, fn in V.items():
            t = time.perf_counter(); fn(); best[k] = min(best[k], time.perf_counter() - t)
    print(f"== {f}: {N} answers; Dataset build (one-off) {t_ds*1e3:.1f} ms = {t_ds/N*1e9:.2f} ns/answer")
    for k in V:
        print(f"   {k:42s} {best[k]/N*1e9:6.2f} ns/answer")
