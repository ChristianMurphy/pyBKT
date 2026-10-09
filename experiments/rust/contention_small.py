"""Small E-step calls under CPU contention (planning round 6): is the ~8 ms per-call stall specific to OpenMP's
fixed blocks, or does any threaded E-step stall? Same synthetic shapes as ../omp_controlled.py.

  python contention_small.py ANSWERS STUDENTS VARIANT     # variants: cpp_serial cpp_par rayon_t4 lean_t4 rust_default
rayon_t4 and lean_t4 force threads with 256-answer chunks; rust_default uses bkt_lean's default 65,536-answer
chunks, so a call this small runs on one thread.
"""
import sys, time
import numpy as np

n, S, variant = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(0)
L = np.full(S, n // S, np.int64); L[-1] += n - L.sum()
data = rng.integers(1, 3, (1, n)).astype(np.int32); res = np.ones(n, np.int64)
starts = np.concatenate([[1], 1 + np.cumsum(L[:-1])]).astype(np.int64)
pr, le, fo, gu, sl = 0.3, np.array([0.2]), np.array([0.0]), np.array([0.2]), np.array([0.1])
if variant.startswith("cpp"):
    from pyBKT.fit import E_step
    d = {"data": data, "resources": res, "starts": starts, "lengths": L}
    m = {"prior": pr, "learns": le, "forgets": fo, "guesses": gu, "slips": sl}
    par = 0 if variant == "cpp_serial" else 1
    call = lambda: E_step.run(d, m, 1, par, {})
elif variant == "rayon_t4":
    import bkt_rs
    call = lambda: bkt_rs.e_step(data, res, starts, L, pr, le, fo, gu, sl, threads=4, write_alpha=False, chunk_attempts=256)
elif variant == "lean_t4":
    import bkt_lean
    call = lambda: bkt_lean.e_step(data, res, starts, L, pr, le, fo, gu, sl, threads=4, alpha="none", chunk_attempts=256)
else:
    import bkt_lean
    call = lambda: bkt_lean.e_step(data, res, starts, L, pr, le, fo, gu, sl, threads=4, alpha="none")
for _ in range(20):
    call()
t = []
for _ in range(200):
    t0 = time.perf_counter(); call(); t.append(time.perf_counter() - t0)
t = np.array(t) * 1e6
print(f"{n},{variant},{np.median(t):.1f},{np.percentile(t, 90):.1f},{t.min():.1f}")
