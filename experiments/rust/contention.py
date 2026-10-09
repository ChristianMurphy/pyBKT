"""E-step call time under CPU contention: C++ OpenMP (fixed blocks per thread) vs Rust rayon (work stealing)
vs Rust scoped threads (chunks claimed through an atomic counter). Planning round 6.

  python contention.py FIXTURE VARIANT [REPS]     # one fresh process per measurement; prints CSV
  ./contention.sh                                 # driver: idle, then 2 and 4 busy background processes
FIXTURE is as_all or synth5m (make_fixture.py). Needs the C++ pyBKT plus inst/default (bkt_rs) and
inst/lean (bkt_lean) on PYTHONPATH (build.sh, build_lean.sh).
"""
import sys, time
import numpy as np

fixture, variant = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
z = np.load(f"/tmp/claude-0/exp/{fixture}.npz")
d = {k: z[k] for k in ("data", "resources", "starts", "lengths")}
m = {"prior": float(z["prior"]), **{k: z[k] for k in ("learns", "forgets", "guesses", "slips")}}
a = (z["data"], z["resources"], z["starts"], z["lengths"], float(z["prior"]), z["learns"], z["forgets"], z["guesses"], z["slips"])
CHUNK = 16384   # small enough that every thread gets several chunks on both fixtures

if variant.startswith("cpp"):
    from pyBKT.fit import E_step
    par = 0 if variant == "cpp_serial" else 1
    call = lambda: E_step.run(d, m, 1, par, {})
elif variant.startswith("rayon"):
    import bkt_rs
    call = lambda: bkt_rs.e_step(*a, threads=int(variant[-1]), write_alpha=False, chunk_attempts=CHUNK)
else:
    import bkt_lean
    call = lambda: bkt_lean.e_step(*a, threads=int(variant[-1]), alpha="none", chunk_attempts=CHUNK)

for _ in range(3):
    call()
t = []
for _ in range(reps):
    t0 = time.perf_counter(); call(); t.append(time.perf_counter() - t0)
t = np.array(t) * 1e3
print(f"{fixture},{variant},{np.median(t):.2f},{np.percentile(t, 90):.2f},{t.min():.2f}")
