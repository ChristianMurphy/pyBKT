"""Does explicit SIMD beat autovectorization for the SIMD-across-students E-step? (planning round 10)

Five ways to compile the same lanes kernel (bkt_lean/src/lanes.rs), all safe Rust:
  plain           [f64; L] arrays, `[bool; L]` masks with `if` selects; build baseline (SSE2 on x86-64)
  blend           [f64; L] arrays, u64 all-ones/zero masks with bitwise blends (vectorizer-friendly); baseline
  plain_dispatch  `plain` compiled inside fearless_simd's runtime dispatch, so LLVM may use AVX2/AVX-512
  blend_dispatch  `blend` inside the runtime dispatch
  fearless        explicit fearless_simd vectors (f64x2/x4/x8) with runtime dispatch
plus `plain`/`blend` from a `-C target-cpu=native` build (not portable: the autovectorizer's ceiling here).

  python autovec.py FIXTURE BUILD IMPL LANES THREADS [REPS]
FIXTURE: as_all or synth5m (make_fixture.py). BUILD: lean, lean_simd or lean_native (inst/<BUILD> from
build_lean.sh; lean_native is built with RUSTFLAGS="-C target-cpu=native"). One fresh process per
measurement; prints one CSV row with the median ns per answer over REPS calls and the largest relative
difference of the counts and log-likelihood from the exact serial kernel (lanes=0).
"""
import os, sys, time
import numpy as np

fixture, build, impl, lanes, threads = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 9
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join("/tmp/claude-0/exp/rust/inst", build))
import bkt_lean

z = np.load(f"/tmp/claude-0/exp/{fixture}.npz")
a = (z["data"], z["resources"], z["starts"], z["lengths"], float(z["prior"]), z["learns"], z["forgets"], z["guesses"], z["slips"])
n = int(z["data"].shape[1])


def flat(r):
    return np.concatenate([np.ravel(r[0]), np.ravel(r[1]), np.ravel(r[2]), [r[3]]])


ref = flat(bkt_lean.e_step(*a, threads=0, alpha="none", lanes=0))
call = lambda: bkt_lean.e_step(*a, threads=threads, alpha="none", lanes=lanes, lane_impl=impl)
got = flat(call())
rel = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))
call()
t = []
for _ in range(reps):
    t0 = time.perf_counter(); call(); t.append(time.perf_counter() - t0)
t = np.array(t) * 1e9 / n
print(f"{fixture},{build},{impl},{lanes},{threads},{np.median(t):.2f},{t.min():.2f},{rel:.1e}", flush=True)
