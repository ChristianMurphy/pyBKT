# Rust E-step for pyBKT: prototype report

This was an experiment. Nothing in `/home/user/pyBKT` was modified (`git status` is clean), and nothing was committed or pushed. Everything is under `/tmp/claude-0/exp/rust/`.

Measurement machine: a shared VM with 4 vCPUs (Xeon Sapphire Rapids, 2.1 GHz, AVX-512) and 15 GB RAM. Other agents' jobs ran at the same time, with a 1-minute load average of 1.4 to 3.3 during the benchmarks. Read the caveats before quoting the multithreaded numbers.

## TL;DR

* **What it is.** `bkt_rs` is a pyo3 + numpy + rayon + fearless_simd extension that reimplements `E_step.cpp`. It has `#![forbid(unsafe_code)]` at the crate root, so the crate itself contains no `unsafe`.
* **The exact serial path is bit-identical to the C++ serial path.** This holds on both fixtures for trans, emission, init, alpha and loglike (C++ truncates loglike to an int).
  * `predict` is bit-identical to `E_step.predict`.
  * Twenty iterations of pyBKT's own `EM_fit` with the Rust E-step swapped in give bit-identical fitted parameters.
  * 1,860 randomized e_step checks (K = 1..4, R = 1..4, unsorted and gapped layouts, degenerate parameters) all pass. Serial is bit-exact; chunked and SIMD are within 1e-12.
* **Speed** (ns/attempt; C++ serial is 47.9 on as_all and 53.2 on synth5m, C++ with 4 OpenMP threads is 28.7 and 14.0):

| | as_all | synth5m |
|---|---|---|
| exact serial | 22.5 (2.1x faster than C++ serial) | 21.5 (2.5x) |
| deterministic rayon, 4 threads | 7.7 (3.7x faster than C++ parallel) | 5.7 (2.5x) |
| SIMD across students (fearless_simd), serial | 6.3 (7.7x faster than C++ serial) | 5.2 (10.2x) |
| SIMD across students, 4 threads | 2.3 (12.7x faster than C++ parallel) | 1.45 (9.7x faster than C++ parallel, 37x faster than C++ serial) |

* **The rayon reduction is deterministic.** Results are bit-identical across runs and for 1, 2, 3, 4, 8 and 16 threads. The SIMD results are additionally the same for every runtime ISA (SSE2, SSE4.2, AVX2, AVX-512). C++ `parallel=1` gave 1-3 distinct results in 20 runs on as_all and 6-7 on synth5m.
* **Going fully safe costs about 0-5%** on the exact kernels, which is within noise. The fearless_simd SIMD kernel is *faster* than the old `unsafe #[target_feature]` version: 6.3 vs 8.5 ns on as_all, 5.2 vs 8.3 on synth5m.
  * One real regression appeared and was fixed. Allocating outputs as a Rust `Vec` instead of a numpy array lost numpy's huge-page madvise and cost about 5 ns/attempt in extra page faults. Outputs are now allocated by numpy and written through the safe `PyReadwriteArray::as_slice_mut`.

## What was built (file list)

| path | what |
|---|---|
| `bkt_rs/Cargo.toml`, `bkt_rs/Cargo.lock`, `bkt_rs/pyproject.toml` | Crate: edition 2021, `rust-version = "1.89"`. Release profile: opt-level 3, fat LTO, codegen-units 1, strip. Built with maturin; pyo3 `abi3-py39`. |
| `bkt_rs/src/lib.rs` | `#![forbid(unsafe_code)]`. pyo3 bindings: `e_step`, `predict`, `simd_level`, `lane_utilization`. Dtype dispatch with no conversion copies (data: int8/int32/int64; resources: int64/int32/uint16). Releases the GIL during compute. Outputs are allocated by numpy and written via `readwrite().as_slice_mut()`. |
| `bkt_rs/src/kernel.rs` | Exact scalar kernel (generic K and R, specialised for K==1 and R==1) and predict kernel. Chunk planner and a deterministic `drive()`: each chunk gets a disjoint output slice via `split_at_mut`, or, when sequences are unordered or overlapping, a chunk-local buffer that is scattered afterwards. Cached rayon pools. |
| `bkt_rs/src/lanes.rs` | SIMD-across-students kernel, written once against a `LaneOps` trait. Implementations: `Plain<L>` (`[f64; L]` arrays left to the autovectorizer) and `Fs2/Fs4/Fs8` (fearless_simd `f64x2/f64x4/f64x8`). Runtime ISA selection via `fearless_simd::dispatch!`. |
| `archive/bkt_rs_v1_unsafe/` | The previous version (raw output pointers, `get_unchecked`, `unsafe #[target_feature]` multiversioning). Kept **only** as the baseline for the cost-of-safety column; not a deliverable. |
| `build.sh` | Builds wheels into `wheels/{default,native}` and unpacks them into `inst/{default,native}`. Pick one with `PYTHONPATH`. |
| `rerun_all.sh` | Rebuilds everything, including the v1 baseline, then runs all checks, the full benchmark and the tables. |
| `py/common.py` | Fixture loading. |
| `py/correctness.py` | Rust vs C++ in the same process, both fixtures, every exact variant plus predict. |
| `py/generic_test.py` | 300 random cases: K 1..4, R 1..4, unsorted starts with gaps, degenerate parameters (B==0 skip, NaN pairs, -inf loglike). Covers exact serial, chunked and every SIMD implementation, L and thread count. |
| `py/determinism.py` | Hashes over thread counts and repeated runs; C++ parallel run to run. |
| `py/ll_accuracy.py` | Loglike vs an exactly rounded `math.fsum` reference. |
| `py/lanes_check.py` | SIMD variants vs the reference; bit equality across implementation, ISA level and thread count. |
| `py/em_compat_check.py`, `py/bkt_rs_compat.py` | Drop-in shim with the `E_step` interface; 20 EM iterations through pyBKT's `EM_fit`. |
| `py/bench_one.py`, `py/bench_all.py`, `py/summarize.py` | Benchmark (a fresh process per measurement) and table generation. |
| `results/` | Raw outputs. `bench_v2.jsonl` and `bench_v2.md` are the final sweep; `bench.jsonl` is the earlier v1-only sweep. |
| `venv/` | uv venv (CPython 3.13) with maturin, numpy, pandas and scikit-learn. pyBKT is built from a *copy* of the repo (`pybkt_copy/`) so C++ and Rust run in the same process. |

## Rebuild and rerun from scratch

Versions used:
* rustc/cargo 1.97.0 at `/root/.cargo/bin` (MSRV of the crate: 1.89)
* maturin 1.x from PyPI, CPython 3.13 via uv, numpy 2, gcc 13.3 for the C++ pyBKT
* from `bkt_rs/Cargo.lock`: pyo3 0.29.3, numpy (crate) 0.29.0, ndarray 0.17.2, rayon 1.12.0, rayon-core 1.13.0, crossbeam-deque 0.8.8, fearless_simd 1.1.0

```sh
# 0. fixtures (already present). make_fixture.py needs the C++ pyBKT (/tmp/claude-0/venv) and /tmp/claude-0/data/as.csv
/tmp/claude-0/venv/bin/python /tmp/claude-0/exp/make_fixture.py    # -> /tmp/claude-0/exp/{as_all,synth5m}.npz with ref_* from E_step.run(d,m,1,0,{})

# 1. venv: maturin + C++ pyBKT built from a COPY of the repo (the repo itself is never touched)
cd /tmp/claude-0/exp/rust
export PATH=/root/.cargo/bin:/root/.local/bin:$PATH
uv venv venv -p 3.13
uv pip install -p venv/bin/python maturin numpy pandas scikit-learn requests setuptools wheel
mkdir pybkt_copy && (cd /home/user/pyBKT && tar --exclude=./build --exclude='*.egg-info' --exclude=./.git -cf - .) | tar -xf - -C pybkt_copy
(cd pybkt_copy && uv pip install -p ../venv/bin/python --no-build-isolation .)

# 2. build. RUSTFLAGS unset = portable x86-64 baseline (SSE2) + runtime dispatch;
#    the second wheel uses RUSTFLAGS="-C target-cpu=native" (NOT portable, for comparison only)
./build.sh            # runs: cd bkt_rs && maturin build --release -i ../venv/bin/python -o ../wheels/<flavor>

# 3. all checks + benchmark (~35 min) + tables -> results/
./rerun_all.sh

# single pieces
cd py
PYTHONPATH=../inst/default ../venv/bin/python correctness.py
PYTHONPATH=../inst/default ../venv/bin/python bench_one.py synth5m rs_e_fearless_L8_t4    # one measurement -> JSON line
../venv/bin/python summarize.py ../results/bench_v2.jsonl
```

`BKT_RS_ISA=sse2|sse4_2|avx2` caps the SIMD level of the lane kernels. Unset, they use the best available level, which is AVX-512 here. `bkt_rs.simd_level()` reports the chosen level.

## Python API

```python
bkt_rs.e_step(data, resources, starts, lengths, prior, learns, forgets, guesses, slips,
              threads=0, write_alpha=True, chunk_attempts=65536, lanes=0, lane_impl="fearless")
  -> (trans (R,2,2) [r][from][to], emission (K,2,2) [k][obs][state], init (2,1), loglike: float, alpha (2,N) | None)
bkt_rs.predict(data, resources, starts, lengths, prior, learns, forgets, guesses, slips,
               threads=0, pred_learns=None, pred_forgets=None, chunk_attempts=65536) -> (2,N)
```

* `threads`
  * `0`: the exact serial path, in C++ order.
  * `>= 1`: deterministic chunks on a rayon pool of that size.
* `lanes` in {2, 4, 8}: SIMD across students. K == 1 only; always chunked.
* `lane_impl`
  * `"fearless"`: explicit vectors.
  * `"autovec_dispatch"`: plain arrays inside `dispatch!`.
  * `"autovec"`: plain arrays, baseline ISA only.

## Semantics notes (found while reproducing the C++)

* **Output layouts.** `E_step.run` returns raw Eigen column-major buffers that Python labels as C-order arrays:
  * `trans[r]` is `[from][to]`. pyBKT's `EM_fit` transposes it.
  * `emission[k]` is `[obs][state]`.
  * **`alpha` is labelled `(2, N)` but holds `N x 2` interleaved data**, so `alpha_true = raw.ravel().reshape(N, 2).T`. The fixtures' `ref_alpha` is this raw buffer.
  * `predict` uses a RowMajor map, so its `(2, N)` layout is real.

  bkt_rs returns the same count layouts and a genuine `(2, N)` alpha. Alpha was compared against the de-interleaved C++ alpha.
* **`total_loglike` is truncated to an integer** by `PyLong_FromLong(double)`. pyBKT's convergence test `|ll_i - ll_{i-1}| < tol` therefore effectively compares integers. bkt_rs returns the float. A drop-in replacement would change `EM_fit` iteration counts unless it truncates too.
* **C++ parameter-loading bug.** `learn`, `forget`, `guess` and `slip` are initialised to -1 *once*, outside the per-resource and per-subpart loops. Without `fixed`, every resource therefore uses `learns[0]`/`forgets[0]`, and every subpart uses `guesses[0]`/`slips[0]`. This was verified: changing `learns[1]` does not change the C++ output. bkt_rs uses per-index parameters. For K>1 and R>1 comparisons the C++ was called with `fixed` arrays of -1, which forces a per-index reset. This deserves its own upstream fix.
* C++ leaves positions that no sequence covers uninitialised (`new[]`). bkt_rs sets them to 0.
* bkt_rs does not read `resources` when R == 1. When R > 1 it validates `starts`, `lengths` and `resources` up front and raises `ValueError`; C++ has undefined behaviour on bad input.

## Correctness

The reference is `E_step.run(d, m, 1, 0, {})` from a C++ build of the same source, in the same process. It first reproduces the fixtures' `ref_*` arrays **bit for bit**. Values are max relative error (max absolute in parentheses).

| variant | as_all trans / emission / init | synth5m trans / emission / init | alpha | bit-identical? |
|---|---|---|---|---|
| (a) exact serial, any input dtype | 0 / 0 / 0 | 0 / 0 / 0 | 0 | **yes**, all outputs; loglike also equal after int truncation |
| (b) chunked rayon, 1-16 threads | 1.8e-13 / 4.3e-13 / 9.5e-15 (1.3e-8 / 8.5e-8 / 1.5e-11) | 9.6e-14 / 1.3e-14 / 1.9e-14 (2.9e-7 / 2.6e-8 / 1.2e-9) | 0 | alpha yes; counts differ only by summation order |
| (c) no alpha | equal to (a) serial / (b) chunked | same | n/a | serial yes |
| (c) predict, serial and 4 threads | n/a | n/a | 0 vs `E_step.predict` | **yes** |
| (d) int8 data + int32/uint16 resources | same bits as int32/int64 | same | 0 | yes |
| (e) SIMD lanes, every impl, L and ISA | 2.9e-13 / 4.4e-13 / 9.7e-15 | 9.6e-14 / 1.3e-14 / 1.9e-14 | 0 | alpha yes; counts and loglike differ by summation order |
| for context: C++ parallel vs C++ serial | 3.7e-13 / 2.3e-13 / 9.0e-15 | 1.1e-13 / 2.7e-14 / 1.9e-14 | | no, and it varies run to run |

**Why serial is bit-identical.** The Rust uses the exact C++ operation order:
* forward step: `a00*x0 + a01*x1`, then multiply by the likelihood, `norm = x0 + x1`, two true divisions, `ll += ln(norm)`;
* backward pair: `((A(i,j)*alpha_j)*gamma_i)/p_i`, with NaN mapped to 0;
* sums: one running accumulator in the same sequence and time order.

Eigen's 2x2 GEMV gives the same bits because a two-term IEEE sum is commutative. Multiplying by 1.0 when B == 0 is exact. Neither side contracts to FMA: GCC `-O2` has no `-march`, and rustc never contracts implicitly (even with target-cpu=native). Both call glibc `log`. The chunked and SIMD variants differ only in summation order: per-chunk or per-lane partial sums, reduced in a fixed order.

**Loglike accuracy** vs an exactly rounded `math.fsum` of every log(norm) (`ll_accuracy.py`), as absolute error:

| | C++ / exact serial order | chunked | SIMD lanes |
|---|---|---|---|
| as_all | -3.8e-8 | -7.5e-9 | -4.1e-10 |
| synth5m | **-2.2e-5** | +1.6e-7 | -1.9e-9 |

The C++ single running sum is the least accurate of the three. The SIMD loglike differs from the C++ value by 5.9e-12 relative on synth5m, which is above the 1e-12 target. It is within 5e-16 of the exact sum, so that 5.9e-12 is error in the reference. The SIMD path accumulates the product of the normalisers per lane and splits off the binary exponent each step, so it takes one `ln` per sequence instead of one per attempt.

**Randomized** (`generic_test.py`): 1,860 e_step checks and 300 predict checks, 0 failures.

**EM end to end** (`em_compat_check.py`): pyBKT `EM_fit` on as_all, 20 iterations, once with the C++ E-step and once with the Rust shim. `prior, learns, forgets, guesses, slips, As, emissions, pi_0` are all bit-identical.

## Determinism

| | result |
|---|---|
| Rust exact serial, 5 runs | 1 distinct output hash per fixture |
| Rust chunked, threads in {1,2,3,4,8,16} x 5 runs | **1 hash per fixture**, covering every output including alpha and loglike |
| Rust SIMD, threads {0,1,4} x ISA {SSE2, SSE4.2, AVX2, AVX-512} x impl {fearless, autovec, autovec_dispatch} | **1 hash per (fixture, L)**, equal to the v1 lanes output: independent of threads, CPU features and implementation, because there is no FMA and the lane reduction order is fixed |
| C++ `parallel=1`, 20 runs | 1-3 distinct results on as_all (varies between sessions), 6-7 on synth5m |

The reduction tree depends only on `chunk_attempts` (default 65,536 attempts per chunk) and L, never on the number of threads.

## Benchmarks

Method:
* Each cell is a fresh Python process: load inputs, record `ru_maxrss`, one warm-up call (validated against `ref_*` at 1e-12), then 9 timed calls.
* The number is the **min** over 3 rounds x 9 runs. In parentheses is the max/min ratio of the three per-round minimums, i.e. run-to-run spread across processes and time.
* "x C++ serial" and "x C++ par" are speedups against C++ `E_step.run(d,m,1,0,{})` and `(d,m,1,1,{})` on 4 OpenMP threads. Predict rows are compared against `E_step.predict`.
* Peak RSS is measured above the inputs.
* Builds:
  * "safe, portable" is the deliverable: default RUSTFLAGS, x86-64 baseline plus runtime dispatch.
  * "native" uses `-C target-cpu=native` and is not portable.
  * "v1 (unsafe)" is the archived pre-`forbid(unsafe_code)` build.


#### as_all (449,220 attempts): C++ and exact-path variants

ns/attempt = best per-process min over 3 rounds x 9 runs; (x) = max/min of the 3 per-round mins.

| variant | safe, portable build | x C++ serial | x C++ par | safe, target-cpu=native | v1 (unsafe), portable | safe/unsafe time ratio | peak RSS over inputs, MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| C++ E_step.run(...,parallel=0) | 47.90 (1.08) | 1.00 | 0.60 | - | - | - | 7.1 |
| C++ E_step.run(...,parallel=1), 4 OpenMP threads | 28.72 (1.63) | 1.67 | 1.00 | - | - | - | 7.5 |
| C++ E_step.predict serial | 40.77 (1.05) | 1.00 | 0.62 | - | - | - | 7.1 |
| C++ E_step.predict parallel | 25.07 (1.25) | 1.63 | 1.00 | - | - | - | 7.5 |
| (a) exact serial port | 22.46 (1.05) | 2.13 | 1.28 | 21.19 (1.08) | 21.66 (1.04) | 1.04 | 7.5 |
| (b) chunked rayon, 1 thread | 22.53 (1.08) | 2.13 | 1.27 | 21.34 (1.02) | 21.68 (1.02) | 1.04 | 7.6 |
| (b) chunked rayon, 2 threads | 12.89 (1.04) | 3.72 | 2.23 | 12.57 (1.08) | 12.83 (1.11) | 1.00 | 7.7 |
| (b) chunked rayon, 4 threads | 7.70 (1.36) | 6.22 | 3.73 | 7.12 (1.09) | 7.61 (1.20) | 1.01 | 7.7 |
| (c) no alpha, serial | 21.38 (1.01) | 2.24 | 1.34 | 20.72 (1.03) | 20.93 (1.04) | 1.02 | 0.7 |
| (c) no alpha, 4 threads | 6.41 (1.61) | 7.48 | 4.48 | 6.25 (1.58) | 6.93 (1.29) | 0.92 | 0.9 |
| (c) predict, serial | 10.13 (1.02) | 4.03 | 2.48 | 10.21 (1.05) | 10.12 (1.01) | 1.00 | 7.3 |
| (c) predict, 4 threads | 4.15 (2.14) | 9.82 | 6.04 | 7.98 (1.12) | 4.11 (2.44) | 1.01 | 7.5 |
| (d) int8 data + uint16 resources, serial | 22.10 (1.03) | 2.17 | 1.30 | 21.18 (1.00) | 21.44 (1.01) | 1.03 | 7.5 |
| (d) int8/uint16, no alpha, 4 threads | 6.46 (1.07) | 7.41 | 4.45 | 6.16 (1.19) | 6.47 (1.43) | 1.00 | 1.1 |

#### as_all: SIMD across students (no alpha unless noted; runtime dispatch picked AVX-512 on this CPU)

| impl | L | serial ns/att | x C++ serial | 4 threads ns/att | x C++ par | native build serial | native build 4 thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| fearless_simd f64xL, dispatch! | 2 | 10.00 (1.02) | 4.79 | 3.24 (2.69) | 8.87 | - | - |
| fearless_simd f64xL, dispatch! | 4 | 6.25 (1.01) | 7.66 | 2.27 (1.53) | 12.67 | 6.21 (1.01) | 6.35 (1.18) |
| fearless_simd f64xL, dispatch! | 8 | 6.33 (1.05) | 7.57 | 3.33 (1.83) | 8.63 | 6.26 (1.56) | 2.02 (2.74) |
| plain [f64;L] inside dispatch! | 2 | 12.17 (1.01) | 3.94 | 5.98 (1.42) | 4.81 | - | - |
| plain [f64;L] inside dispatch! | 4 | 9.08 (1.05) | 5.27 | 4.69 (1.76) | 6.13 | - | - |
| plain [f64;L] inside dispatch! | 8 | 13.06 (1.01) | 3.67 | 4.71 (1.34) | 6.09 | - | - |
| plain [f64;L], no dispatch (SSE2 baseline) | 2 | 13.44 (1.01) | 3.56 | 4.64 (1.30) | 6.19 | - | - |
| plain [f64;L], no dispatch (SSE2 baseline) | 4 | 12.68 (1.01) | 3.78 | 6.09 (1.41) | 4.72 | 9.10 (1.19) | 3.43 (2.05) |
| plain [f64;L], no dispatch (SSE2 baseline) | 8 | 13.76 (1.07) | 3.48 | 4.79 (1.87) | 6.00 | 13.61 (1.37) | 5.92 (1.34) |
| v1 plain arrays + unsafe #[target_feature] | 4 | 8.50 (1.40) | 5.64 | 5.04 (1.66) | 5.70 | - | - |
| v1 plain arrays + unsafe #[target_feature] | 8 | 9.97 (1.18) | 4.80 | 3.52 (2.32) | 8.16 | - | - |
| fearless, L=4, 4 threads, *with* alpha | 4 | | | 3.72 (2.19) | 7.72 | | |

#### as_all: fearless lanes, serial, ISA capped with BKT_RS_ISA (portable build)

| ISA | L=2 | L=4 | L=8 |
|---|---:|---:|---:|
| sse2 | 10.76 (1.00) | 7.87 (1.05) | 9.17 (1.02) |
| sse4_2 | 11.51 (1.03) | 8.46 (1.14) | 9.43 (1.08) |
| avx2 | 11.36 (1.08) | 6.97 (1.02) | 6.81 (1.02) |
| avx512 (auto) | 10.00 (1.02) | 6.25 (1.01) | 6.33 (1.05) |

#### synth5m (5,000,000 attempts): C++ and exact-path variants

ns/attempt = best per-process min over 3 rounds x 9 runs; (x) = max/min of the 3 per-round mins.

| variant | safe, portable build | x C++ serial | x C++ par | safe, target-cpu=native | v1 (unsafe), portable | safe/unsafe time ratio | peak RSS over inputs, MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| C++ E_step.run(...,parallel=0) | 53.17 (1.05) | 1.00 | 0.26 | - | - | - | 76.2 |
| C++ E_step.run(...,parallel=1), 4 OpenMP threads | 14.00 (1.06) | 3.80 | 1.00 | - | - | - | 76.0 |
| C++ E_step.predict serial | 47.41 (1.04) | 1.00 | 0.26 | - | - | - | 76.2 |
| C++ E_step.predict parallel | 12.49 (1.09) | 3.79 | 1.00 | - | - | - | 76.0 |
| (a) exact serial port | 21.48 (1.03) | 2.48 | 0.65 | 20.95 (1.01) | 20.72 (1.02) | 1.04 | 78.9 |
| (b) chunked rayon, 1 thread | 21.94 (1.01) | 2.42 | 0.64 | 21.17 (1.02) | 20.93 (1.10) | 1.05 | 78.8 |
| (b) chunked rayon, 2 threads | 11.26 (1.01) | 4.72 | 1.24 | 10.94 (1.03) | 10.52 (1.15) | 1.07 | 79.0 |
| (b) chunked rayon, 4 threads | 5.70 (1.01) | 9.33 | 2.46 | 5.46 (1.04) | 5.43 (1.08) | 1.05 | 79.0 |
| (c) no alpha, serial | 19.29 (1.01) | 2.76 | 0.73 | 18.49 (1.04) | 18.74 (1.04) | 1.03 | 0.7 |
| (c) no alpha, 4 threads | 5.00 (1.09) | 10.63 | 2.80 | 4.74 (1.07) | 4.81 (1.04) | 1.04 | 0.8 |
| (c) predict, serial | 9.15 (1.01) | 5.18 | 1.37 | 9.12 (1.04) | 8.81 (1.03) | 1.04 | 78.6 |
| (c) predict, 4 threads | 2.46 (1.03) | 19.30 | 5.09 | 2.41 (1.03) | 2.37 (1.13) | 1.04 | 78.8 |
| (d) int8 data + uint16 resources, serial | 22.14 (1.09) | 2.40 | 0.63 | 20.65 (1.03) | 20.94 (1.04) | 1.06 | 78.9 |
| (d) int8/uint16, no alpha, 4 threads | 4.96 (1.22) | 10.73 | 2.83 | 4.88 (1.06) | 5.06 (1.05) | 0.98 | 0.8 |

#### synth5m: SIMD across students (no alpha unless noted; runtime dispatch picked AVX-512 on this CPU)

| impl | L | serial ns/att | x C++ serial | 4 threads ns/att | x C++ par | native build serial | native build 4 thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| fearless_simd f64xL, dispatch! | 2 | 9.97 (1.04) | 5.33 | 2.61 (1.15) | 5.37 | - | - |
| fearless_simd f64xL, dispatch! | 4 | 6.14 (1.04) | 8.66 | 1.69 (1.18) | 8.28 | 6.11 (1.14) | 1.64 (1.13) |
| fearless_simd f64xL, dispatch! | 8 | 5.19 (1.03) | 10.24 | 1.45 (1.09) | 9.67 | 5.24 (1.04) | 1.47 (1.18) |
| plain [f64;L] inside dispatch! | 2 | 12.43 (1.02) | 4.28 | 3.17 (1.03) | 4.41 | - | - |
| plain [f64;L] inside dispatch! | 4 | 8.89 (1.02) | 5.98 | 2.34 (1.04) | 5.98 | - | - |
| plain [f64;L] inside dispatch! | 8 | 10.98 (1.01) | 4.84 | 2.83 (1.06) | 4.95 | - | - |
| plain [f64;L], no dispatch (SSE2 baseline) | 2 | 13.86 (1.06) | 3.84 | 3.56 (1.05) | 3.93 | - | - |
| plain [f64;L], no dispatch (SSE2 baseline) | 4 | 12.19 (1.05) | 4.36 | 3.14 (1.14) | 4.46 | 8.80 (1.03) | 2.30 (1.13) |
| plain [f64;L], no dispatch (SSE2 baseline) | 8 | 11.64 (1.02) | 4.57 | 3.09 (1.04) | 4.54 | 11.39 (1.02) | 2.99 (1.02) |
| v1 plain arrays + unsafe #[target_feature] | 4 | 8.25 (1.05) | 6.44 | 2.21 (1.19) | 6.33 | - | - |
| v1 plain arrays + unsafe #[target_feature] | 8 | 8.35 (1.11) | 6.36 | 2.16 (1.09) | 6.48 | - | - |
| fearless, L=4, 4 threads, *with* alpha | 4 | | | 2.69 (1.20) | 5.21 | | |

#### synth5m: fearless lanes, serial, ISA capped with BKT_RS_ISA (portable build)

| ISA | L=2 | L=4 | L=8 |
|---|---:|---:|---:|
| sse2 | 10.85 (1.04) | 7.91 (1.01) | 7.72 (1.04) |
| sse4_2 | 11.61 (1.01) | 8.28 (1.04) | 7.89 (1.01) |
| avx2 | 11.43 (1.02) | 6.83 (1.11) | 5.58 (1.03) |
| avx512 (auto) | 9.97 (1.04) | 6.14 (1.04) | 5.19 (1.03) |

1-min load average during the runs: min 1.37, median 2.08, max 2.55

### Reading the benchmark

* **Exact kernels.** Serial is 2.1-2.5x faster than C++ serial. It is *faster than C++ with 4 OpenMP threads* on as_all and close to it on synth5m.
  * The C++ serial loop pays for Eigen `Map`s of dynamic size, a heap-sized GEMV on 2x2 blocks, `As.block(...)` copies, and per-sequence `std::vector` resizing. The Rust inner loop is straight scalar code: the K==1 and R==1 specialisations keep the accumulators in registers.
  * C++ parallel scales badly on as_all (1.7x on 4 threads) for two reasons: it splits *sequences* into 4 equal-count blocks, and as_all is heavily skewed (the top 1% of students hold 17% of attempts).
  * The Rust chunks are cut by attempt count. Going from 1 to 4 threads gives 2.9x on as_all and 3.8x on synth5m.
* **Skipping alpha.** This saves the 2N-double output: 76 MiB of RSS at 5M attempts, and roughly 10-25% of the time when threaded.
* **Narrow dtypes.** int8 data and uint16 resources make no measurable time difference: the kernel is latency bound, not bandwidth bound. They do cut input memory 4x for data and 4x for resources.
* **SIMD across students.** This is the largest win. fearless_simd f64x4/f64x8 at 6.2/5.2 ns/attempt serial is 3.5-4x faster than the exact scalar path.
  * Interleaving 4-8 independent students hides the divide and the add/multiply latency chains of the 2-state recurrence. It also removes the per-attempt `ln`.
  * fearless_simd beats plain `[f64; L]` arrays inside `dispatch!` (9.0 ns) because LLVM does not vectorize the masked selects or the u64 exponent manipulation well. Without `dispatch!` the arrays stay at SSE2 and run at 12-13 ns.
  * `target-cpu=native` gives nothing extra over runtime dispatch: fearless_simd already selects AVX-512.
  * By ISA cap at L=8: SSE2 7.7 ns, AVX2 5.6 ns, AVX-512 5.2 ns on synth5m.
* **Stability.**
  * Single-threaded numbers repeat within about 5% across rounds.
  * 4-thread numbers on as_all, where one call takes 1-3 ms, spread up to 2.7x because of other jobs on this 4-vCPU VM. The synth5m 4-thread numbers spread 1.0-1.2x.
  * C++ parallel on synth5m measured 14.0 ns in this sweep and 21.1 ns in an earlier sweep under different load. All "x C++ par" ratios carry that uncertainty.

## Cost of going fully safe (`#![forbid(unsafe_code)]`)

What was replaced:

| v1 (`unsafe`) | now (safe) |
|---|---|
| `get_unchecked` | ordinary indexing on per-sequence sub-slices (`&data[st..st+T]`), so most bounds checks are hoisted or elided |
| raw `*mut f64` output pointers shared across threads with `unsafe impl Send/Sync` | disjoint `&mut [f64]` per chunk from `split_at_mut`; a chunk-local buffer plus a serial scatter when sequences are unordered or overlapping |
| `PyArray::new` / `.data()` | `PyArray2::zeros(...).readwrite().as_slice_mut()` |
| `unsafe fn` with `#[target_feature]` plus `is_x86_feature_detected!` | `fearless_simd::dispatch!` |

Measured cost:
* **Exact kernels:** safe/unsafe time ratio 0.92-1.07 across all rows of both fixtures, i.e. within noise (the largest steady difference is about 4-5% on synth5m serial).
* **SIMD kernel:** the safe fearless version is faster than v1: 6.25 vs 8.50 ns on as_all and 5.19 vs 8.35 ns on synth5m, serial.
* **Pitfall found on the way:** a first safe version allocated outputs as `vec![0.0; 2N]` and handed them to numpy without a copy. That was 1.2-1.5x slower *only* when outputs were written (synth5m serial 25.6 vs 20.8 ns; predict 13.3 vs 8.9 ns). The cause is page faults: numpy's allocator calls `madvise(MADV_HUGEPAGE)` on large arrays and a Rust `Vec` does not, which costs about 1 µs per 4 KiB page, or about 5 ns per attempt at 16 bytes per attempt. Allocating through numpy and writing through `as_slice_mut` fixed it.
* **`unsafe` in dependencies** (expected and fine): pyo3 and pyo3-ffi (FFI with CPython), numpy and ndarray (raw array views), rayon and rayon-core with crossbeam (work stealing), and fearless_simd (intrinsics behind safe, token-checked wrappers; about 120 `unsafe` occurrences, mostly target-feature calls). The `bkt_rs` crate itself has none, and the compiler enforces that.

## SIMD questions

**SIMD within one student's 2-state recurrence: not worthwhile.** Each attempt depends on the previous one through `x -> A x (2x2) -> *lik -> norm = x0 + x1 -> x / norm`. With f64x2, `A x` needs a broadcast or shuffle of `x`, the normalisation needs a horizontal add, and the divide stays on the critical path. The latency chain (about 30 cycles per step) does not get shorter, and vector width cannot help a serial dependence. Evidence: the `target-cpu=native` build, where LLVM is free to SLP-vectorize the 2-wide operations with AVX-512, runs the exact kernel at the same speed (21.2 vs 22.5 ns as_all, 21.0 vs 21.5 synth5m). Vectorising *across* students turns the latency-bound chain into throughput-bound work and gives 3.5-4x per core.

**Skewed lengths and lane utilisation** (`bkt_rs.lane_utilization`, = useful lane-steps / (L x group max length)):

| fixture | lengths (min / median / mean / p99 / max) | L=2 sorted / unsorted | L=4 sorted / unsorted | L=8 sorted / unsorted |
|---|---|---|---|---|
| as_all | 1 / 23 / 106.5 / 1,122 / 6,154 | 0.986 / 0.760 | 0.936 / 0.582 | 0.842 / 0.460 |
| synth5m | all 20 | 0.9997 | 0.999 | 0.999 |

Without sorting, L=8 on as_all would waste 54% of lane-steps. Within each 64k-attempt chunk the kernel sorts sequences by length (descending, ties broken by index, so the order is deterministic) and groups L consecutive sequences. That restores 84-99% utilisation; the losses are mostly in the long tail, where neighbouring lengths differ a lot. Sorting over a single global chunk would reach 0.94 for L=8 but would serialise the work.

The skew is visible in timings: on as_all, L=8 is no faster than L=4 (6.3 vs 6.3 ns), while on synth5m L=8 wins (5.2 vs 6.1 ns). The remaining gap between as_all and synth5m comes from the scalar gathers (data codes, and A for R>1) and per-group setup, which f64x8 does not amortise as well on short or ragged groups. Possible improvements: a gather with an explicit mask instead of clamped indices, or refilling a lane with the next sequence as soon as its current one ends instead of masking.

## Packaging assessment (shipping this as pyBKT's compiled backend)

* **Build backend.** Use `maturin` (as here) through `pyproject.toml`, or `setuptools-rust` if the existing `setup.py` must stay. With pyo3 `abi3-py39`, **one wheel per platform** covers CPython 3.9 through 3.13+, instead of one wheel per Python version today. The numpy crate binds the numpy C API at runtime, so one wheel works with numpy 1.x and 2.x.
* **CI matrix.** Use `PyO3/maturin-action` or cibuildwheel:
  * manylinux2014 x86_64 and aarch64 (Docker or `--zig`), optionally musllinux;
  * macOS x86_64 and arm64 (or universal2);
  * Windows x64, optionally arm64.

  That is about 6-8 wheels in total. The local build here was tagged `manylinux_2_34` because it was built on Ubuntu 24.04; release wheels must be built in the manylinux2014 container. aarch64 was type-checked here (`cargo check --target aarch64-unknown-linux-gnu`; fearless_simd uses NEON with no runtime detection needed) but not run.
* **No OpenMP.** rayon replaces it. The `.so` links only libc, libm and libgcc_s. The current C++ needs libgomp on Linux and Homebrew `libomp` plus `-Xpreprocessor -fopenmp` on macOS, which is a known source of install failures.
* **Portable wheels without losing SIMD.** Default RUSTFLAGS plus `fearless_simd::dispatch!` gives AVX2 or AVX-512 at runtime with identical results on every CPU. Do **not** ship `-C target-cpu=native` builds.
* **Building from sdist** needs a Rust toolchain at least as new as the MSRV (pip will not install one). Today's `setup.py` silently falls back to pure Python when the C++ build fails. There are two ways to keep that behaviour:
  * (a) `setuptools-rust` with `RustExtension(..., optional=True)`, plus the existing Python E-step as the fallback;
  * (b) split into `pyBKT` (pure Python) and an optional `pyBKT-rs` wheel imported with `try:`.

  maturin itself has no "optional extension" mode. With wheels for all major platforms, very few users would build from sdist.
* **MSRV: 1.89**, set by fearless_simd 1.1.0. pyo3 0.29 needs 1.83. Confirmed: `cargo +1.89 check` passes and `cargo +1.88` is rejected. The v1 build without fearless_simd built on 1.83.
* **Binary size.**

| build | stripped `.so` | wheel |
|---|---|---|
| portable (this deliverable) | 7.1 MB | 2.4 MB |
| target-cpu=native | 3.3 MB | 1.1 MB |
| v1 (no fearless) | 2.4 MB | 0.9 MB |
| current C++ `E_step` `.so` | 4.4 MB (unstripped) | n/a |

  The size comes from monomorphisation: 3 SIMD impls x 3 widths x 5 ISA levels x 9 input-dtype combinations x {R==1, R>1}. Shipping only the fearless f64x4/f64x8 kernels and a smaller dtype set would bring it to roughly 2-3 MB.
* **API changes a real integration needs:**
  * a compat layer for the C++ output conventions (or fixing `EM_fit` and `M_step` to use sane layouts);
  * a decision on the int-truncated loglike and on the `learns[0]` parameter bug (both above);
  * an `EM_fit` option to request `write_alpha=False`.

  `py/bkt_rs_compat.py` shows the shim.

## Caveats

* The VM is shared and was busy (other agents' benchmarks, load average 1.4-3.3 with 4 vCPUs). Single-threaded results are solid to about 5%. 4-thread results, especially on as_all where calls take milliseconds, and all C++-parallel baselines carry ±20-50% uncertainty. See the spread columns.
* Peak RSS comes from `ru_maxrss` deltas in a fresh process (KiB granularity). Overhead is dominated by the alpha or prediction output (16 bytes per attempt). The no-alpha variants need under 1 MiB.
* Only x86-64 (Sapphire Rapids) was measured. AVX-512 behaviour on other CPUs (for example, frequency effects on older Skylake-SP) was not checked; fearless_simd only selects AVX-512 on Ice Lake-class or newer.
* The SIMD kernel is K == 1 only. Multi-subpart models (K > 1) use the exact or chunked kernels.
* `predict` with a non-empty `fixed` (where filtering parameters differ from prediction parameters) is supported through `pred_learns`/`pred_forgets` but was only tested indirectly, through equal parameters.
* The `as_all_multi` fixture listed in `make_fixture.py` was not present. K>1 and R>1 coverage comes from the randomized tests against the C++.

---

# Part 2: `bkt_lean`, a minimal-dependency crate with zero-copy marshalling, plus Rust-side EM

Goal: keep `#![forbid(unsafe_code)]` in our own code, avoid `unsafe`-heavy dependencies where that is practical, and cut marshalling costs. `bkt_rs` is kept unchanged except for the clippy fixes, for comparison.

## What changed

| | `bkt_rs` | `bkt_lean` (default) | `bkt_lean --features simd` |
|---|---|---|---|
| direct dependencies | pyo3, numpy, rayon, fearless_simd | **pyo3 only**: `default-features = false`, features `macros`, `extension-module`, `abi3-py311` | pyo3, fearless_simd (optional) |
| crates linked into the `.so`, incl. itself | 20 | **5**: bkt_lean, pyo3, pyo3-ffi, libc, once_cell | 6 |
| all crates incl. build deps and proc-macros | 30 | 14 | 15 |
| non-comment `unsafe` lines in linked deps (ours = 0) | 3,704 | 2,457 | 2,560 |
| threads | rayon pool | `std::thread::scope`. Chunks are claimed through an `AtomicUsize` and results are sent over `mpsc` to the calling thread, which stores them by chunk index, so the reduction order is fixed | same |
| SIMD lanes | fearless_simd + plain arrays, with dispatch | plain `[f64; L]` arrays, build baseline only (SSE2 on x86-64) | + fearless_simd f64x2/4/8 with runtime dispatch |
| Python minimum | 3.9 (abi3) | **3.11** (abi3). The buffer protocol is only in the limited API from 3.11; pyo3 gates `pyo3::buffer` on `any(not(Py_LIMITED_API), Py_3_11)` | 3.11 |
| MSRV | 1.89 | **1.83** (checked with `cargo +1.83 check`) | 1.89 |
| stripped `.so` / wheel | 7.2 MB / 2.4 MB | 3.0 MB / 0.96 MB | 9.6 MB / 2.8 MB |

Unsafe counts come from `audit/deps_unsafe.py` and are in `audit/deps_unsafe.md`. Method: lines containing the `unsafe` keyword after stripping `//`, `/* */` comments and string literals, in all `.rs` files of each crate (tests, benches and examples excluded). Per crate:
* pyo3: 1,253
* libc: 799 (mostly `extern` declarations)
* pyo3-ffi: 358
* ndarray: 395
* numpy: 234
* crossbeam: 245
* rayon and rayon-core: 150
* fearless_simd: 103 (the method differs slightly from the 112 quoted in the task)
* matrixmultiply: 95
* once_cell: 47

Dropping numpy, ndarray, rayon and their trees removes about 1,250 linked `unsafe` lines. The remaining 95% is pyo3, pyo3-ffi and libc, which any CPython extension written in Rust needs.

**RustSec** (`audit/rustsec_check.py` against the clone at `/tmp/claude-0/exp/depaudit/advisory-db`, commit 550efd3, 2026-10-08):
* Every package in both `Cargo.lock` files was checked, including build-time and optional dependencies.
* **No advisory affects the resolved versions**, and there are no informational (unmaintained or unsound) notices.
* The ones that would apply to older versions are already patched here: pyo3 RUSTSEC-2026-0013/0176/0177 (fixed in 0.28.2/0.29.0) and crossbeam-epoch RUSTSEC-2026-0204 (fixed in 0.9.20).
* The matcher was sanity-checked: it does flag pyo3 0.28.1, crossbeam-epoch 0.9.18 and once_cell 1.0.0.

**clippy:** `cargo clippy --all-targets -- -D warnings` is clean on bkt_rs and on bkt_lean with and without `simd`.
* bkt_rs fixes:
  * `v != v` became `v.is_nan()`, the deny-level error at kernel.rs:410;
  * two `vec![0..n]` became `std::iter::once(..).collect()`;
  * doc-list indentation;
  * `from_bits` was renamed to `f64_from_bits`;
  * index loops became `zip` iterators.
* Outputs are **bit-identical** before and after: `correctness.txt` is byte-for-byte identical, the lanes hashes are identical across all impl × L × ISA combinations, the chunked hash is unchanged (`b550668678e13265`), and the generic tests still pass with 0 failures.

## Zero-copy marshalling: what pyo3's safe API allows

* **Inputs.** `PyBuffer::<T>::as_slice(py)` returns `&[ReadOnlyCell<T>]`, a true zero-copy view.
  * `ReadOnlyCell<T>` wraps `UnsafeCell<T>`, so it is **`!Sync`**. The slice cannot go to scoped threads, and it cannot cross `py.detach`, which requires `Send`.
  * Its lifetime is tied to the `Python<'py>` token, so it is GIL-bound. pyo3 does this deliberately: other Python code could mutate the buffer.
  * Safe options, all measured:
    * `borrow`: zero copy, GIL held, calling thread only. Used for serial work, including the lane kernels run inline.
    * `copy`: `PyBuffer::to_vec`, a memcpy into an owned `Vec<T>` that is `Sync`.
    * `codes`: convert to a `Vec<u8>` of 0/1/2 codes, plus `Vec<u32>` resources when R > 1, on every call.
    * `Dataset`: a `#[pyclass(frozen)]` holding the converted codes, so conversion happens **once per fit**. `fit()` does this internally.
* **Outputs.** Counts are tiny and returned as Python lists. Alpha and predictions (16 bytes per attempt) can go to:
  * **(a)** `PyByteArray::new_with(16N, |buf| ...)`: `split_at_mut` per chunk; workers write `f64::to_ne_bytes` through `chunks_exact_mut(8)`; then `np.frombuffer(ba).reshape(2, N)`, which is zero copy and writable.
  * **(b)** caller-allocated `np.empty((2, N))` passed as a writable `PyBuffer<f64>`, giving `as_mut_slice` → `&[Cell<f64>]`. This is `!Send`, so serial runs write directly, and threaded runs send finished chunks over `mpsc` to the calling thread, which writes them while the other chunks are still computing (pipelined).
  * **(c)** baseline: `Vec<f64>`, then a copy into a new `bytes` object.
* `bkt_lean` keeps the GIL during compute, as the C++ module does (it never releases it). Worker threads never touch Python.

## Correctness of `bkt_lean`

Sources: `results/lean_check.txt`, `results/lean_simd_check.txt`, `results/fit_check.txt`. There are 0 failures in both builds.

* **Exact serial is bit-identical to C++** on both fixtures, for every combination of:
  * input: copy, borrow, codes, Dataset;
  * alpha: none, bytearray, bytes, numpy;
  * dtype: int32/int64 and int8/uint16.

  `predict` is bit-identical to `E_step.predict` for threads 0 and 4 and every input and output mode.
* **Chunked (4 threads):**
  * one hash over threads {1,2,3,4,8} × output modes;
  * **bit-identical to `bkt_rs` chunked** (same chunk plan);
  * max relative error vs C++: 4.3e-13 (as_all), 9.6e-14 (synth5m).
* **Lanes:** plain and fearless, each L ∈ {2,4,8}, are each identical over threads {0,1,4} and bit-identical to `bkt_rs` lanes. Max relative error vs C++ is at most 4.4e-13.
* **Random generic checks:** 1,783 (default build) and 1,966 (simd build), all passing.
  * Inputs: K = 1..4, R = 1..4, unsorted or gapped layouts, which exercise the pack-and-drain output path, and degenerate parameters.
  * Serial runs are bit-exact; chunked and lanes runs are within 1e-12.
* **`fit()`** compared with pyBKT `EM_fit` using the C++ E-step:

| case | iterations (pyBKT / Rust) | parameters |
|---|---|---|
| as_all, 20 iterations, tol=-1, Rust serial | 20 / 20 | **bit-identical**: prior, learns, forgets, guesses, slips, As, emissions, pi_0 |
| same, Rust 4 threads (chunked) | 20 / 20 | max rel 8.5e-13 |
| same, Rust lanes L4, 4 threads | 20 / 20 | max rel 7.8e-13 |
| synthetic K=3, R=4, Rust serial (C++ run with per-index `fixed = -1` arrays) | 20 / 20 | **bit-identical** (generic M-step) |
| as_all, default tol=1e-3, Rust `compat_int_loglike=True` | 24 / 24 | **bit-identical**: the integer-truncated stopping test is reproduced |
| as_all, default tol=1e-3, Rust float loglike (default) | 24 / **47** | differs by 0.3-2% (more iterations) |

  The last row shows the cost of the C++ truncation. Because pyBKT compares *integer* loglikelihoods, EM stops at 24 iterations on as_all, when it has not converged to tol=1e-3. The float test runs 47 iterations, and the parameters move another 0.3-2%. `compat_int_loglike=True` replicates pyBKT exactly; the default is the float.

## Benchmarks: `bkt_lean` vs `bkt_rs` vs C++

Method:
* As before: a fresh process per measurement, the minimum over 3 rounds (in parentheses, the max/min of the per-round minimums), 9 runs per process for E-step and predict, and 3 for 20-iteration fits.
* Minor page faults per call come from `ru_minflt`.
* 1-minute load average during the sweep: 0.85-2.90, median 1.9, on 4 vCPUs.
* Every E-step call includes the full Python-visible work: argument parsing, buffer acquisition, input copy or conversion, output allocation, and `np.frombuffer` + reshape.

#### as_all (449,220 attempts): E-step

ns/attempt = min over 3 rounds x 9 runs (max/min of per-round mins). Minor page faults per call (`ru_minflt`).

| variant | ns/attempt | x C++ serial | x C++ par | minor faults / call |
|---|---:|---:|---:|---:|
| C++ E_step.run parallel=0 | 49.07 (1.04) | 1.00 | 0.59 | 195 |
| C++ E_step.run parallel=1 (4 OpenMP) | 29.10 (1.03) | 1.69 | 1.00 | 195 |
| bkt_rs exact serial, alpha | 21.98 (1.02) | 2.23 | 1.32 | 24 |
| bkt_rs exact serial, no alpha | 21.46 (1.07) | 2.29 | 1.36 | 0 |
| bkt_rs 4 thr, alpha | 7.43 (1.17) | 6.61 | 3.92 | 25 |
| bkt_rs 4 thr, no alpha | 6.57 (1.35) | 7.47 | 4.43 | 6 |
| bkt_rs fearless L4 serial | 6.27 (1.02) | 7.83 | 4.64 | 0 |
| bkt_rs fearless L8 4 thr | 4.10 (1.55) | 11.96 | 7.09 | 40 |
| lean serial, input copy (to_vec) | 21.48 (1.13) | 2.28 | 1.35 | 49 |
| lean serial, input borrow (&[ReadOnlyCell], 0 copy) | 21.22 (1.01) | 2.31 | 1.37 | 0 |
| lean serial, input -> u8 codes per call | 21.51 (1.06) | 2.28 | 1.35 | 12 |
| lean serial, Dataset (converted once) | 21.07 (1.03) | 2.33 | 1.38 | 0 |
| lean 4 thr, input copy | 7.15 (1.35) | 6.86 | 4.07 | 58 |
| lean 4 thr, input -> codes per call | 8.98 (1.13) | 5.46 | 3.24 | 20 |
| lean 4 thr, Dataset | 7.35 (1.27) | 6.67 | 3.96 | 8 |
| lean serial, alpha -> bytearray (a) | 22.58 (1.09) | 2.17 | 1.29 | 244 |
| lean serial, alpha -> np.empty buffer (b) | 22.50 (1.04) | 2.18 | 1.29 | 130 |
| lean serial, borrow input + alpha -> np.empty (b) | 21.74 (1.03) | 2.26 | 1.34 | 25 |
| lean serial, alpha Vec<f64> -> bytes copy (c) | 34.80 (1.03) | 1.41 | 0.84 | 3931 |
| lean 4 thr, alpha -> bytearray (a) | 7.92 (1.40) | 6.20 | 3.67 | 250 |
| lean 4 thr, alpha -> np.empty, main-thread writes (b) | 8.16 (1.25) | 6.01 | 3.57 | 342 |
| lean 4 thr, alpha -> bytes copy (c) | 18.25 (1.01) | 2.69 | 1.59 | 3943 |
| lean lanes plain [f64;4] serial | 12.34 (1.71) | 3.98 | 2.36 | 0 |
| lean lanes plain [f64;8] serial | 14.38 (1.08) | 3.41 | 2.02 | 160 |
| lean lanes plain L4, 4 thr | 5.71 (1.51) | 8.59 | 5.09 | 26 |
| lean lanes plain L8, 4 thr | 6.23 (1.56) | 7.88 | 4.67 | 143 |
| lean +simd fearless L4 serial | 6.19 (1.39) | 7.92 | 4.70 | 0 |
| lean +simd fearless L8 serial | 6.08 (1.09) | 8.07 | 4.78 | 160 |
| lean +simd fearless L4, 4 thr | 4.73 (1.65) | 10.38 | 6.15 | 15 |
| lean +simd fearless L8, 4 thr | 4.35 (1.44) | 11.27 | 6.68 | 148 |

#### as_all: predict (one-step state predictions, 16 bytes/attempt output)

| variant | ns/attempt | x C++ serial | x C++ par | minor faults / call |
|---|---:|---:|---:|---:|
| C++ E_step.predict serial | 41.83 (1.09) | 1.00 | 0.59 | 195 |
| C++ E_step.predict parallel | 24.64 (1.01) | 1.70 | 1.00 | 195 |
| bkt_rs predict serial (numpy out) | 10.20 (1.05) | 4.10 | 2.42 | 81 |
| bkt_rs predict 4 thr | 4.51 (1.83) | 9.27 | 5.46 | 81 |
| lean predict serial -> bytearray (a) | 11.02 (1.01) | 3.80 | 2.24 | 244 |
| lean predict serial -> np.empty (b) | 10.52 (1.05) | 3.97 | 2.34 | 73 |
| lean predict serial -> bytes copy (c) | 22.29 (1.02) | 1.88 | 1.11 | 3931 |
| lean predict 4 thr -> bytearray (a) | 7.82 (1.29) | 5.35 | 3.15 | 251 |
| lean predict 4 thr -> np.empty (b) | 5.42 (1.17) | 7.72 | 4.55 | 482 |
| lean predict 4 thr -> bytes copy (c) | 15.03 (1.21) | 2.78 | 1.64 | 3941 |

#### as_all: full EM fit, 20 iterations (tol=-1)

| variant | seconds per fit | ns/attempt/iteration | x C++ serial fit | x C++ parallel fit |
|---|---:|---:|---:|---:|
| pyBKT EM_fit + C++ E-step, parallel=False | 0.45 (1.10) | 50.04 | 1.00 | 0.65 |
| pyBKT EM_fit + C++ E-step, parallel=True | 0.29 (1.07) | 32.47 | 1.54 | 1.00 |
| pyBKT EM_fit + bkt_rs shim, serial | 0.20 (1.05) | 22.81 | 2.19 | 1.42 |
| pyBKT EM_fit + bkt_rs shim, 4 thr | 0.07 (1.06) | 7.88 | 6.35 | 4.12 |
| bkt_lean.fit serial (exact) | 0.20 (1.10) | 21.74 | 2.30 | 1.49 |
| bkt_lean.fit 4 thr (chunked) | 0.06 (1.18) | 6.86 | 7.30 | 4.73 |
| bkt_lean.fit 4 thr, plain lanes L4 | 0.04 (1.65) | 4.32 | 11.59 | 7.52 |
| bkt_lean.fit +simd fearless L8, serial | 0.06 (1.08) | 6.31 | 7.93 | 5.14 |
| bkt_lean.fit +simd fearless L8, 4 thr | 0.02 (1.59) | 2.45 | 20.46 | 13.28 |

#### synth5m (5,000,000 attempts): E-step

ns/attempt = min over 3 rounds x 9 runs (max/min of per-round mins). Minor page faults per call (`ru_minflt`).

| variant | ns/attempt | x C++ serial | x C++ par | minor faults / call |
|---|---:|---:|---:|---:|
| C++ E_step.run parallel=0 | 54.67 (1.04) | 1.00 | 0.26 | 19532 |
| C++ E_step.run parallel=1 (4 OpenMP) | 14.06 (1.01) | 3.89 | 1.00 | 19532 |
| bkt_rs exact serial, alpha | 21.73 (1.08) | 2.52 | 0.65 | 165 |
| bkt_rs exact serial, no alpha | 20.02 (1.08) | 2.73 | 0.70 | 0 |
| bkt_rs 4 thr, alpha | 6.11 (1.52) | 8.95 | 2.30 | 573 |
| bkt_rs 4 thr, no alpha | 5.03 (1.46) | 10.86 | 2.79 | 1 |
| bkt_rs fearless L4 serial | 6.13 (1.02) | 8.91 | 2.29 | 0 |
| bkt_rs fearless L8 4 thr | 1.48 (1.20) | 36.98 | 9.51 | 1 |
| lean serial, input copy (to_vec) | 20.33 (1.02) | 2.69 | 0.69 | 647 |
| lean serial, input borrow (&[ReadOnlyCell], 0 copy) | 19.92 (1.04) | 2.75 | 0.71 | 945 |
| lean serial, input -> u8 codes per call | 20.01 (1.02) | 2.73 | 0.70 | 241 |
| lean serial, Dataset (converted once) | 19.71 (1.01) | 2.77 | 0.71 | 0 |
| lean 4 thr, input copy | 5.63 (1.11) | 9.72 | 2.50 | 648 |
| lean 4 thr, input -> codes per call | 5.85 (1.03) | 9.35 | 2.41 | 241 |
| lean 4 thr, Dataset | 5.04 (1.07) | 10.85 | 2.79 | 1 |
| lean serial, alpha -> bytearray (a) | 29.32 (1.02) | 1.86 | 0.48 | 20179 |
| lean serial, alpha -> np.empty buffer (b) | 23.31 (1.02) | 2.35 | 0.60 | 761 |
| lean serial, borrow input + alpha -> np.empty (b) | 22.87 (1.13) | 2.39 | 0.61 | 1059 |
| lean serial, alpha Vec<f64> -> bytes copy (c) | 37.45 (1.05) | 1.46 | 0.38 | 39711 |
| lean 4 thr, alpha -> bytearray (a) | 14.44 (1.15) | 3.79 | 0.97 | 20180 |
| lean 4 thr, alpha -> np.empty, main-thread writes (b) | 7.94 (1.01) | 6.88 | 1.77 | 1418 |
| lean 4 thr, alpha -> bytes copy (c) | 19.06 (1.08) | 2.87 | 0.74 | 39712 |
| lean lanes plain [f64;4] serial | 11.89 (1.07) | 4.60 | 1.18 | 0 |
| lean lanes plain [f64;8] serial | 11.53 (1.09) | 4.74 | 1.22 | 0 |
| lean lanes plain L4, 4 thr | 3.40 (1.07) | 16.08 | 4.14 | 1 |
| lean lanes plain L8, 4 thr | 3.04 (1.13) | 17.98 | 4.62 | 1 |
| lean +simd fearless L4 serial | 5.94 (1.12) | 9.20 | 2.37 | 0 |
| lean +simd fearless L8 serial | 4.96 (1.02) | 11.03 | 2.84 | 0 |
| lean +simd fearless L4, 4 thr | 1.66 (1.07) | 32.94 | 8.47 | 2 |
| lean +simd fearless L8, 4 thr | 1.39 (1.10) | 39.39 | 10.13 | 1 |

#### synth5m: predict (one-step state predictions, 16 bytes/attempt output)

| variant | ns/attempt | x C++ serial | x C++ par | minor faults / call |
|---|---:|---:|---:|---:|
| C++ E_step.predict serial | 47.25 (1.08) | 1.00 | 0.28 | 19532 |
| C++ E_step.predict parallel | 13.22 (1.11) | 3.57 | 1.00 | 19532 |
| bkt_rs predict serial (numpy out) | 9.21 (1.07) | 5.13 | 1.44 | 165 |
| bkt_rs predict 4 thr | 3.03 (1.14) | 15.57 | 4.36 | 572 |
| lean predict serial -> bytearray (a) | 16.85 (1.06) | 2.80 | 0.78 | 20179 |
| lean predict serial -> np.empty (b) | 11.17 (1.09) | 4.23 | 1.18 | 761 |
| lean predict serial -> bytes copy (c) | 26.38 (1.03) | 1.79 | 0.50 | 39711 |
| lean predict 4 thr -> bytearray (a) | 10.35 (1.09) | 4.57 | 1.28 | 20180 |
| lean predict 4 thr -> np.empty (b) | 4.55 (1.20) | 10.40 | 2.91 | 2509 |
| lean predict 4 thr -> bytes copy (c) | 15.80 (1.03) | 2.99 | 0.84 | 39712 |

#### synth5m: full EM fit, 20 iterations (tol=-1)

| variant | seconds per fit | ns/attempt/iteration | x C++ serial fit | x C++ parallel fit |
|---|---:|---:|---:|---:|
| pyBKT EM_fit + C++ E-step, parallel=False | 5.82 (1.03) | 58.17 | 1.00 | 0.29 |
| pyBKT EM_fit + C++ E-step, parallel=True | 1.69 (1.49) | 16.94 | 3.43 | 1.00 |
| pyBKT EM_fit + bkt_rs shim, serial | 2.31 (1.02) | 23.14 | 2.51 | 0.73 |
| pyBKT EM_fit + bkt_rs shim, 4 thr | 0.65 (1.29) | 6.47 | 8.99 | 2.62 |
| bkt_lean.fit serial (exact) | 1.99 (1.03) | 19.95 | 2.92 | 0.85 |
| bkt_lean.fit 4 thr (chunked) | 0.52 (1.03) | 5.24 | 11.11 | 3.24 |
| bkt_lean.fit 4 thr, plain lanes L4 | 0.34 (1.06) | 3.39 | 17.17 | 5.00 |
| bkt_lean.fit +simd fearless L8, serial | 0.52 (1.02) | 5.23 | 11.11 | 3.24 |
| bkt_lean.fit +simd fearless L8, 4 thr | 0.15 (1.16) | 1.51 | 38.57 | 11.23 |

#### Per-call overhead on a tiny input (1 student, 5 attempts; 3000 calls per process)

| variant | min µs | median µs | p90 µs |
|---|---:|---:|---:|
| C++ E_step.run serial | 3.3 | 3.6 | 4.3 |
| C++ E_step.run parallel (OpenMP) | 5.2 | 7.9 | 9.2 |
| C++ predict parallel | 4.6 | 5.6 | 7.3 |
| bkt_rs serial, no alpha | 1.9 | 2.1 | 3.6 |
| bkt_rs threads=4 (1 chunk -> inline) | 5.0 | 8.2 | 11.3 |
| bkt_rs serial + alpha (numpy array) | 2.0 | 2.1 | 3.8 |
| lean serial, copy input | 3.4 | 3.6 | 4.1 |
| lean serial, borrow input | 3.4 | 3.6 | 4.0 |
| lean threads=4 (1 chunk -> inline) | 3.4 | 3.7 | 4.4 |
| lean serial + alpha bytearray + np.frombuffer | 3.9 | 4.5 | 5.5 |
| lean serial + alpha into np.empty | 3.7 | 4.2 | 6.8 |
| lean predict -> bytearray + np.frombuffer | 2.1 | 2.3 | 2.8 |
| lean 4 thr, 4 chunks (forces 4 thread spawns; 20 attempts) | 46.9 | 109.8 | 383.0 |

1-min load average during the sweep: min 0.85, median 1.90, max 2.90

### Findings

1. **The lean crate loses nothing on the exact path.** Serial with a Dataset runs at 21.1 / 19.7 ns/attempt (as_all / synth5m), vs bkt_rs at 21.5 / 20.0. On 4 threads it is 7.4 / 5.0 vs 6.6 / 5.0.
   * One codegen trap appeared on the way. A `&mut dyn Emit` call inside the per-sequence loop, even when it was never taken, forced the accumulators out of registers and cost 3 ns/attempt. Fix: a `const EMIT: bool` instantiation, so the no-output kernel contains no call.
2. **Input marshalling cost** (ns/attempt, synth5m, int32 data, R = 1, relative to Dataset):

   | mode | serial | 4 threads |
   |---|---:|---:|
   | copy (`to_vec`) | +0.6 | +0.6 |
   | per-call u8 conversion (`codes`) | +0.3 | +0.8 |
   | borrow (zero copy) | +0.2 | serial only |

   * With threads, the per-call copy is an Amdahl term, because it happens on the calling thread before the parallel work: 5.6 vs 5.0 ns.
   * On as_all, copy and borrow are within noise of Dataset serially. With 4 threads, copy costs about the same as Dataset, but per-call codes conversion costs +1.6 ns (9.0 vs 7.4).
   * `fit()` converts once, so this cost becomes negligible over 20 iterations.
3. **Output marshalling is dominated by page faults** (synth5m, 80 MB output):

   | mode | serial ns/attempt | 4 threads ns/attempt | minor faults per call |
   |---|---:|---:|---:|
   | (b) caller `np.empty`, numpy uses huge pages | 23.3 (+3.6 over no-alpha) | 7.9 | about 760 |
   | (a) bytearray: pymalloc → `mmap`, no huge pages, zeroed by pyo3 before use | 29.3 | 14.4 | 20,180 |
   | (c) Vec + copy into bytes | 37.5 | 19.1 | 39,700 |

   * Option (a)'s zeroing pass is serial and hits every 4 KiB page before the workers start, so the threads cannot hide it.
   * C++ `new double[]` takes the same faults (19,532 per call).
   * On as_all (7 MB) the gap is small: (a) 22.6 vs (b) 22.5 serial.
   * **Recommendation: (b), with the caller passing `np.empty`.** It is the only safe way to get huge-page memory without the numpy crate. Its threaded cost is the serial copy into `Cell<f64>` on the calling thread: 7.9 ns/attempt vs 6.1 for bkt_rs, whose numpy-crate slice lets workers write in parallel. That gap is the price of avoiding the numpy crate.
4. **Is fearless_simd worth it?** Plain arrays in the default build stay at SSE2 (no multiversioning without `unsafe` or a dependency); fearless_simd dispatches to AVX-512 here:

   | ns/attempt | plain arrays | fearless_simd | speedup |
   |---|---:|---:|---:|
   | as_all, L=8, serial | 14.4 | 6.1 | 2.4x |
   | as_all, best 4-thread (plain L=4, fearless L=8) | 5.7 | 4.4 | 1.3x |
   | synth5m, L=8, serial | 11.5 | 5.0 | 2.3x |
   | synth5m, L=8, 4 threads | 3.0 | 1.4 | 2.2x |
   | 20-iteration fit, synth5m, 4 threads (plain L=4, fearless L=8) | 0.34 s | 0.15 s | 2.3x |

   * Plain lanes on 4 threads are already 4-5x faster than C++ parallel.
   * The question is whether 103 audited `unsafe` lines, behind token-checked safe APIs, plus MSRV 1.89 instead of 1.83 and a 3x larger `.so`, are worth about 2x. Shipping it as an optional feature or a separate wheel keeps the default build lean.
5. **Rust-side EM (`fit`)**, 20 iterations, vs pyBKT `EM_fit` with the C++ E-step:

   | | seconds | vs C++ serial fit | vs C++ parallel fit |
   |---|---:|---:|---:|
   | as_all, exact serial | 0.20 | 2.3x | 1.5x |
   | as_all, 4 threads | 0.06 | 7.3x | 4.7x |
   | as_all, fearless L8, 4 threads | 0.022 | 20x | 13x |
   | synth5m, exact serial | 1.99 | 2.9x | 0.85x |
   | synth5m, 4 threads | 0.52 | 11x | 3.2x |
   | synth5m, plain lanes, 4 threads | 0.34 | 17x | 5.0x |
   | synth5m, fearless L8, 4 threads | 0.15 | **39x** | 11x |

   Driving pyBKT's Python `EM_fit` with the bkt_rs shim costs 0.65 s on synth5m with 4 threads, vs 0.52 s for `bkt_lean.fit`. The roughly 25% difference is per-iteration marshalling plus Python M-step overhead.
6. **Per-call overhead on 5 attempts** (median µs per call):

   | | median µs |
   |---|---:|
   | C++ serial | 3.6 |
   | C++ OpenMP | 7.9 |
   | bkt_rs serial | 2.1 |
   | bkt_rs `threads=4` (rayon `install` on a cached pool) | 8.2 |
   | bkt_lean serial (copy or borrow) | 3.6 |
   | bkt_lean `threads=4` | 3.7 (a single chunk runs inline; no threads spawned) |
   | bkt_lean forced to spawn 4 scoped threads | 110 (min 47, p90 383) |

   * The 4-7 ms C++ OpenMP overhead mentioned in the task was **not** reproduced in this sweep: 7.9 µs median, 9.2 µs p90, at load ≤ 2.9. It is presumably triggered by heavier contention, such as OpenMP spin-wait threads competing with other processes.
   * Scoped threads cost about 50-400 µs per call when they actually spawn. The fixed chunk plan avoids that for inputs under one chunk (65,536 attempts). A persistent std-only worker pool would remove it for mid-size inputs too; not done.
   * bkt_lean serial spends about 1.5 µs more than bkt_rs on argument handling: buffer format probing (i8 → i32 → i64) and list conversions.

## Explicit SIMD or autovectorization? (planning round 10, `autovec.sh`, `results/autovec.csv`)

Question: does the SIMD-across-students kernel need explicit SIMD (`fearless_simd`), or does LLVM's
autovectorizer get the same speed from plain arrays? Five builds of the same `lanes_kernel`, all safe Rust:

* `plain`: `[f64; L]` arrays with `[bool; L]` masks and `if` selects, compiled for the baseline (SSE2).
* `blend`: new for this test. `[f64; L]` arrays with u64 all-ones/zero masks and bitwise blends, the shape
  vector hardware uses, also baseline. It tests whether the masks were what stopped the vectorizer.
* `plain_dispatch`, `blend_dispatch`: the same code compiled inside `fearless_simd::dispatch!`, so LLVM may
  use AVX2 or AVX-512 in a portable wheel (feature `simd`).
* `fearless`: explicit `f64x2/x4/x8` vectors with runtime dispatch (feature `simd`).
* `plain`, `blend` from a `-C target-cpu=native` build: not portable; the autovectorizer's ceiling here.

Host for this run: a different VM from Part 1, an Intel Xeon @ 2.80 GHz, family 6 model 85 (Cascade
Lake). It has AVX-512F/BW/DQ/VL, but `fearless_simd`'s dispatch picks **AVX2** here, so the dispatched
rows compare code generation at the same ISA. Load average under 1. Every variant matches the exact
serial kernel to 6e-12.

Median ns per answer (L is the better of 4 and 8 for each row; full grid in the CSV):

| Kernel | as_all, 1 thread | synth5m, 1 thread | as_all, 4 threads | synth5m, 4 threads |
| --- | --- | --- | --- | --- |
| exact scalar, one student at a time | 24.7 | 23.1 | 12.0 | 7.0 |
| `plain`, portable (SSE2) | 17.3 | 15.7 | 9.4 | 4.8 |
| `blend`, portable (SSE2) | 16.9 | 16.0 | 7.5 | 5.3 |
| `plain_dispatch` / `blend_dispatch` (AVX2) | 17.4 / 17.7 | 16.0 / 14.6 | 7.9 / 8.1 | 5.8 / 5.1 |
| `plain` / `blend`, `target-cpu=native` (not portable) | 12.2 / 13.4 | 13.0 / 12.1 | 7.4 / 6.6 | 4.2 / 4.5 |
| `fearless`, explicit SIMD (AVX2) | **8.8** | **7.7** | **4.9** | **3.2** |

Dispatch check (`results/autovec_isa_cap.txt`, synth5m, L = 8, 1 thread): capping the dispatch at SSE2
instead of AVX2 moves `blend_dispatch` from 14.8 to 17.8 ns and `fearless` from 7.6 to 12.2 ns, so the
dispatch does reach the kernel. `plain_dispatch` gains nothing. Explicit SIMD capped at SSE2 (12.2 ns) is
still faster than the best autovectorized code at AVX2 (14.8 ns).

Findings:
* **Restructuring is the first win, and the autovectorizer delivers some of it.** Laying 4–8 students side
  by side takes the scalar kernel from 23–25 ns to 16–17 ns portable (1.4–1.5x) with plain arrays and no
  dependency, or 12–13 ns with a non-portable native build.
* **Explicit SIMD adds 1.4–2x per core on top**, on both CPUs measured: here 7.7–8.8 ns against 12–17 ns;
  in Part 1 (Sapphire Rapids, AVX-512) 5.2–6.3 ns against 8.9–13 ns. A wider ISA alone doesn't explain it,
  since explicit SIMD at SSE2 beats autovectorized code at AVX2.
* **The masks weren't the obstacle.** Full-width blends (`blend`) are within noise of `plain`. What else
  keeps LLVM from vectorizing the lane loop fully (per-lane loads of each student's next answer, the
  divides, the exponent bit manipulation) wasn't isolated; the next step would be reading the generated
  assembly.
* **With 4 threads the gap narrows** to about 1.3–1.6x (`fearless` 3.2–4.9 ns against 4.2–7.5 ns for the
  best autovectorized builds), though 4-thread numbers on this shared VM are less stable.
* **Cost of explicit SIMD:** one dependency (`fearless_simd`, 103 `unsafe` lines by the Part 2 audit),
  minimum Rust 1.89 instead of 1.83, and code duplicated per ISA (this test's `lean_simd` `.so` is 20 MB
  with the extra dispatched variants; the shipped Part 2 build was 9.6 MB, against 3.0 MB without SIMD).

**Answer:** autovectorization gets part of the way, not the same result. Start with the restructured
lanes and plain arrays, which need no dependency and are portable. Offer explicit SIMD as an optional
feature or a separate wheel for the extra 1.4–2x per core, as Part 2 already suggested. The `blend`
variant didn't earn its place and needn't ship.

## Replicate Part 2

```sh
cd /tmp/claude-0/exp/rust
./build.sh          # bkt_rs (portable + native)
./rerun_lean.sh     # build_lean.sh (inst/lean, inst/lean_simd), clippy x3, audit/deps_unsafe.py,
                    # audit/rustsec_check.py, lean_check.py x2, fit_check.py, bench_lean_all.py (3 rounds), summarize_lean.py
# single pieces
PYTHONPATH=inst/default:inst/lean venv/bin/python py/lean_check.py
(cd py && PYTHONPATH=../inst/lean_simd ../venv/bin/python bench_lean_one.py synth5m lean_fit_t4_L8_fearless)
(cd bkt_lean && cargo tree -e normal,no-proc-macro && cargo tree -e normal,no-proc-macro --features simd)
```

Versions: rustc 1.97.0 (MSRV bkt_lean default 1.83, `+simd` 1.89), pyo3 0.29.3, fearless_simd 1.1.0, maturin 1.x, CPython 3.13. Default RUSTFLAGS; no `target-cpu=native`.

### Part 2 caveats

* GIL: `bkt_lean` holds the GIL for the whole call, like the C++ module. Copy, codes and Dataset inputs could release it, but `py.detach` needs a `Send` closure, and the borrow and numpy-output modes cannot be `Send` by construction.
* abi3 at 3.11+ is required for the buffer protocol. Supporting 3.9 and 3.10 would mean non-abi3 per-version wheels; both versions are or will soon be EOL.
* The plain-lanes numbers depend on the build's baseline ISA. An x86-64-v3 wheel would close much of the gap with fearless_simd, but it would not run on older CPUs.
* The machine is shared. Spreads are in parentheses; as_all multithreaded cells (1-3 ms per call) are the noisiest.
