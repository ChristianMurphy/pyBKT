# Performance roadmap

Goal: make pyBKT fast and lean on very large data without changing the numbers
the test suite pins, then add ways to fit data larger than memory and to update
predictions as new attempts arrive.

Measure with `benchmarks/bench.py`, run from outside the repository root so
the installed build is used. All numbers below are on 4 cores with Python 3.13
and NumPy 2.

## Targets

| Area | Target | How it is checked |
| --- | --- | --- |
| Correctness | Every test passes on both builds, on every CI leg | `pytest tests`, compiled and `PYTHONPATH=source-py` |
| Results | Unchanged unless a pull request is about changing them | `em_parity` (rtol 1e-9), `model_outputs` (rtol 1e-7) |
| Robustness | No crash at any sequence length or template count | `tests/test_long_sequences.py` |
| Conversion | At least 2× faster; multipair at least 50× | `bench.py --model …` |
| Predict | At least 3× faster; no templates × rows matrix | `bench.py`, `--model multigs` |
| Memory | Peak bytes per row reported for every change | `mb_*` fields in `bench.py` output |
| Scale | Fit data larger than RAM with bounded memory | phase 5 |

## Findings

Measured on the code before phase 1.

- **Crash on long histories.** The compiled E-step kept three arrays per student
  on the stack (48 bytes per attempt). One student with 200,000 attempts
  segfaulted the process. OpenMP worker threads have smaller stacks, so
  parallel fits crash at shorter lengths. The per-thread expected counts were
  also on the stack, so a very large number of templates would overflow it too.
- **`parallel=False` was sticky.** It called `omp_set_num_threads(1)` for the
  whole process, so every later fit ran on one thread: 84 ms per E-step became
  310 ms for the rest of the session.
- **Leaks and undefined behaviour.** Array buffers were freed with `delete` on
  a `void*` rather than `delete[]`. `predict_onestep_states` leaked a copy of
  the forward messages (16 bytes per attempt) on every call.
- **Conversion rescanned the whole frame per skill.** 5M rows and 100 skills
  took 32 s, almost all of it in `df[skill] == name`.
- **Multipair conversion sliced the frame row by row.** That is about 170 µs
  per row: 17 s for 100k rows, roughly 15 minutes for 5M.
- **Predict built a templates × rows matrix** with an outer product, then
  picked one entry per column. With 50 templates and 1M rows that is 400 MB
  and 3.3 s. Predict also ran the full E-step, backward pass included, and then
  copied the forward messages element by element.
- **The EM kernel itself is not the bottleneck.** It runs at about 17 ns per
  attempt per iteration on 4 threads. The Python and pandas layers around it
  are what dominate.
- **Memory is dominated by the input DataFrame.** The kernel needs about
  28 bytes per attempt during EM (int32 answers, int64 resources, two doubles
  of forward messages). A DataFrame with string IDs costs 10 to 20 times that.

### Bugs found, left for their own pull requests

Each of these changes results, so each needs its own pull request that updates
the xfail markers or reference values and says why.

- The compiled E-step returns the log-likelihood with `PyLong_FromLong`, which
  cuts off the fractional part. So the `tol` stop compares whole numbers, and
  choosing the best of `num_fits` breaks ties toward the first fit. Fixing it
  changes the compiled `model_outputs` references.
- #70: fixed and per-template parameters are read once and reused for every
  template, because `learn`, `guess` and so on are not reset inside the loop.
- #72: the pure-Python E-step skips the evidence from one-attempt students.
- #65: the pure-Python fit fails on NumPy 2.
- Under pandas 3, multipair keys come from the repr of a pandas
  `StringArray`, not a NumPy array. A model fitted under pandas 2 then reports
  every pair as "not fitted" when it predicts under pandas 3.
- Compiled parallel fits are not reproducible to the last bit, because threads
  add their counts in whatever order they reach the critical section.
- `bigT` and the row indices are 32-bit `int`, so more than 2^31 attempts in
  one skill would overflow. Phase 1 widens `bigT` in the E-step only.
- Rows with a missing `user_id` are dropped from `lengths` by `groupby` but kept
  in `data`, which misaligns `starts`.

## Phases

### 1. Crash fixes, predict, conversion (this branch)

Nothing changes numerically. The E-step, the expected counts, the forward
messages and both prediction outputs are bit-identical to the previous build,
and conversion output is identical across 46 input shapes.

- E-step scratch space moved to the heap, reused per thread.
- Thread count set per call, not for the whole process. `delete[]` fixed, and
  the leak in `predict_onestep_states` fixed.
- `E_step.predict`: the forward pass alone, giving one-step state predictions
  directly. `E_step.run` keeps its signature and output.
- Correct-answer predictions take the guess and slip of each attempt's template
  directly, instead of building a templates × rows matrix. This applies to both
  builds.
- `convert_data`: one `groupby` instead of one scan per skill. The multipair,
  multiprior, multigs and multilearn loops are vectorized, and each distinct
  value is looked up once.

| Benchmark | Before | After |
| --- | --- | --- |
| convert, 1M rows | 0.49 s | 0.15 s |
| convert, 20M rows | 12.2 s | 4.0 s |
| convert, 5M rows, 100 skills | 31.6 s | 2.0 s |
| convert, multipair, 100k rows | 17.2 s | 0.06 s |
| convert, multigs, 1M rows | 3.0 s | 0.48 s |
| predict, 1M rows | 60 ms | 14 ms |
| predict, 20M rows | 1.22 s | 0.43 s |
| predict, multigs × 50 templates, 1M rows | 3.27 s | 0.12 s |
| peak memory, multigs × 50 templates | 1.63 GB | 0.81 GB |
| one student, 400k attempts | segfault | works |

### 2. Memory in RAM

- Let the C++ code read narrow integer types as they are (int8 answers, int32
  resources) instead of converting to int32 and int64 copies.
- Skip writing the `alpha` output on EM iterations that do not use it. It is
  16 bytes per attempt, allocated and filled on every iteration.
- Build arrays from categorical codes rather than string columns, and avoid
  copying and sorting the caller's frame in place where a sort index is enough.
- Accept data already converted to arrays (or Arrow) so callers can skip pandas.

Risk: low. Results are unchanged. The only API change is new, optional input
types.

### 3. E-step CPU

- Give each thread fixed blocks of students and add the per-block counts in a
  fixed order. That keeps the speed and makes parallel results reproducible.
- Schedule work by attempts, not by number of students, so a few long histories
  do not leave threads idle.
- Fit all `num_fits` restarts in one pass over the data, since the pass is
  limited by memory bandwidth. Fit skills in parallel.
- Merge identical answer sequences and weight them by how often they occur.
  The result is exact up to the order of summation. It is a large win for short
  sequences with one template, where many students share a pattern.

Risk: medium. Summation order changes the last bits, which is well inside
rtol 1e-9, but the reference files must not be regenerated just to pass.

### 4. Vectorized pure Python

Step all students forward and backward at once with NumPy, grouped by sequence
length, instead of looping per student and per attempt in Python. The goal is
at least 20× on the pure-Python build. Parity with the compiled build is held
at the existing 1e-9 tolerance. This depends on the #65 and #72 fixes landing
first, so the pure-Python tests are not xfail when it is checked.

Risk: medium to high, for parity. Mitigate with the bit-level comparison
scripts used in phase 1.

### 5. Larger than RAM: exact chunked EM

The expected counts add up across students, so EM can stream students in
chunks and match the in-memory fit, up to summation order.

- Pass 0 reads the data once to fix skills, template and resource names, and
  each student's position. It writes the compact arrays (about 5 to 12 bytes
  per attempt) to `.npy` files.
- Each EM iteration memory-maps those arrays. The kernel reads them without
  copying, and the operating system pages them in. Memory stays bounded by the
  page cache, not by the data size.
- Also offer an iterator API for sources that cannot be stored locally.

Risk: medium. It is a new API. Each EM iteration is a full pass, so speed is
limited by I/O, which the compact arrays keep small.

### 6. Streaming updates

Keep each student's forward state (P(known) after their latest attempt) for
each skill. A new attempt then updates its prediction in O(1), with the
parameters held fixed. The result must equal `predict` over the full history,
and the tests check that equality.

Risk: low. The algebra is the same forward recursion.

### 7. Approximate online EM (opt-in)

Stepwise or online EM (Cappé and Moulines 2009; Liang and Klein 2009) with a
decaying step size, for data that never stops arriving. It is never the
default. It is judged against batch EM on parameter recovery and AUC, not
against the reference values.

Risk: results differ by design, so it stays behind its own flag and tests.

## Alternatives considered

- **Numba or Cython for the pure-Python build.** Fast, but adds a heavy
  dependency or a second compiled path. Vectorized NumPy keeps the pure-Python
  build dependency-free.
- **Rust (pyo3) or a rewrite of the C++.** The C++ kernel is already near
  memory bandwidth, and the cost is in the Python layers.
- **GPU (JAX or CuPy).** The two-state model does little arithmetic per byte.
  It only pays off when many restarts or skills are batched together, and it
  adds a large dependency.
- **Polars for conversion.** Faster than pandas, but a new required
  dependency. Phase 2 accepts Arrow input instead, which Polars can produce.
