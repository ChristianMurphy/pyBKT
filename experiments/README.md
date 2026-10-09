# pyBKT experiments: end-to-end performance, online EM, Rust

These are experiments, not a deliverable. Nothing in this directory is imported by
pyBKT, and nothing here changes library behaviour. The library work is on branch
`claude/pybkt-performance-research-xqv9jv`: phase 1 crash and leak fixes, a
forward-only predict, a vectorized `convert_data`, and `benchmarks/ROADMAP.md`.
This branch, `claude/pybkt-experiments-xqv9jv`, is that branch plus this
`experiments/` directory.

Machine for every number below: 4 vCPU (Intel Xeon @ 2.10 GHz), 15 GB RAM,
Linux, Python 3.13, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1,
pyarrow 25.0.1, polars 2.0.0, duckdb 1.5.6. The C++ build is phase 1.

Two caveats apply throughout:

- Background jobs (Rust builds, research agents) ran during some timings, and
  load average reached 2–3 on 4 cores. Where that matters, it is noted.
- The online EM and SQUAREM experiments use the NumPy reference in `bkt_np.py`:
  one skill at a time, one template, forget fixed at 0 (pyBKT's default).

## Reproduce from scratch

```bash
# 1. environment
python3.13 -m venv venv && . venv/bin/activate
pip install numpy setuptools pandas requests scikit-learn pyarrow polars duckdb pytest
git checkout claude/pybkt-experiments-xqv9jv
pip install --no-build-isolation .          # pyBKT with the C++ extension (phase 1 build)
python -c "import pyBKT.fit.E_step"         # must succeed; setup.py silently falls back to pure Python
# The pure-Python E-step timing needs NumPy < 2, which needs Python <= 3.12:
#   uv venv -p python3.12 v12np1 && uv pip install -p v12np1 "numpy<2" pandas requests scikit-learn

# 2. data (ASSISTments 2009 sample and Cognitive Tutor sample from the pyBKT examples)
cd experiments && ./fetch_data.sh           # writes ../data/as.csv, ../data/ct.csv  (paths: see DATA below)

# 3. fixtures used by the kernel benchmarks (need the C++ build installed)
python make_fixture.py                      # as_all.npz, synth5m.npz (+ C++ reference outputs)

# 4. experiments (each prints its table; raw rows go to results/*.csv)
python check_smoothing.py                   # sec 1
python online_em.py                         # sec 2, self-checks tying code to equations
python cappe_moulines.py 40000              # sec 2, synthetic, known truth
python cm_real.py 12                        # sec 2, ASSISTments, held-out
python compare_em.py 3                      # sec 2, earlier ad-hoc variants (cold start)
python compare_online.py synth 40000        # sec 2, event-level streaming
python compare_online.py warm 12            # sec 2, batch-then-stream on real data
python synth_discount.py                    # sec 2, discounting
python squarem.py 15                        # sec 3
python np_estep.py as_all.npz && python np_estep.py synth5m.npz   # sec 4
python dedup.py                             # sec 6
python load_bench.py all ../data/as.csv     # sec 7
python load_bench.py make-synth 20000000 && python load_bench.py all ../data/synth.csv
python e2e_pybkt.py ../data/as.csv 1 0      # sec 7  (args: file, num_fits, parallel)
python omp_overhead.py                      # sec 7
python live_bench.py                        # sec 8
# Rust: see rust/REPORT.md
```

DATA: the scripts read `/tmp/claude-0/data/*.csv` and write fixtures to
`/tmp/claude-0/exp/`, the paths used when this was run. Edit the constants at
the top of each script, or symlink, to use other locations.

## 1. Exact E-step from a forward pass alone

`bkt_np.StudentStream` keeps a filter φ (2 numbers) and
ρ[x] = E[statistics | x_t = x, y_1:t] (2 × 10 numbers) for each student. This
is the forward smoothing of additive statistics used by Cappé (2011). After a
student's last answer, φ·ρ equals the forward-backward expected counts.

- maximum relative difference 2.5e-16, log-likelihood difference 8e-12 (300 random students, `check_smoothing.py`)
- equal to pyBKT's C++ `E_step.run` counts, once pyBKT's transposed layout is accounted for
- still exact when students' answers are interleaved in time (`online_em.py` check b)

What this makes possible: an exact E-step in one pass with 22 floats of state
per student-skill pair. A student's history no longer needs to sit in one chunk,
which enables exact out-of-core EM, and an event stream can be absorbed exactly.

## 2. Online EM, tied to the papers

The code is in `online_em.py`; its docstring maps each line to an equation.

| Piece | Source | Where |
| --- | --- | --- |
| BKT fits the exponential-family setting: closed-form M-step θ̄(s), statistics s̄(Y; θ) | Cappé & Moulines 2009, Assumption 1, eqs. 11–13 | `bkt_np.m_step`, `bkt_np.fb_counts` |
| s ← s + γ(s̄(Y; θ) − s), θ = θ̄(s), with one **student sequence** as one observation | C&M 2009, eq. 15 | `StudentLevelOnlineEM.update` |
| γ_n = γ₀·n^(−α), α ∈ (½, 1] | C&M 2009, Theorem 5 | `StudentLevelOnlineEM.gamma` |
| α = 1, γ₀ = 1 is Neal & Hinton's incremental EM in online mode | C&M 2009, §2.3 | checked in `online_em.py` (a) |
| No M-step until enough data (they skip the first 20 observations) | C&M 2009, Assumption 1(c), §4 | `burn_in=20` |
| Polyak–Ruppert averaging of θ | C&M 2009, eq. 29 | `average_from` |
| Keep s in a compact subset for stability | C&M 2009, §3.1 | `pseudo` counts |
| Forward smoothing of additive statistics inside one HMM sequence | Cappé 2011 (arXiv:0908.2359) | `bkt_np.StudentStream` |
| Discounting old evidence for drifting data | Mongillo & Denève 2008 | `EventLevelOnlineEM(discount=…)` |
| Combining many interleaved students: swap each student's old smoothed contribution for the new one | Our adaptation, from Neal & Hinton's incremental idea; not from a paper | `EventLevelOnlineEM.update` |

Grounding (see `online_bkt_literature.md`): the student-level algorithm is C&M
2009 applied directly, because students are independent observations. The
event-level algorithm combines Cappé 2011's within-sequence recursion with
bookkeeping we designed. The literature check found no BKT-specific paper that
does online EM of the global parameters with streaming statistics. The closest
is KT² (arXiv:2506.09393), which is incremental EM without discounting. That
check could not reach arXiv or the EDM sites, so its paper-level claims rest on
abstracts. Cappé 2011's recursion has been checked numerically, not against the
paper's text.

### Results

**Synthetic stream with known truth** (40k students, Poisson(12) answers each,
481k answers; `cappe_moulines.py`, 3 random starts). Maximum absolute error over
prior, learn, guess and slip:

| Method | Passes | Error (mean of 3 starts) |
| --- | --- | --- |
| batch EM | 20–22 | 0.0013 |
| C&M eq. 15, α = 0.75, 50 students per step, + PR averaging | 1 | 0.0016 |
| C&M eq. 15, α = 0.75, 50 students per step | 1 | 0.0017 |
| C&M eq. 15, α = 0.75, 1 student per step | 1 | 0.0019 |
| C&M eq. 15, α = 0.6, 50 students per step | 1 | 0.0020 |
| C&M eq. 15, α = 1 (Neal & Hinton / naive accumulation) | 1 | 0.0069–0.0080 |
| C&M eq. 15, α = 0.6, 1 student per step (no averaging) | 1 | 0.0117 |
| event-level smoothed, discount 0.99997 (`synth_discount.py`) | 1 | 0.0061 |
| event-level smoothed, no discount | 1 | 0.030 |
| filtering only, using the current mastery estimate as the E-step (biased) | 1 | 0.039 |

Following the paper's recipe (α ∈ (½, 1), burn-in, averaging) matters a lot.
With it, one online pass comes close to fully converged batch EM. Using only the
current mastery estimate is biased: prior ended at 0.339 against a true 0.300.

**ASSISTments, held out** (`cm_real.py 12`: 12 largest skills, 3 random starts,
80/20 split by student, cold start). The table gives the difference from batch
EM in held-out log-likelihood per answer. Batch EM averaged 38 passes.

| Method | Mean | Median | Worst | Δ AUC |
| --- | --- | --- | --- | --- |
| C&M, α = 0.75, 50 per step, 3 passes | −0.0057 | −0.0002 | −0.082 | −0.018 |
| C&M, α = 0.75, 10 per step, 1 pass | −0.0076 | −0.0006 | −0.082 | −0.019 |
| C&M, α = 0.75, 50 per step, 1 pass | −0.0079 | −0.0012 | −0.081 | −0.020 |

The worst case is a skill where batch EM finds a different optimum (see the
pathologies below). The earlier ad-hoc stepwise variant in `compare_em.py`,
without burn-in and with poorly chosen steps, was about 0.08 AUC behind; this
recipe is 0.02 behind.

**Batch fit, then stream** (`compare_online.py warm 12`). Batch-fit on the
first half of each skill's training answers by time, then process the second
half online. The reference is a batch refit on all the data. A per-skill
breakdown is in `results/compare_online_warm.csv`.

- An event-level update over the new half tracks one full EM iteration over all
  the data: the median held-out log-likelihood is the same.
- A periodic refit of 10 EM iterations over all the data is −0.0018 from the
  full refit. Freezing the half model is −0.026.

**Pathologies to design for**

- **Parameters stuck at 0 or 1.** On "Area Triangle", guess reached 0, and EM's
  update keeps a parameter at 0 or 1 forever. The fix is pseudo-counts or
  projection (C&M §3.1).
- **Several optima.** On "Multiplication and Division Integers", batch EM went
  to a degenerate mode with guess 0.91. BKT is not identifiable in general
  (Beck & Chang 2007), and pyBKT's `num_fits` restarts exist for this reason.
  An online learner stays in the mode its starting point is in.

### What this means for an online design

- **Inference engine**, O(1) per answer, exact. This answers pyBKT issue #55.
  See section 8.
- **Parameter learning**, preferred: student-level C&M eq. 15. Update when a
  student's session or sequence completes; use α ≈ 0.75, mini-batches of tens
  of students, burn-in, averaging and pseudo-counts. Warm-start from a batch fit.
- **When updates are needed within a session:** the event-level smoothed
  statistics, plus a discount only if drift is expected.
- **Anchor:** periodic batch refits. Restarts are what handle the multiple optima.
- Update mastery estimates immediately and global parameters rarely. M-steps
  every 100 or every 1,000 events made no measurable difference.

## 3. EM acceleration: SQUAREM

`squarem.py` implements SqS3 (Varadhan & Roland 2008) in logit space. It needs
SQUAREM's adaptive step cap and a monotone safeguard: without them, 8 of 45 runs
extrapolated onto a 0/1 boundary and ended thousands of nats worse. With them,
on 15 skills × 3 starts at tolerance 1e-4:

- total E-steps 1,149 → 778 (1.48×); median 1.27× per run (p10 0.6, p90 2.2)
- never worse, once better, parameters match plain EM to about 2.5e-4 (median)

The gain is modest at pyBKT-like tolerances. It would be opt-in, because
`em_parity` pins 20 plain iterations.

## 4. Vectorized NumPy E-step (candidate pure-Python backend)

`np_estep.py` steps all students together, sorted by length. It agrees with
the C++ counts to within 3e-13 relative error.

| Fixture | Pure-Python pyBKT | Vectorized NumPy | C++ serial | C++, 4 threads |
| --- | --- | --- | --- | --- |
| ASSISTments merged (lengths up to 6,154) | 31,481 ns/answer | 676 | 47.5 | 28.6 |
| synthetic, 250k students × 20 | 35,668 | 169 | 52.8 | 22.5 |

It is 47–210× faster than today's pure-Python build with no new dependency.
Long, skewed sequences hurt it, because few students are active per step.

## 5. Rust

See `rust/REPORT.md`. (Summary added when the prototype finished; see below.)

## 6. Exact work sharing through prefixes

With forward-only smoothing, students whose histories share a prefix share φ
and ρ at that prefix. An exact E-step then costs one update per trie node, not
one per answer (`dedup.py`; counted, not implemented):

- ASSISTments, 110 skills: 421,318 answers → 153,417 trie nodes, 2.75× less
  work (1.65× with the template in the key)
- synthetic 5M answers (250k × 20): 24.8× less work

## 7. End to end: load → convert → fit → predict

### Loading (`load_bench.py`, one fresh process per loader)

Every loader must produce byte-identical compact arrays (the same digest): int8
answers and int32 skill codes sorted by (skill, user, order), plus per-student
starts and lengths.

| Loader | ASSISTments 83 MB CSV, 30 cols | | Synthetic 871 MB CSV, 20M rows | |
| --- | --- | --- | --- | --- |
| | time | peak | time | peak |
| pyBKT today (read all columns, then `convert_data`) | 3.24 s | 590 MB | 60.8 s | 4,976 MB |
| pandas, `usecols` + `int8` | 1.20 s | 289 MB | 33.0 s | 2,599 MB |
| pandas, `engine="pyarrow"` | 0.80 s | 515 MB | 25.9 s | 4,493 MB |
| pyarrow.csv, dictionary-encoded strings | 0.38 s | 372 MB | 17.0 s | 3,041 MB |
| polars, eager | 0.54 s | 349 MB | 22.8 s | 2,682 MB |
| polars, lazy (filter + rank + sort in Rust) | 0.27 s | 349 MB | 12.3 s | 2,493 MB |
| duckdb SQL | 0.35 s | 319 MB | 8.2 s | 2,813 MB |
| compact `.npy`, memory-mapped (after the first conversion) | 0.001 s | 31 MB | 0.03 s | 198 MB |

Parquet input helps pandas (34.9 s against 60.8 s for the CSV) but not
Polars or DuckDB here; the time is dominated by the 20M-row sort.

Two data findings:

- **multigs data is stored dense**, templates × answers. On ASSISTments with
  every skill merged that is 816 × 449k int32, 1.47 GB, to hold 449k answers. A
  sparse layout (one template index per answer) would be about 800× smaller.
- **Phase 1 `convert_data` speed depends on whether pyarrow is installed.**
  pandas 3 then stores strings in pyarrow. Without pyarrow, phase 1 converts
  ASSISTments in 0.57 s vs 2.60 s before. With it, 0.99 s vs 1.22 s, and about
  90 MB more peak memory. Old and new code use the same memory in the same
  environment, so this is not a phase 1 regression.

### Fitting (`e2e_pybkt.py`, ASSISTments, 110 skills, 421k answers)

| Build | `parallel` | fit (1 fit per skill) | predict |
| --- | --- | --- | --- |
| before phase 1 | True (default) | 9.6 s | 4.4 s |
| before phase 1 | False | 3.2 s | 2.8 s |
| phase 1 | True (default) | 7.7 s | 1.6 s |
| phase 1 | False | **1.3 s** | 1.6 s |

With the default `num_fits=5`, fit takes 33.5 s. Profiling shows 833 E-step
calls taking 6.5 s, about 7.8 ms each. Most of that is OpenMP's per-call cost
(`omp_overhead.py`):

| Answers in call | old, parallel | phase 1, parallel | phase 1, serial | phase 1, `OMP_NUM_THREADS=2` |
| --- | --- | --- | --- | --- |
| 5 | 108 µs | 4,061 µs | 4 µs | 5 µs |
| 1,800 | 92 µs* | 7,144 µs | 83 µs | 49 µs |
| 22,900 | 1,175 µs | 4,972 µs | 1,049 µs | 612 µs |

\* The old build had already turned serial by this point, because of its sticky
`omp_set_num_threads(1)`.

These were measured at load average 2.5–3 on 4 cores. Four spinning OpenMP
threads competing for busy cores turn a barrier into milliseconds. This needs
re-measuring on an idle machine. Either way, **real BKT data has many small
skills, and parallelizing inside one small E-step costs more than it saves.**
Options: use one thread below roughly 50k answers, and parallelize across skills
(or `num_fits` restarts) instead of within an E-step.

### Phase 1 regression check

Fixing the sticky `parallel=False` does not slow down the default path; the
old build pays the same overhead whenever `parallel=True`. It does change
processes that used to call `parallel=False` once and then (by accident) stayed
serial.

## 8. Live mastery updates

`live_bench.py`: exact forward filtering per (student, skill), with parameters
fixed, against pyBKT's `predict` over the full history.

- per answer in plain Python: 2.4 µs (0.42M answers per second). NumPy
  micro-batches of 4,096 were slower at 3.9 µs, so per-answer work is too small
  for NumPy. A compiled loop is the next step (see the Rust report).
- matches pyBKT's predict to within 1.6e-9 on 294k answers, outside two issues
  found in pyBKT (below). Error builds up along long sequences with
  near-degenerate parameters (guess around 1e-6), so this is not bit-exact.
- **Numerical lesson:** the textbook update P(L | wrong) = k·slip / (1 − P(correct))
  cancels badly when k ≈ 1 and guess ≈ 0. k drifted above 1 and blew up to
  2.8. Normalize the two unnormalized state weights instead, as pyBKT's forward
  pass does.

### pyBKT bugs found along the way

1. **A skill name containing regex characters is never predicted.**
   `convert_data` matches `'^(' + '|'.join(skills) + ')$'` without escaping.
   "Order of Operations +,-,/,* () positive reals" does not match itself, so
   `predict` silently leaves its 2,339 ASSISTments rows at the 0.5 placeholder.
2. **Tied `order_id`s are ordered arbitrarily.** pyBKT sorts with an unstable
   quicksort, then a mergesort by user. In ASSISTments, 29.6% of rows are in
   (user, skill) groups with tied `order_id`s, often several rows per
   multi-skill problem. Their order, and so fit and predictions, depends on the
   sort algorithm and on input order.
3. Earlier findings, still open: the compiled log-likelihood is truncated to an
   integer, which affects early stopping and the choice among `num_fits`;
   multipair keys differ under pandas 3; #70; #72; #65. Details are in
   `benchmarks/ROADMAP.md`.

## 9. Alternative fitters in the literature

Khajah, *Supercharging BKT with Multidimensional Generalizable IRT and Skill
Discovery*, JEDM (text supplied by the owner):

- It fits BKT by **gradient descent** (NAdam) as a PyTorch RNN cell, GPU optional.
- It speeds up the recurrence with a stride C (5–7). It enumerates the 2^C state
  paths per stride, giving up to a 47% speedup over its plain RNN cell.
- Speed: hmm-scalable (C++, gradient descent) is fastest in that paper. The
  accelerated RNN gets within 10–20% of it on 3,000 students × 100–500 answers.
- Accuracy: brute-force grid search is slightly ahead on AUC, because it avoids
  local optima. That agrees with our multiple-optima finding.
- Its value is flexibility (BKT+IRT, skill discovery), not raw speed. For pyBKT,
  it is a reference point for a gradient-based fitter option, not a faster EM.

An alternative to its stride trick, not tested here: BKT's forward pass is a
chain of 2 × 2 matrix products, so it can be parallelized over time with an
associative (prefix) scan, exactly up to rounding. See Särkkä & García-Fernández
(2021) on temporal parallelization of Bayesian smoothers. That would help the
long, skewed sequences that hurt SIMD across students. It needs a log-scale or
rescaling scheme inside the scan.

The deep-research run (`deep_research_summary.md`) verified only the nanobind
binding benchmarks: a binding swap gives smaller, faster-building wheels, not
faster fitting. The proxy blocked most academic sources, so its claims about
algorithms and numerics did not reach verification.

## Papers that would firm this up

The research tools here could not reach arXiv or the EDM proceedings. Uploading
any of these would let the claims above be checked against the text:

1. Cappé (2011), *Online EM algorithm for hidden Markov models*, arXiv:0908.2359.
   Checks `StudentStream` and the event-level design. **Most useful.**
2. Mongillo & Denève (2008), *Online learning with hidden Markov models*,
   Neural Computation 20(7). The discounting.
3. Gao et al. (2025), KT², arXiv:2506.09393. The closest BKT-specific online work.
4. Varadhan & Roland (2008), SQUAREM, Scand. J. Stat. 35(2). Step-length rules.
5. Beck & Chang (2007) on identifiability, and Pardos & Heffernan (2010) on
   visualizing EM convergence for BKT. The pathologies in section 2.

## Files

`bkt_np.py` reference BKT · `online_em.py` online EM mapped to equations ·
`check_smoothing.py` · `compare_em.py` · `compare_online.py` · `synth_discount.py` ·
`cappe_moulines.py` · `cm_real.py` · `squarem.py` · `np_estep.py` · `dedup.py` ·
`make_fixture.py` · `load_bench.py` · `e2e_pybkt.py` · `omp_overhead.py` ·
`live_bench.py` · `conv_mem.py` · `fetch_data.sh` · `results/*.csv` ·
`online_bkt_literature.md` · `deep_research_summary.md` · `rust/` (source,
scripts, results, `REPORT.md`; no build output)

## References

- Cappé, O. & Moulines, E. (2009). Online expectation-maximization algorithm for latent data models. JRSS-B 71(3). arXiv:0712.4273.
- Cappé, O. (2011). Online EM algorithm for hidden Markov models. JCGS 20(3). arXiv:0908.2359.
- Mongillo, G. & Denève, S. (2008). Online learning with hidden Markov models. Neural Computation 20(7).
- Neal, R. & Hinton, G. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants.
- Varadhan, R. & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. Scand. J. Stat. 35(2).
- Khajah, M. M. Supercharging BKT with multidimensional generalizable IRT and skill discovery. JEDM.
- Gao et al. (2025). KT². arXiv:2506.09393.
- Beck, J. & Chang, K. (2007). Identifiability: a fundamental problem of student modeling. UM 2007.
- Särkkä, S. & García-Fernández, Á. (2021). Temporal parallelization of Bayesian smoothers. IEEE TAC 66(1).
- Badrinath, A., Wang, F. & Pardos, Z. (2021). pyBKT. EDM 2021. pyBKT issue #55.
