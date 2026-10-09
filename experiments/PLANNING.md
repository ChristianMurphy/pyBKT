# Turning the research into deliverable chunks: working notes

Status: draft notes, revised as decisions are made (latest: 2026-10-09, after syncing with upstream).
Evidence for every number is in `README.md` (sections in brackets) or `rust/REPORT.md`.

Contents: 1 inventory · 2 ways to slice · 3 decisions · 4 what a new user gets · 5 upstream sync ·
6 adoption lens · 7 plan v3 · 8 critique · 9 open questions · 10 coverage check.

## 1. Inventory: what the research produced

Each item is tagged by how it affects results, using the classes from README discussion:
**[same]** bit-identical or within a stated float tolerance; **[fix]** a bug fix that changes results;
**[opt-in]** a new, optional estimator or feature; **[infra]** tests, benchmarks, docs.

### Already built (branch `claude/pybkt-performance-research-xqv9jv`, not merged anywhere)
- [same] Phase 1: no stack overflow on long histories, no sticky `parallel=False`, leak and `delete[]`
  fixes, forward-only predict (3–26x), vectorized `convert_data` (3–16x, multipair 290x). Built on the
  regression harness branch. Merges cleanly onto upstream master (section 5).

### Bugs found (K = known bug; renamed from B1–B13 so they don't clash with Train B's PR numbers)
| # | Bug | Effect | Size of fix |
| --- | --- | --- | --- |
| K1 | C++ log-likelihood truncated to an integer | EM stops early (24 vs 47 iterations on merged ASSISTments); best-of-`num_fits` ties go to the first restart | tiny |
| K2 | #70: every template/resource uses the first one's parameters | multigs/multilearn fits wrong in C++ | **fixed upstream (#71)** |
| K3 | #72: pure-Python skips one-answer students | params differ by up to 0.085 vs C++ | **fixed upstream (#73)** |
| K4 | #65: pure-Python fit fails on NumPy 2 | pure-Python unusable on NumPy 2 | small–medium |
| K5 | Skill names with regex characters never matched in predict | silent 0.5 predictions (2,339 ASSISTments rows) | tiny |
| K6 | Tied `order_id` ordered by unstable quicksort | 29.6% of ASSISTments rows in ambiguous order; results depend on input order | tiny |
| K7 | multipair keys are numpy/pandas reprs | models fitted under pandas 2 fail under pandas 3 | small (key format change = compatibility question) |
| K8 | NaN user ids dropped from lengths but kept in data | misaligned sequences | tiny |
| K9 | C++ parallel reduction order nondeterministic | last-bit differences run to run | small |
| K10 | `bigT` / indices are 32-bit | overflow past 2^31 answers per skill | small |
| K11 | PyPI 1.4.3 wheel is pure Python (`py3-none-any`, no extension) | every pip user gets the ~75x slower path, with #72 | **fixed on master (#66, #69)**; needs a release |
| K12 | #58: `import pyBKT.models` fails with scikit-learn ≥ 1.8 | package unusable on current scikit-learn | **fixed on master (#64, #68)**; needs a release |
| K13 | `E_step.run` returns `alpha` as an N x 2 buffer labelled (2, N) | confusing internal layout (found by the Rust work) | small, internal |

### Performance, results unchanged
| # | Item | Measured benefit | Cost / risk |
| --- | --- | --- | --- |
| P1 | Serial below a size threshold, or parallel across skills, instead of OpenMP inside small E-steps | 6x fit speedup under load (7.7 s -> 1.3 s); per-call cost only 7.9 µs when idle | needs a controlled re-measurement first |
| P2 | Skip writing `alpha` during EM | 16 B/answer per iteration not written | tiny |
| P3 | Sparse multigs layout (one template index per answer) | 1.47 GB -> ~2 MB on merged ASSISTments | medium; touches kernel and converter |
| P4 | Narrow dtypes (int8 answers, int32 resources) | ~4x less kernel input memory | small |
| P5 | Vectorized NumPy E-step for the pure-Python backend | 54x end-to-end fit (18–103x per skill), matches C++ to 3.4e-15 | depends on K4 (K3 now fixed upstream); stable posterior-form backward pass |
| P6 | Optional DuckDB loader / one-query `convert_data` | ~4x at 20M rows, identical output | optional dependency |
| P7 | Compact store (`.npy`, memory-mapped) | reload 20M rows in 0.03 s | new file format to maintain |
| P8 | Prefix sharing in the E-step | 2.75x less work (ASSISTments), ~25x (short sessions) | counted only, not implemented |
| P9 | Rust backend (`bkt_lean`) | 2.2x serial; 10–35x with SIMD + threads; whole fit in Rust | Rust toolchain to build from source; abi3 Python ≥ 3.11; packaging matrix |

### New capabilities (opt-in, different estimators or new APIs)
| # | Item | Grounding | Evidence | Maturity |
| --- | --- | --- | --- | --- |
| N1 | Live mastery updates, O(1) per answer (issue #55) | standard BKT forward filtering | matches `predict` to 1.6e-9; 2.4 µs/answer in Python | ready |
| N2 | Beta priors / MAP-EM | Beck & Chang 2007 | +0.0023 held-out ll, stuck-at-0/1 fits 15% -> 0% | ready; needs default prior strengths |
| N3 | SQUAREM acceleration | Varadhan & Roland 2008 | 1.4–1.7x fewer EM steps, never a worse optimum in 48 runs | ready; opt-in |
| N4 | Exact out-of-core EM via forward-only smoothing | Cappé 2011 Prop. 1 | equals forward-backward to 1e-16 | design proven; not built as a feature |
| N5 | Student-level online EM | Cappé & Moulines 2009 eq. 15 | one pass within 0.0016 of truth (batch 0.0013); median held-out gap 0.0012 | experimental |
| N6 | Event-level online EM | Cappé 2011 + own adaptation | only alpha = 0.8 stable | research only |
| N7 | GPU backend (wgpu) | — | not measurable here | not recommended now |

### Infrastructure
- Regression harness (other branch), benchmark script, golden-output checks used in this research.
- Docs: change classification, citations, replication commands.

## 2. Ways to slice the work (considered; plan v3 combines 1 and 2)

1. **By result-risk class**: all [same] first, then [fix] one per PR, then [opt-in]. Easiest to review and
   to accept upstream; benefits arrive in order of safety, not of value.
2. **By user journey / persona**: (a) "my big dataset is slow" (P1, P2, P3, P6, P7, P9);
   (b) "my parameters are implausible" (K1, N2, N3); (c) "I run a live tutor" (N1, N5);
   (d) "I can't install the C++ build" (K3, K4, P5). Each track tells its own story.
3. **By pipeline layer**: load -> convert -> kernel -> EM algorithm -> API. Clean ownership, but each
   layer alone gives a partial benefit.
4. **By paper**: one chunk per technique, each with its citation and evidence (Beck & Chang priors;
   SQUAREM; Cappé 2011 smoothing; Cappé & Moulines online EM). Easy to explain and cite; ignores the
   engineering prerequisites.
5. **By backend strategy**: (a) keep and harden C++; (b) introduce Rust alongside; (c) Rust replaces C++.
   Decides a lot of the downstream work.
6. **Vertical slices with a benchmark target each**: every chunk must move one number in
   `benchmarks/bench.py` or the harness (e.g. "fit 20M rows under X s and Y GB").

## 3. Decisions so far (owner)
Round 1:
- **Destination:** upstream PRs to CAHLR/pyBKT, landing first on the fork.
- **Users:** all four: batch researchers, very large datasets, users without a compiler, live tutors.
- **Priority:** correctness first.
- **Rust:** optional backend, later, once C++ and Python are stable.

Round 2:
- **Process:** open a tracking issue upstream first, listing the trains, then send PRs.
- **Result-changing fixes:** ask the maintainer, case by case (listed in the tracking issue).
- **First train:** sync with upstream first; some issues are already logged and fixed (done in section 5).
- **Explaining the work:** a five-line summary on every PR, README sections for user-visible changes, and
  notebooks or docs pages for the opt-in features.

## 4. What a new user gets today (PyPI 1.4.3, checked 2026-10-09)

`pip install pyBKT` on a fresh Python 3.13 environment, current dependencies:

1. **No compiled code.** PyPI's 1.4.3 ships only `pybkt-1.4.3-py3-none-any.whl` (24 files, no `.so`/`.pyd`).
   Cause, from the workflow history in section 5: 1.4.3 was built by `release.yml` with `python -m build` under build
   isolation, where NumPy was missing, and `setup.py` silently fell back to pure Python. Everyone on pip
   gets the ~75x slower path.
2. **`import pyBKT.models` fails with scikit-learn ≥ 1.8** (#58).
3. **With an older scikit-learn, fitting fails on NumPy 2** (#65): `EM_fit.py` line 43 assigns a (1, 1)
   array into a scalar slot. It is a one-line fix.
4. Metadata: `requires_python` is unset; classifiers list Python 3.5–3.8.

**On upstream master (unreleased):** 1 is fixed for Linux and macOS (#66, #69), 2 is fixed (#64, #68).
3 and 4 remain. None of it reaches users until a release.

## 5. Upstream sync (2026-10-09, CAHLR/pyBKT master at `cc1682e`)

Seven single-purpose PRs from this fork were merged on 2026-10-08. There has been no release since 1.4.3
(tag `1.4.3` on `06fc180`, 2026-08-05).

| Upstream PR | What it did | Plan item now done |
| --- | --- | --- |
| #64 | Skip scikit-learn metrics that fail the import-time probe (fixes #58) | K12 |
| #66 | NumPy declared in `[build-system] requires` | old A3 (build requirement) |
| #67 | `PYBKT_REQUIRE_CPP=1` fails the install when the C++ build fails (opt-in) | old A3 (loud failure) |
| #68 | scikit-learn optional; AUC computed with NumPy | not in the plan; fewer required dependencies |
| #69 | cibuildwheel: Linux x86_64/aarch64 (manylinux, musllinux), macOS arm64/x86_64, CPython 3.10–3.14; each wheel's test imports `E_step` | old A4 except Windows; K11 once released |
| #71 | C++ E-step reads each resource's and subpart's parameters (fixes #70) | old B6, K2 |
| #73 | Pure-Python E-step counts the last answer of one-attempt sequences (fixes #72) | K3; C1 no longer changes results |

**Still open on master** (read in the source, not inferred):
- K4 / #65: `source-py/pyBKT/fit/EM_fit.py:43`. The same line in `source-cpp/pyBKT/fit/EM_fit.py:25` is
  harmless only because the C++ result is an int, which is K1.
- K1: `source-cpp/pyBKT/fit/E_step.cpp:341`, `PyLong_FromLong(*total_loglike)`.
- K5: `data_helper.py:131`, `all_skills.str.match(skill_name)` with an unescaped name.
- K6: `data_helper.py:99` sorts by `order_id` with the default quicksort; the mergesort by user at
  `:117` keeps whatever order quicksort left for ties.
- Phase 1's C++ items: variable-length stack array `double s_likelihoods[2*T]` (`E_step.cpp:225`), sticky
  `omp_set_num_threads(1)` (`:60`), `delete memory` on `new[]` memory (`:34`).
- K7–K10, K13 untouched. No `tests/` directory and no test CI. Metadata as in section 4.

**Fork branches merged onto master in memory** (`upstream_sync_check.sh`; log in
`results/upstream_sync_check.log`). Nothing was committed or pushed by this check.

| Branch | Merge | Compiled, NumPy 2 | Pure Python, NumPy 2 | Pure Python, NumPy 1 |
| --- | --- | --- | --- | --- |
| `test/regression-harness` | clean | 13 pass, 3 fail | 6 pass, 2 skip, 8 xfail (#65) | 13 pass, 2 skip, 1 fail |
| `claude/pybkt-performance-research-xqv9jv` (phase 1) | clean | 21 pass, 3 fail | 11 pass, 5 skip, 8 xfail (#65) | 18 pass, 5 skip, 1 fail |

Every failure is `XPASS(strict)`: an expected-failure marker for a bug that is now fixed. Three markers
need deleting: `CPP_TEMPLATE_BUG` in `tests/test_backend_parity.py` and the #70 marker in
`tests/test_parameter_recovery.py` (compiled), and the #72 marker in `tests/test_one_attempt_students.py`
(pure Python). I had expected phase 1 to conflict with #71 in `E_step.cpp`; it doesn't. Git merges it
cleanly and the merged code reads per-class parameters (#71's loop-local `learn`/`forget`/`guess`/`slip`).

**Release path (found by reading the workflows; not run):** two workflows can publish to PyPI.
- `release.yml` runs when a GitHub release is published (and builds wheels on every PR). It builds the
  cibuildwheel wheels and the sdist, then publishes with trusted publishing. This is the path that
  published 1.4.3; the tag `1.4.3` has no `v`.
- `publish.yml` runs when a tag matching `v*` is pushed. It builds an sdist only (`uv build --sdist`),
  publishes it, then creates a GitHub release with the workflow token.
- Consequences: a `1.4.4`-style tag (as before) runs only `release.yml` and ships wheels. A `v`-prefixed
  tag created from the release page runs both, and both upload an sdist with the same filename. PyPI is
  likely to reject the second one, so one job fails. Pushing only a `v` tag ships an sdist with no wheels,
  because a release created with the workflow token does not trigger `release.yml`.

**Not checked:** issue and PR discussions upstream. This session can read CAHLR's git history but not its
issues, so comments on #55, #65 and others are unread. Check them before posting the tracking issue.

## 6. Adoption lens (from upstream history)

- **This works already:** the maintainer merged seven single-purpose PRs from this fork on 2026-10-08:
  #64 that afternoon, the other six in one six-minute batch that evening, hours after they were written.
  Each fixed one thing and said so. Repeat that shape, and expect merges in batches.
- Before that, upstream merged small community fixes in batches months apart (PRs #48, #51, #56 merged
  2026-03-04; #63 merged 2026-08-04, released as 1.4.3 the next day). Large changes have no precedent.
- So: each PR must be **independently mergeable**, small, tested by the harness, and readable in five
  minutes. Avoid chains where possible; where unavoidable (harness first), say so in the PR.
- Group PRs into **release trains**, each with a one-sentence promise a user understands. The maintainer
  can then release one train at a time.
- **No new required dependencies.** Optional extras only (`pyBKT[duckdb]`), and a separate package for Rust.
  Upstream is moving the same way (#68 made scikit-learn optional).
- **Propose before building** anything that adds API surface: one upstream tracking issue with the
  trains, linking evidence, asking which opt-in features they want in-tree.

## 7. Plan v3: release trains

Every PR description uses the same five lines: **what changes for users · class ([same]/[fix]/[opt-in])
· evidence (numbers, tests) · research basis (citation, if any) · how to verify (one command)**.
User-visible changes also get a README section; opt-in features get a notebook or docs page.

### Train A — "the next release installs, imports, fits and runs compiled" (everyone)
Done upstream: build requirement, `PYBKT_REQUIRE_CPP`, optional scikit-learn, Linux/macOS wheels, #58,
#70, #72 (#64–#73).

| PR | Content | Class | Size | Notes |
| --- | --- | --- | --- | --- |
| A1 | #65: NumPy 2 fit (one line in each `EM_fit.py`) | fix | S | first; unblocks pure Python on NumPy 2 |
| A2 | Regression harness on top of A1: delete the three obsolete markers and `FAILS_ON_NUMPY2`; CI runs compiled on Python 3.10–3.14 and pure Python on NumPy 1 and 2 | infra | M (506 lines, 12 files) | all later result-checked PRs depend on it |
| A3 | One release path: `release.yml` also runs on tag push, `publish.yml` retired; short "how to release" note | infra | S | the maintainer's process: ask in the tracking issue first |
| A4 | Metadata and install docs: `python_requires >= 3.10` (matches the wheel matrix), classifiers 3.10–3.14, README "Installing" says wheels are compiled | infra | S | |
| A5 | Windows wheels (MSVC, `/openmp`) | infra | M | open question 2 |
| — | Release request, with notes on the result changes from #70 and #72 | — | — | open question 1 |

### Train B — "results are correct and reproducible"
Result-changing fixes ([fix]) go into the tracking issue first; the maintainer decides each one.

| PR | Content | Class | Size |
| --- | --- | --- | --- |
| B1 | C++ safety from phase 1: heap scratch space instead of the stack array (crash at 200k answers), `delete[]`, leak, `parallel=False` no longer sticky | same | S |
| B2 | Escape skill names in the skill regex (K5) | fix | S |
| B3 | Stable order for tied `order_id` (K6) | fix | S |
| B4 | Handle missing user ids consistently (K8) | fix | S |
| B5 | Float log-likelihood from C++ (K1): stopping rule, best restart | fix | S |
| B6 | Deterministic parallel reduction in C++ (K9) | same (last bits) | S |
| B7 | multipair keys stable across pandas versions, still loading old keys (K7) | fix | M |

### Train C — "faster, same results"
| PR | Content | Benefit | Size |
| --- | --- | --- | --- |
| C1 | Vectorized NumPy E-step replaces the per-student pure-Python loop. Since #73 this changes no results; to re-check against master's pure Python | 54x pure-Python fit; matches C++ to 3.4e-15 | M |
| C2 | Forward-only predict (both builds) | 3–26x predict, multigs memory halved | S |
| C3 | Vectorized `convert_data` | 3–16x; multipair 290x | M |
| C4 | Skip alpha in EM; narrow dtypes; fix alpha's labelled shape (K13) | less memory traffic | S |
| C5 | Parallel policy (serial for small skills, parallel across skills), *after* a controlled measurement | up to 6x on many small skills, if confirmed | M |
| C6 | Sparse multigs layout; 64-bit indices (K10) | 1.47 GB -> MBs; > 2^31 answers | M |
| C7 | Read only the needed columns with narrow dtypes when given a file path (pandas only) | 61 s / 5.0 GB -> 33 s / 2.6 GB at 20M rows; no new dependency | S |
| C8 | Prepared-data object: convert once, reuse across `fit`, `predict`, `crossvalidate` and restarts; stop mutating and re-sorting the caller's DataFrame | removes repeated conversion; the Rust work showed per-call copies cost ~40% of a fast E-step | M |
| C9 | All `num_fits` restarts in one pass over the data; skills in parallel; schedule threads by answers, not students | default `num_fits=5` is 4.3x the single-fit time today | M |

### Train D — "better parameters" (opt-in, research-backed; propose in the tracking issue first)
| PR | Content | Research | Size |
| --- | --- | --- | --- |
| D1 | Fit diagnostics: implausible / boundary / restart disagreement warnings | Pardos & Heffernan 2010; Beck & Chang 2007 | S |
| D2 | Beta priors (MAP-EM), `priors=` | Beck & Chang 2007 | M |
| D3 | SQUAREM, `accelerate="squarem"` | Varadhan & Roland 2008 | M |

### Train E — "live tutors and large data" (opt-in)
| PR | Content | Research | Size |
| --- | --- | --- | --- |
| E1 | Live mastery update API (issue #55), normalized update | forward filtering | S |
| E2 | Optional DuckDB loader (`pyBKT[duckdb]`) | — | M |
| E3 | Compact store + exact out-of-core EM via forward smoothing | Cappé 2011 Prop. 1 | L |
| E4 | Student-level online EM, experimental: warm-start from a batch fit, periodic batch refit as anchor, pseudo-counts; plus docs saying what today's `partial_fit` does (a warm-started refit on new data only) | Cappé & Moulines 2009; Neal & Hinton 1998 | M |
| E5 | Prefix sharing | — | M |

### Separate track R — optional Rust package
`bkt_lean` as a separate optional package with the same API (`pip install pybkt-rs`), picked up when
installed. Does not ask upstream to adopt a Rust toolchain. Proposed only after Trains A–C.

### Not planned (documented as researched, with the reason)
| Idea | Reason | Where |
| --- | --- | --- |
| Event-level online EM | only alpha = 0.8 stable in our multi-student adaptation | README §2 |
| GPU backend | no benefit at typical sizes; wgpu best fit if ever needed | README §5 notes, chat |
| Gradient-based fitter (BKT as a PyTorch RNN) | flexibility for extensions (BKT+IRT), not speed; heavy dependency | README §9 |
| Parallel scan over time (Särkkä & García-Fernández 2021) | would help very long sequences; untested | README §9 |
| nanobind / pybind11 binding swap | binding overhead isn't the cost; smaller wheels only | `deep_research_summary.md` |
| Numba / Cython pure-Python path | vectorized NumPy gets 54x with no new dependency | ROADMAP alternatives |
| Polars as a required dependency | DuckDB/pyarrow optional is enough; pandas path improved instead | README §7 |

### Tracking issue outline (to draft once open question 3 is answered)
1. One paragraph: what was researched, and that #64–#73 were the first results.
2. Train A: what's left before a release (A1–A5) and the release-path note.
3. Result-changing fixes for the maintainer to decide (B2–B5, B7): one line each with the effect, the
   size of the change, and how the harness shows the new numbers.
4. Speed, results unchanged (B1, B6, Train C): one line each with the measured benefit.
5. Opt-in research features (Trains D and E): which, if any, they want in-tree, and where docs should live.
6. Rust: a separate package later; no change to upstream's toolchain.
7. Links to evidence on the fork (this directory). Keep the issue itself under one screen per section.

## 8. Critique

### Plan v2 points, revisited after the sync
1. ~~A4 (wheels) is the hardest PR.~~ Done upstream for Linux and macOS. Windows is the open part.
2. **Releases are outside our control.** Still true, and now the main lever (point 13).
3. **Result changes need a compatibility story.** Sharper now: master already changes results for C++
   multigs/multilearn fits (#70) and pure-Python fits with one-answer students (#72). The next release
   changes numbers whether or not we add anything. B5 adds a stopping-rule change on top.
4. **Reference-file conflicts:** B2–B5 all touch `tests/reference/*.json`; sequence them. C1 no longer
   does (it now matches master).
5. **Opt-in research features are the least certain to be wanted.** The tracking issue is the gate.
6. ~~Docs home undecided.~~ Decided in round 2: README sections and notebooks or docs pages.
7. **Evidence portability:** each PR cites its own reproducible check, not this research's headlines.
8. **Cut line:** A–C make pyBKT work and fast for everyone; D–E stay proposals until the maintainer opts in.
9. C8 may land better as an internal cache first, with a public object proposed later.
10. C7 must keep every column the model type needs; its test covers each model type.
11. C9 must keep results identical, drawing random initial parameters in today's order.
12. **About 30 PRs:** now 29 (old A3 and B6 merged upstream; the release-path PR added). Still too many to
    send at once.

### New points (v3)
13. **The release is now worth more than any PR.** Master already gives Linux and macOS users compiled
    wheels and fixes #58, #70 and #72. Until a release, every `pip install` still gets 1.4.3. A1 is one
    line. Gating the release on more than A1 (or A1 + A2) delays the biggest win.
14. **The release itself is a risk.** It is the first cibuildwheel release, and the publish path is
    ambiguous (section 5). The cheapest mitigation is one sentence in the release request ("tag it like
    1.4.3, without a `v`"); A3 fixes it for good. Also check the wheel list on PyPI after the release.
15. **C1's value depends on Windows.** Once wheels ship, pure Python mainly serves Windows (unless A5),
    other platforms without wheels, and failed builds. With Windows wheels, C1 drops in priority; without
    them, it stays high.
16. **The harness adds CI cost and upkeep:** 7 jobs per PR and reference JSON to regenerate. The upside is
    that each result-changing PR shows its effect as a JSON diff, which is exactly what "ask the
    maintainer" needs.
17. **Phase 1 is a branch, not a PR.** Split it into B1, C2, C3 off master. Keep `benchmarks/bench.py` and
    `ROADMAP.md` on the fork, or offer them later as one docs PR.
18. **Unread discussions:** the tracking issue should be checked against existing issues (#55, #65 and
    any newer ones) so it doesn't duplicate or contradict them.
19. **IDs:** inventory bugs are now K1–K13, so "B3" always means a Train B PR.

### First batch (proposal)
A1 (#65), A2 (harness), A4 (metadata/docs), B1 (C++ safety, results unchanged), and the release request.
All results-unchanged except A1, which only removes a crash. A3 and A5 wait for the maintainer's answer in
the tracking issue.

## 9. Open questions (round 3)
1. When to ask for a release: right after A1, after A1 + A2, or after the first batch?
2. Windows wheels (A5): in Train A, later, or not at all?
3. The tracking issue: draft the full text here for you to post, or will you write it?
4. Release path: send A3 as a PR, mention the tag format in the release request only, or ask in the issue?

## 10. Coverage check: every experiment mapped to the plan

| Experiment (script) | Finding | Plan item |
| --- | --- | --- |
| Phase 1 branch (4 commits) | crash, leaks, sticky threads; predict 3–26x; convert 3–16x | B1, C2, C3 |
| Golden-output checks (`golden.py`, kernel bit-identity) | phase 1 changes are bit-identical | evidence for B1, C2, C3 |
| Long-sequence / conversion tests | segfault at 200k answers; conversion behaviour pinned | B1, C3 (tests ship with them) |
| `bench.py` + ROADMAP findings | where time and memory go | A1/C PR evidence |
| `check_smoothing.py` | forward-only smoothing equals forward-backward (2.5e-16) | E3 basis |
| `compare_em.py`, `compare_online.py`, `synth_discount.py` | ad-hoc online variants lag; filtering-only is biased | E4 design (what not to do) |
| `cappe_moulines.py`, `cm_real.py`, `online_em.py` | C&M eq. 15 one pass ≈ batch; median held-out gap 0.0012 | E4 |
| `cappe2011_events.py` | event-level stable only at alpha = 0.8 | Not planned |
| `degeneracy_audit.py` | 9% implausible, 12% boundary, 74/110 skills disagree across restarts | D1 |
| `map_em.py` | priors: +0.0023 held-out ll, boundary fits 15% -> 0% | D2 |
| `squarem.py`, `squarem_tol.py` | 1.4–1.7x fewer EM steps, no basin change | D3 |
| `np_estep.py`, `py_vs_vec_fit.py`, `vec_vs_cpp_fit.py`, `ab_estep.py` | 54x pure-Python fit, exact vs C++; stable form costs 11–15% | C1 |
| `dedup.py` | prefix sharing 2.75–25x less work (counted) | E5 |
| `load_bench.py`, `load_breakdown.py` | pandas `usecols` halves time/memory; DuckDB fastest | C7, E2 |
| `duckdb_convert.py` | whole `convert_data` in SQL, identical output | E2 (and B3's tie-break) |
| `conv_mem.py` | pandas 3 + pyarrow strings change memory, not a regression | docs note in C3 |
| `e2e_pybkt.py`, `omp_overhead.py` | default parallel 6x slower than serial under load; 7.9 µs/call idle | C5 (after re-measurement) |
| `live_bench.py` | O(1) live updates exact to 1.6e-9; naive formula unstable; regex and tie bugs | E1, B2, B3 |
| Rust `bkt_rs`, `bkt_lean`, `rayon_simd.py`, `lean_gap.py` | 2.2x serial, 10–35x SIMD+threads; copy-once matters; rayon optional | Track R; C8 (convert once) |
| Dependency and safety audits (numpy, Arrow, Parquet, wgpu, rayon) | pyo3-only lean build; rayon acceptable | Track R |
| GPU assessment | not worth it at typical sizes | Not planned |
| Literature checks (KT², Khajah, deep research) | online EM for BKT parameters is new for BKT; gradient fitters for flexibility | D/E docs; Not planned |
| PyPI check (planning round 2) | pure-Python wheel; import fails on scikit-learn ≥ 1.8; fit fails on NumPy 2 | A1, A4, release request (rest fixed upstream) |
| `upstream_sync_check.sh` (planning round 3) | both fork branches merge cleanly onto master; only obsolete xfail markers fail | A2, B1, C2, C3 |
