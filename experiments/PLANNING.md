# Turning the research into deliverable chunks: working notes

Status: decision log, revised as decisions are made (latest: 2026-10-09, planning round 5).
**The actionable plan is `HANDOFF.md`**; upstream texts are in `ISSUE_DRAFTS.md`. Evidence for every
number is in `README.md` (sections in brackets) or `rust/REPORT.md`.

Contents: 1 inventory · 2 ways to slice · 3 decisions · 4 what a new user gets · 5 upstream sync ·
6 adoption lens · 7 plan v3 (with prerequisites, conflicts, evidence branch) · 8 critique ·
9 open questions · 10 coverage check.

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
| K8 | Missing user ids: `groupby` drops them from `lengths`, the rows stay at the end of the data | fit ignores those rows silently; predict returns invalid values for them (1.12 and 0.0 in pure Python, 0.0 compiled); `nan_user_ids.py` (an earlier note said "misaligned sequences"; a test shows they aren't) | tiny |
| K9 | C++ parallel reduction order nondeterministic | last-bit differences run to run | small |
| K10 | `bigT` / indices are 32-bit | overflow past 2^31 answers per skill | small |
| K11 | PyPI 1.4.3 wheel is pure Python (`py3-none-any`, no extension) | every pip user gets the ~75x slower path, with #72 | **fixed on master (#66, #69)**; needs a release |
| K12 | #58: `import pyBKT.models` fails with scikit-learn ≥ 1.8 | package unusable on current scikit-learn | **fixed on master (#64, #68)**; needs a release |
| K13 | `E_step.run` returns `alpha` as an N x 2 buffer labelled (2, N) | confusing internal layout (found by the Rust work) | small, internal |

### Performance, results unchanged
| # | Item | Measured benefit | Cost / risk |
| --- | --- | --- | --- |
| P1 | Serial below a size threshold, or parallel across skills, instead of OpenMP inside small E-steps | controlled 2026-10-09: ~8 ms per OpenMP call whenever cores are busy (up to 92x slower than serial), 1.7–2.9x faster when idle; dynamic scheduling doesn't help | B8 (wave 2) and C9 |
| P2 | Skip writing `alpha` during EM | 16 B/answer per iteration not written | tiny |
| P3 | Sparse multigs layout (one template index per answer) | 1.47 GB -> ~2 MB on merged ASSISTments | medium; touches kernel and converter |
| P4 | Narrow dtypes (int8 answers, int32 resources) | ~4x less kernel input memory | small |
| P5 | Vectorized NumPy E-step for the pure-Python backend | 20–24x vs pyBKT's default parallel fit, 32–54x vs serial (re-measured 2026-10-09); matches C++ to 3.4e-15 | depends on K4 (K3 now fixed upstream); stable posterior-form backward pass |
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
- **Process:** open a tracking issue upstream first, listing the trains, then send PRs. (Superseded in
  round 5: topic issues, no umbrella.)
- **Result-changing fixes:** ask the maintainer, case by case (listed in the tracking issue).
- **First train:** sync with upstream first; some issues are already logged and fixed (done in section 5).
- **Explaining the work:** a five-line summary on every PR, README sections for user-visible changes, and
  notebooks or docs pages for the opt-in features.

Round 3:
- **Release ask:** after A1 (NumPy 2 fix) and A2 (harness) are merged.
- **Windows wheels:** yes, in Train A (A5).
- **Tracking issue:** drafted here for the owner to review and post (now `ISSUE_DRAFTS.md`, split into
  topic issues in round 5).
- **Release path:** one sentence in the release ask ("tag it like 1.4.3, without a `v`"); propose the
  cleanup (A3) in the tracking issue for later.

Round 4:
- **Train A's remainder** goes into #65 as follow-up comments; the new tracking issue covers the rest.
- **Scope:** #45 only through the diagnostics warning (D1); #50 later, outside the first batches.
- **Evidence:** a clean, neutrally named fork branch with a trimmed set (section 7, "Evidence branch").
- **Next:** keep refining the plan before any branch work.

Round 5 (after reading all 55 upstream issues and the full git history):
- **Issue layout:** topic issues, no umbrella; existing threads reused (#65, #55/#57, #54).
- **Cadence:** staged waves. Wave 1 now: #65 follow-up, Rust discussion, single-tree proposal, #54
  comment. Wave 2 after the release: bug issues one at a time with their PRs, live-update proposal on #55.
  Wave 3: speed, diagnostics, optional estimators, large data.
- **Bug reports:** one issue per result-changing bug, each with a short repro.
- **Priority after Train A:** correctness, then live updates.
- **Single Python source tree:** propose early (wave 1); build it only if welcomed, before the Python
  bug fixes.
- **Windows wheels:** ask in #65 first, because of #32.
- **`data_helper.py` order:** bug fixes first; the vectorized `convert_data` (C3) comes after and keeps
  the fixed behaviour.
- **Rust:** a discussion issue now, framed as "SIMD and multithreading help, but they are riskier to build
  in C and C++"; build nothing upstream until the maintainer replies.

Round 7 (after fetching upstream's side branches and the 13 active forks):
- **Single source tree:** cite the unmerged `nopy` branch (2023, C++ only) and offer both directions; (a),
  one tree with a NumPy fallback, recommended.
- **Class-value cluster:** build on the unmerged `noerr` branch (2022) with credit; fix its multigs
  fallback.
- **NumPy 2 PR:** credit the three forks that fixed it first, in the PR text.
- **WIP limit:** at most three upstream PRs open at once; result-changing fixes one at a time.

Round 6 (after a controlled contention study and the rest of the issue tracker):
- **Class-value cluster** (#29, #45, #47, #50, #52): wave 2, after live updates; reproduce first.
- **OpenMP default for small calls** (B8, was C5): wave 2 with the C++ fixes. Measured: ~8 ms per call
  when cores are busy; dynamic scheduling doesn't help.
- **Pure-Python process pool:** fixed only by the vectorized E-step (C1); no separate pool PR.
- **Rust design to propose:** `bkt_lean` plus rayon (work stealing didn't stall under contention; per-call
  scoped threads did).

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

**Upstream issues** (read 2026-10-09 from the public issue pages; the API isn't available for CAHLR here):

| Issue | State | What it says | Effect on the plan |
| --- | --- | --- | --- |
| #65 "Publishing compiled wheels, and making scikit-learn optional" (owner's) | open | three problems: build isolation, **NumPy 2 fit fails at `EM_fit.py:43`**, scikit-learn on musllinux. zpardos (2026-10-09): "the three proposals sound good"; default `PYBKT_REQUIRE_CPP=0` and **document it in the README** | Train A continues this thread: A1 is its item 2; A4 adds the README note he asked for; the release ask goes here |
| #55 incremental/online updates (2025-10) and #57 constant-time `update` API (2025-11) | open, no maintainer reply | both ask for a one-step mastery update with fitted parameters (`update_single_step(skill, prior, obs)`, `update_single(prior, is_correct, skill)`), not online parameter learning | E1 answers both; E4 (online parameters) is a different request nobody has made |
| #45 multilearn: `crossvalidate` returns an AUC where fit + predict raise | open | collaborator (2023-11): crossvalidate makes "best effort" 0.5 predictions for unknown parameters; offered consistency or a warning | same family as K5's silent 0.5 predictions; a warning fits D1 |
| #50 multigs model fails in `Roster` | open | `ValueError` broadcasting (186,) into (1,); a second report traces `process_data` using `True` instead of the class names | not in the research; candidate for Train B after a reproduction |
| #54 `random.randint(0, 1e8)` | open | fixed by merged PRs #48 and #56 | housekeeping: can be closed |
| #39 seeded C++ models not deterministic | closed (#41) | seeding fixed | K9 is a different, last-bit effect of the parallel reduction |
| #37 "Latest pyBKT not built", #60 "New pip release for bugfix" | closed | earlier build and release requests | precedent: release asks get answered |
| #53 `No module named 'pyBKT'` | open, no details | — | may be a failed build; wheels may help, unconfirmed |

**Side effect found:** fork commits whose messages contain `#N` show up in the upstream issue's timeline
(#65 shows this plan's commit). Earlier fork commits mention #72 too. From now on, fork commit messages
say "upstream issue 65" without the `#`.

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
| A1 | #65 item 2: NumPy 2 fit (one line in each `EM_fit.py`) | fix | S | first; unblocks pure Python on NumPy 2 |
| A2 | Regression harness on top of A1: delete the three obsolete markers and `FAILS_ON_NUMPY2`; CI runs compiled on Python 3.10–3.14 and pure Python on NumPy 1 and 2 | infra | M (506 lines, 12 files) | all later result-checked PRs depend on it |
| A3 | One release path: `release.yml` also runs on tag push, `publish.yml` retired; short "how to release" note | infra | S | the maintainer's process: ask in the tracking issue first |
| A4 | Metadata and install docs: `python_requires >= 3.10` (matches the wheel matrix), classifiers 3.10–3.14, README "Installing" says wheels are compiled and documents `PYBKT_REQUIRE_CPP` (asked for in #65) | infra | S | |
| A5 | Windows wheels (MSVC, `/openmp`) | infra | M | decided: in Train A; needs B1 first (prerequisites below); can follow the release |
| — | Release request in #65, after A1 + A2, with the tag note and the result changes from #70 and #72 | — | — | decided |

### Train B — "results are correct and reproducible"
Result-changing fixes ([fix]) go into the tracking issue first; the maintainer decides each one.

| PR | Content | Class | Size |
| --- | --- | --- | --- |
| B1 | C++ safety from phase 1: heap scratch space instead of the stack arrays (crash at 200k answers; also what MSVC needs for A5), `delete[]`, leak, `parallel=False` no longer sticky | same | S |
| B2 | Escape skill names in the skill regex (K5) | fix | S |
| B3 | Stable order for tied `order_id` (K6) | fix | S |
| B4 | Missing user ids (K8): drop them with a warning, or raise; never return values above 1 | fix | S |
| B5 | Float log-likelihood from C++ (K1): stopping rule, best restart | fix | S |
| B6 | Deterministic parallel reduction in C++ (K9) | same (last bits) | S |
| B7 | multipair keys stable across pandas versions, still loading old keys (K7) | fix | M |

### Train C — "faster, same results"
| PR | Content | Benefit | Size |
| --- | --- | --- | --- |
| C1 | Vectorized NumPy E-step replaces the per-student pure-Python loop and its per-iteration process pool. Since #73 this changes no results (re-checked 2026-10-09 against master's pure Python: within 3.4e-15) | 20–24x vs the default parallel fit, 32–40x vs serial | M |
| C2 | Forward-only predict (both builds) | 3–26x predict, multigs memory halved | S |
| C3 | Vectorized `convert_data` | 3–16x; multipair 290x | M |
| C4 | Skip alpha in EM; narrow dtypes; fix alpha's labelled shape (K13) | less memory traffic | S |
| C5 | Parallel policy (serial for small skills, parallel across skills), *after* a controlled measurement | up to 6x on many small skills, if confirmed | M |
| C6 | Sparse multigs layout; 64-bit indices (K10) | 1.47 GB -> MBs; > 2^31 answers | M |
| C7 | Read only the needed columns with narrow dtypes when given a file path (pandas only) | 61 s / 5.0 GB -> 33 s / 2.6 GB at 20M rows; no new dependency | S |
| C8 | Prepared-data object: convert once, reuse across `fit`, `predict` and `evaluate` calls (restarts and `crossvalidate` folds already reuse one conversion); stop mutating and re-sorting the caller's DataFrame | removes repeated conversion; the Rust work showed per-call copies cost ~40% of a fast E-step | M |
| C9 | All `num_fits` restarts in one pass over the data; skills in parallel; schedule threads by answers, not students | default `num_fits=5` is 4.3x the single-fit time today | M |

### Train D — "better parameters" (opt-in, research-backed; propose in the tracking issue first)
| PR | Content | Research | Size |
| --- | --- | --- | --- |
| D1 | Fit diagnostics: implausible / boundary / restart disagreement warnings, and a warning when predictions fall back to 0.5 (#45) | Pardos & Heffernan 2010; Beck & Chang 2007 | S |
| D2 | Beta priors (MAP-EM), `priors=` | Beck & Chang 2007 | M |
| D3 | SQUAREM, `accelerate="squarem"` | Varadhan & Roland 2008 | M |

### Train E — "live tutors and large data" (opt-in)
| PR | Content | Research | Size |
| --- | --- | --- | --- |
| E1 | Live mastery update API (issues #55 and #57), normalized update | forward filtering | S |
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
| Numba / Cython pure-Python path | vectorized NumPy gets 20–40x with no new dependency | ROADMAP alternatives |
| Polars as a required dependency | DuckDB/pyarrow optional is enough; pandas path improved instead | README §7 |

### Prerequisites and conflicts (found in planning round 4)

**Windows wheels (A5) depend on B1.** Upstream has never built the C++ extension on Windows (README:
"Optional - for OS X and Linux"). A strict check (`g++ -std=c++17 -pedantic-errors -Wvla`, a stand-in for
MSVC, which has no compiler here) finds on master:
- five variable-length stack arrays in `E_step.cpp` (lines 200, 201, 225, 243, 262), four of them with
  GCC's `__attribute__((aligned(16)))`. MSVC rejects both. Phase 1 (B1) replaces them with
  `std::vector`; the same check on phase 1 merged onto master finds none.
- `#include <alloca.h>` in `E_step.cpp`, `predict_onestep_states.cpp` and `synthetic_data_helper.cpp`.
  MSVC has no such header. No file calls `alloca`, so the includes are dead; phase 1 removes only the
  first.
- `setup.py` passes GCC flags (`-fopenmp`, `-fPIC`) on every non-macOS platform; MSVC needs `/openmp`.
So A5 = B1 + delete two dead includes + an MSVC branch in `setup.py` + `windows-latest` in the wheel
matrix. `release.yml` builds wheels on every PR, so the A5 PR tests itself.

**Two copies of the Python code.** `source-cpp/pyBKT` and `source-py/pyBKT` each hold the same 24 Python
files; 17 are identical, 7 differ (`EM_fit.py` by 230 lines, `predict_onestep.py` by 67, the others by
2–4). Every Python change is made twice: phase 1's `convert_data` change is 88 lines in each copy. #72
was a drift between the copies. One tree, with the compiled E-step optional at import, would halve every
later Python PR and remove that class of bug. It's a packaging change, so it needs the maintainer's
interest first (open question 1).

**Conflict hotspots** (several PRs edit the same file; merge them in this order):

| File | PRs touching it | Suggested order |
| --- | --- | --- |
| `fit/E_step.cpp` | B1, B5, B6, C4, C5, C6, E5 | B1 (also unblocks A5), B5, B6, C4, C6, C5 after its measurement |
| `util/data_helper.py` (two copies) | C3, B2, B3, B4, B7, C6, C7 | C3 first (same results, vectorized), then each fix as a small diff on top, so every result change is isolated |
| `models/Model.py` (two copies) | B7, C8, C9, D1, D2, E1 | E1 and D1 add code only; C8/C9 last (largest) |
| `fit/EM_fit.py` (two copies, differ) | A1, B5, C1, C4, C9, D3 | A1 first; C1 replaces the pure-Python loop |
| `tests/reference/*.json` | every [fix] PR | one at a time; each PR's JSON diff shows its result change |

### Evidence branch (decided in round 4; not created yet)
A neutrally named fork branch holding only what a reader of the tracking issue needs:
- `EVIDENCE.md`: every number in the issue mapped to a script, a command, a result file and the data.
- **Small standalone repros** for the section 1 bugs: `integer_loglike.py` (K1), `regex_skill_names.py`
  (K5), `tied_order_ids.py` (K6), `nan_user_ids.py` (K8); 20–30 lines on synthetic data, seconds to run,
  no ASSISTments download. K7 needs pandas 2 and 3 environments, so it is described instead.
- The scripts behind the speed and research numbers (`bench.py`, `py_vs_vec_fit.py`, `load_bench.py`,
  `duckdb_convert.py`, `degeneracy_audit.py`, `map_em.py`, `squarem.py`, `squarem_tol.py`,
  `check_smoothing.py`, `cappe_moulines.py`, `live_bench.py`), their result CSVs, `fetch_data.sh`, and
  `rust/` (lean crate and REPORT only).
- Left out: `PLANNING.md`, `HANDOFF.md`, `ISSUE_DRAFTS.md`, superseded scripts (`np_estep_old.py`, the ad-hoc online
  variants), session notes.

### Issue layout (round 5)
Topic issues in waves replace the single tracking issue; texts in `ISSUE_DRAFTS.md`, diagram and order in
`HANDOFF.md` sections 3 and 4. Train B's order is B5 (integer log-likelihood, a 2021 regression), B2
(regex names), B4 (missing ids), B3 (ties), B7 (multipair keys), with B6 after B5. T1 (one Python source
tree) is new: proposed in wave 1, built before B2–B4 only if welcomed.

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

### New points after reading the issues (v3.1)
20. **Train A already has a home.** #65 is the owner's issue and the maintainer approved its three
    proposals. Posting Train A's remainder (A1, the release ask) as a follow-up there keeps one thread per
    topic. The new tracking issue then covers only Trains B–E and R, which makes it shorter.
21. **Live updates are the most-requested feature** (#55, #57, a year without a reply). E1 is small, needs
    no new dependency, and is exact. It may deserve to move ahead of Train C.
22. **The tracking issue can close loops:** #54 is already fixed; #45 and #53 get a pointer to the work
    that addresses them. That's useful to the maintainer even if no opt-in feature is accepted.
23. **Fork-commit backlinks** (section 5) are visible to the maintainer before the tracking issue exists.
    Harmless, but the tracking issue should come soon so the links have context.

### New points (v4, planning round 4)
24. **Windows wheels need B1 first** (prerequisites above). The A5 PR can't be tested here (no MSVC), but
    the wheel workflow runs on PRs, so CI tests it. Expect a few iterations.
25. **Two copies of the Python code double the review load** and caused #72. Consolidating first would
    make every later Python PR smaller, but it is the largest packaging change in the plan and comes
    from an outside contributor. Ask before building.
26. **The harness runs on Linux only.** Running it inside cibuildwheel (`CIBW_TEST_COMMAND`) would test
    each compiled wheel on macOS, Windows, musl and aarch64. Its tolerances (`rtol` 1e-7 for outputs,
    1e-9 for EM parity) are untested off Linux.
27. **Hotspot order matters more than train order.** `E_step.cpp` and `data_helper.py` each have 6–7 PRs
    queued. Merging them out of order means rebasing every time.
28. **Repros persuade maintainers more than research scripts.** Section 1 of the issue should point to a
    20-line repro per bug.

### First batch (proposal)
A1 (#65 item 2), A2 (harness), A4 (metadata/docs), B1 (C++ safety, results unchanged), and the release
request after A1 + A2. All results-unchanged except A1, which only removes a crash. A5 (Windows) follows
B1. A3 waits for the maintainer's answer in the tracking issue.

## 9. Open questions
Rounds 5 and 6 are answered (section 3). Still open, with defaults in `HANDOFF.md` section 9: the
evidence branch name (default `evidence/bkt-research`) and contents; running the harness on every wheel
platform. Waiting on the maintainer: release and Windows (#65), the Rust discussion, the single-tree
proposal.

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
| `np_estep.py`, `py_vs_vec_fit.py`, `vec_vs_cpp_fit.py`, `ab_estep.py` | 20–24x vs the default parallel pure-Python fit, 32–54x vs serial; exact vs C++; stable form costs 11–15% | C1 |
| `dedup.py` | prefix sharing 2.75–25x less work (counted) | E5 |
| `load_bench.py`, `load_breakdown.py` | pandas `usecols` halves time/memory; DuckDB fastest | C7, E2 |
| `duckdb_convert.py` | whole `convert_data` in SQL, identical output | E2 (and B3's tie-break) |
| `conv_mem.py` | pandas 3 + pyarrow strings change memory, not a regression | docs note in C3 |
| `e2e_pybkt.py`, `omp_overhead.py` | default parallel 6x slower than serial under load; 7.9 µs/call idle | C5 (after re-measurement) |
| `live_bench.py` | O(1) live updates exact to 1.6e-9; naive formula unstable; regex and tie bugs | E1, B2, B3 |
| `nan_user_ids.py` (planning round 4) | missing user ids: silent drop in fit, invalid predictions | B4 |
| `omp_controlled.sh`/`.py`, `results/omp_controlled.csv`, `results/omp_dynamic.csv` (round 6) | OpenMP calls stall ~8 ms under contention; dynamic scheduling doesn't help | B8, C9; Rust framing |
| `rust/contention.sh`, `rust/contention_small.py` (round 6) | rayon work stealing avoids the stall on small calls; per-call scoped threads don't; big calls degrade proportionally everywhere | R design (`bkt_lean` + rayon) |
| `py_vs_vec_fit.py parallel` (round 6) | vectorized E-step is 20–24x faster than pyBKT's default process pool, 32–40x than serial; the pool costs ~25 ms per iteration | C1 |
| Remaining upstream issues (round 6) | class-value cluster #29/#45/#47/#50/#52; process pool on Windows #11/#51/#42 | CV, C1 |
| Upstream side branches and forks (round 7) | `noerr` (class values, 2022), `nopy` (drop pure Python, 2023), three independent NumPy 2 fixes, a Windows pool-size fix, an earlier build-backend attempt; pyBKT-examples' two issues are Windows multiprocessing | CV, T1, A1, C1 |
| Rust `bkt_rs`, `bkt_lean`, `rayon_simd.py`, `lean_gap.py` | 2.2x serial, 10–35x SIMD+threads; copy-once matters; rayon optional | Track R; C8 (convert once) |
| Dependency and safety audits (numpy, Arrow, Parquet, wgpu, rayon) | pyo3-only lean build; rayon acceptable | Track R |
| GPU assessment | not worth it at typical sizes | Not planned |
| Literature checks (KT², Khajah, deep research) | online EM for BKT parameters is new for BKT; gradient fitters for flexibility | D/E docs; Not planned |
| PyPI check (planning round 2) | pure-Python wheel; import fails on scikit-learn ≥ 1.8; fit fails on NumPy 2 | A1, A4, release request (rest fixed upstream) |
| `upstream_sync_check.sh` (planning round 3) | both fork branches merge cleanly onto master; only obsolete xfail markers fail | A2, B1, C2, C3 |
| `integer_loglike.py`, `regex_skill_names.py`, `tied_order_ids.py` (planning round 5) | bug repros: whole-number EM trace stops after 15 iterations once two values round alike; regex-named skill predicted at 0.5 on both call paths; same rows in two orders give parameters 0.14 (compiled) / 0.63 (pure Python) apart | B5, B2, B3 and their issues |
| Upstream issues (all 55) and git history (393 commits), round 5 | integer log-likelihood is a 2021 regression (`dba0ddb`); a 2023 Windows/macOS wheel attempt was removed the same day; Windows C++ unsupported (#32); `Roster` already updates per answer; #21/#27/#36/#38 show the degenerate-fit confusion | B5, A5, E1, D1; Rust framing |
