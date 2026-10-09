# Handoff: pyBKT correctness, speed and live-update work

Start here. This file is the plan another agent can pick up and run; it is current as of 2026-10-09.
- `PLANNING.md` is the decision log and the reasoning behind this plan (inventory, critiques, rounds of
  owner questions).
- `ISSUE_DRAFTS.md` holds the texts for upstream comments and issues, wave by wave.
- `README.md` (this directory) holds the research evidence and replication commands; `rust/REPORT.md` the
  Rust prototypes.

Owner: Christian Murphy (fork `ChristianMurphy/pyBKT`). Upstream: `CAHLR/pyBKT`, maintained by Zachary
Pardos (`zpardos`); Anirudhan Badrinath (`abadrinath947`) answered most issues from 2021 to 2024.

## 0. Rules for whoever picks this up

1. **Nothing goes upstream without the owner's go-ahead for that item**: no PR, issue or comment on
   CAHLR/pyBKT, and no PR on the fork. Prepare branches and drafts; the owner posts or approves.
2. **Fork commit messages never contain `#<number>`.** GitHub links them into the upstream issue's
   timeline (this happened to #65 and #72). Write "upstream issue 65" instead. PR descriptions and issue
   texts can use `#65`; that is where links belong.
3. **One change per PR**, independently mergeable, tested by the harness, small enough to review in five
   minutes. Every PR description has five lines: **what changes for users · class · evidence · research
   basis (citation, if any) · how to verify (one command)**. Classes: `[same]` (results bit-identical or
   within float tolerance), `[fix]` (a bug fix that changes results), `[opt-in]` (new, optional),
   `[infra]` (tests, build, docs).
4. **Result-changing fixes are the maintainer's call**, each through its own issue with a repro.
5. **No new required dependencies.** Optional extras only (`pyBKT[duckdb]`); Rust only as a separate
   optional package unless the maintainer asks otherwise.
6. **Rust code:** `#![forbid(unsafe_code)]` in our crates; avoid `unsafe`-heavy dependencies where
   practical; avoid marshalling copies (convert once, borrow buffers, let numpy allocate outputs).
7. **Branches:** research goes on `claude/pybkt-experiments-xqv9jv`. Each upstream PR gets its own branch
   off `cahlr/master`, named like the merged ones (`fix/...`, `feat/...`, `build/...`, `ci/...`,
   `test/...`). Ask the owner before pushing to an existing owner branch such as `test/regression-harness`.
8. **Before any push:** run the checks in section 8 and say plainly what passed and what didn't.

## 1. Where things stand

**Upstream master `cc1682e`** (2026-10-08) merged seven PRs from the owner: #64 (scikit-learn 1.8 import,
fixes #58), #66 (NumPy build requirement), #67 (`PYBKT_REQUIRE_CPP`), #68 (scikit-learn optional), #69
(cibuildwheel wheels: Linux x86_64/aarch64 manylinux and musllinux, macOS arm64/x86_64, CPython
3.10–3.14), #71 (fixes #70), #73 (fixes #72). **No release since 1.4.3**, which is a pure-Python wheel
that fails to import with scikit-learn ≥ 1.8 and fails to fit on NumPy 2.

**Fork branches**

| Branch | Content | State against upstream master |
| --- | --- | --- |
| `test/regression-harness` (`97a0eb1`) | test suite + Tests workflow, 12 files, 506 lines | merges cleanly; 3 strict-xfail markers now XPASS because #70/#72 are fixed |
| `claude/pybkt-performance-research-xqv9jv` | phase 1: `34e3d21` C++ crash/leak fixes (+ a forward-only `E_step.predict`), `fda90e7` forward-only predict, `37f57be` vectorized `convert_data`, `0a17b76` `bench.py` + `ROADMAP.md` | merges cleanly (including `E_step.cpp` with #71); same 3 markers |
| `claude/pybkt-experiments-xqv9jv` | this directory: research scripts, results, plan, drafts | not for upstream |

`upstream_sync_check.sh` reproduces the merge-and-test check; the log is `results/upstream_sync_check.log`.

**Upstream issues that bear on the plan** (all 55 issues and 18 PRs were read on 2026-10-09):

| Issue | What it says | Where it lands |
| --- | --- | --- |
| #65 (owner's, open) | wheels, NumPy 2 fit (item 2), optional scikit-learn; zpardos approved all three on 2026-10-09 and asked for README docs on `PYBKT_REQUIRE_CPP` | Train A continues here |
| #55, #57 (open) and #16, #19, #22 (closed, 2021) | one-step mastery updates; "current knowledge state" | E1 |
| #45 (open) | `crossvalidate` silently predicts 0.5 for unknown parameters; collaborator offered a warning | D1 |
| #21, #27, #36, #38 (closed) | inverted or varying mastery estimates; answered as random restarts or degenerate parameters | D1, D2 |
| #39 (closed by #41) | seeded C++ fits not reproducible; leftover differences called "floating-point error" | B6 (that leftover is the unordered parallel sum) |
| #32 (closed, 2022) | C++ build fails on Windows even with MSVC; "we do not officially support … Windows" | A5, asked in #65 first |
| #26 (closed, 2021) | slow outside Colab; answer: check whether the C++ build is installed | Train A, C1 |
| #29 (closed), #50 (open) | `Roster` with multigs models fails | look at it with E1 (Roster internals), not before |
| #54 (open) | `random.randint(0, 1e8)`; fixed by merged #48 and #56 | comment that it can be closed |
| #37, #60 (closed) | "latest pyBKT not built", "new pip release" | precedent: release requests get answered |

**Upstream git history that bears on the plan** (full history, 393 commits):

| Commit | Date | What it tells us |
| --- | --- | --- |
| `cda06d3` initial commit | 2017-01 | variable-length stack arrays and `alloca.h` in the E-step: the 200k-answer segfault and the MSVC blocker date from here |
| `8501508`…`71ed5ad` "almost mac compat", "mac compat added?", "mac fix" | 2020-07 | OpenMP on macOS took seven commits |
| `c3e98e4` "optimize data helper/cv"; `dbac5cb` "Model abstraction implemented"; `c1806c4` "regex full match" | 2021-02 | the two-step sort (tied `order_id`), the `groupby` lengths (missing user ids) and the unescaped skill regex |
| `dba0ddb` "completed boost removal" | 2021-04-14 | introduced `PyLong_FromLong(*total_loglike)`; before it, the log-likelihood was returned as a double. The integer log-likelihood is a regression, and the fix restores the original behaviour |
| `9ea4a2c`, `66c05d8` memory-leak fixes; `1721ab7` "fixed disable parallel" | 2021-04 | earlier leak work; `1721ab7` introduced the sticky `omp_set_num_threads(1)` |
| `691f24f` → `df2038b` "trying wheel building" → "removed gh actions" | 2023-07-07 | a cibuildwheel attempt for Ubuntu, Windows and macOS was cut to Ubuntu, then deleted the same day |
| `d7806c6` (#51) | 2024-08 | Windows users run the pure-Python path with multiprocessing |

## 2. Decisions so far (owner)

- Upstream PRs to CAHLR, landing on the fork first. Users: batch researchers, very large datasets, users
  without a compiler, live tutors. **Correctness first.**
- Result-changing fixes: ask the maintainer, one issue per bug with a repro.
- **Topic issues, no umbrella**, posted in **staged waves** (section 3).
- After install and release: **correctness, then live updates**, then speed and opt-in features.
- **Single Python source tree:** propose early; build it only if welcomed.
- **Windows wheels:** in scope; ask in #65 first because of #32.
- **`data_helper.py` order:** bug fixes first, then the vectorized `convert_data`, which must keep the fixed
  behaviour.
- **Rust:** open a discussion issue now, framed as "SIMD and multithreading help, but they are riskier to
  build in C and C++"; build nothing upstream until the maintainer replies.
- Docs: five-line PR summaries, README sections for user-visible changes, notebooks or docs pages for
  opt-in features.
- Evidence for upstream readers goes on a clean, neutrally named fork branch (default name
  `evidence/bkt-research`; confirm with the owner before creating it).

## 3. Discussion plan: topic issues in three waves

Existing threads are reused where they fit; new issues are opened only for a decision the maintainer can
make on its own. Texts are in `ISSUE_DRAFTS.md`.

```mermaid
flowchart LR
  subgraph W1["Wave 1: now"]
    I1["#65 follow-up:<br/>NumPy 2 fix + harness PRs,<br/>README note, Windows question"]
    I2["New: Rust for SIMD and<br/>multithreading (discussion)"]
    I3["New: one Python source tree<br/>(proposal)"]
    I4["#54 comment:<br/>already fixed"]
    I1b["#65: release request<br/>after A1 + A2 merge"]
    I1 --> I1b
  end
  subgraph W2["Wave 2: after the release"]
    I5["New bug: integer<br/>log-likelihood"]
    I6["New bug: skill names<br/>with regex characters"]
    I7["New bug: missing<br/>user ids"]
    I8["New bug: tied order_id"]
    I9["New bug: multipair keys<br/>across pandas versions"]
    I10["#55 comment (links #57):<br/>live mastery update API"]
    I5 --> I6 --> I7 --> I8 --> I9
  end
  subgraph W3["Wave 3: after the fixes"]
    I11["New: faster, same results"]
    I12["New: fit diagnostics<br/>(links #45, #21, #27, #36, #38)"]
    I13["New: optional estimators<br/>(priors, SQUAREM, online EM)"]
    I14["New: very large datasets"]
  end
  W1 --> W2 --> W3
```

Why topic issues and waves: each risky proposal (Rust, single tree, each result change) can be declined
without stalling the rest; the maintainer sees a few threads at a time; and earlier merges build the
trust that later, larger proposals need. Bug issues go one at a time, each with its PR ready.

## 4. Work order

```mermaid
flowchart TD
  A1["A1 NumPy 2 fit<br/>(#65 item 2)"] --> A2["A2 regression harness"]
  A2 --> REL(["Release request in #65"])
  A1 --> REL
  A2 --> A4["A4 metadata + install docs"]
  A2 --> B1["B1 C++ safety<br/>(stack arrays, delete[], leak,<br/>sticky threads, dead alloca.h)"]
  B1 --> A5["A5 Windows wheels<br/>(if #65 says yes)"]
  A2 -.->|if welcomed| T1["T1 one Python source tree"]
  B1 --> B5["B5 float log-likelihood"]
  B5 --> B6["B6 deterministic C++ sums"]
  T1 -.-> B2
  A2 --> B2["B2 escape skill-name lists"]
  B2 --> B4["B4 missing user ids"]
  B4 --> B3["B3 stable order for ties"]
  B3 --> B7["B7 multipair keys"]
  A2 --> E1["E1 live update function<br/>(Roster uses it)"]
  B3 --> C3["C3 vectorized convert_data<br/>(keeps the fixes)"]
  B1 --> C2["C2 forward-only predict"]
  A1 --> C1["C1 vectorized pure-Python E-step"]
  B6 --> C4["C4 skip alpha, narrow dtypes"]
  C3 --> C7["C7 read only needed columns"]
  C3 --> C6["C6 sparse multigs, 64-bit indices"]
  C4 --> C6
  C3 --> C8["C8 convert once"]
  C8 --> C9["C9 restarts in one pass"]
  C9 --> C5["C5 parallel policy<br/>(after measurement)"]
  A2 --> D1["D1 fit diagnostics"]
  D1 --> D2["D2 Beta priors"]
  D1 --> D3["D3 SQUAREM"]
  C1 --> E3["E3 out-of-core EM"]
  C1 --> E4["E4 online EM (experimental)"]
  E3 --> E5["E5 prefix sharing"]
  C3 --> E2["E2 DuckDB loader"]
  classDef w1 fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
  classDef w2 fill:#dcfce7,stroke:#166534,color:#052e16
  classDef w3 fill:#fef3c7,stroke:#92400e,color:#3b1d02
  class A1,A2,A4,REL,B1 w1
  class A5,T1,B2,B3,B4,B5,B6,B7,E1 w2
  class C1,C2,C3,C4,C5,C6,C7,C8,C9,D1,D2,D3,E2,E3,E4,E5 w3
```

Blue is wave 1, green wave 2, amber wave 3. Dashed edges apply only if the maintainer welcomes the single
source tree. Several PRs edit the same files, so merge them in this order:

| File | PRs, in merge order |
| --- | --- |
| `fit/E_step.cpp` | B1 → B5 → B6 → C4 → C6 → C5 (E5 later) |
| `util/data_helper.py` (two copies until T1) | B2 → B4 → B3 → B7 → C3 → C6 / C7 |
| `models/Model.py` (two copies) | E1, D1 (additions) → B7 → C8 → C9 |
| `fit/EM_fit.py` (two copies that differ) | A1 → B5 → C1 → C4 → C9 → D3 |
| `tests/reference/*.json` | one result-changing PR at a time; its JSON diff is the evidence |

## 5. Task cards

Each card says what to build and how to know it's done. Line numbers are on upstream master `cc1682e`.
"Verify" commands assume the environments from section 8.

### Wave 1

**A1 · Pure-Python fit works on NumPy 2** · `[fix]` · branch `fix/em-fit-numpy-2` · issue #65 item 2
- Change: `source-py/pyBKT/fit/EM_fit.py:43` assigns a (1, 1) array into a scalar slot. Store a scalar,
  for example `float(np.squeeze(result['total_loglike']))`. Make the same change at
  `source-cpp/pyBKT/fit/EM_fit.py:25` so both copies stay alike (there the value is a Python int, see B5).
- Verify: pure Python on NumPy 2 fits (`PYTHONPATH=source-py python -c "…Model(seed=0).fit(data=df)…"`);
  the harness's 8 `FAILS_ON_NUMPY2` cases pass.
- Risk: none known; results unchanged where the code ran before.

**A2 · Regression harness** · `[infra]` · from `test/regression-harness` (`97a0eb1`), rebased on A1
- Delete three obsolete markers: `CPP_TEMPLATE_BUG` in `tests/test_backend_parity.py`, the #70 marker in
  `tests/test_parameter_recovery.py`, the #72 marker in `tests/test_one_attempt_students.py`; and
  `FAILS_ON_NUMPY2` in `tests/helpers.py` once A1 is in.
- Verify: all three configurations pass with no xfail/xpass (section 8).
- Follow-up (later, optional): run the harness inside cibuildwheel (`CIBW_TEST_COMMAND`) so every wheel
  platform is tested. Its tolerances (`rtol` 1e-7 outputs, 1e-9 EM parity) are untested off Linux.

**A4 · Metadata and install docs** · `[infra]` · branch `docs/install-and-metadata`
- `setup.py`: `python_requires=">=3.10"` (matches the wheel matrix), classifiers 3.10–3.14 (now 3.5–3.8).
- README: wheels are compiled; how to check (`pyBKT.version`, `import pyBKT.fit.E_step`);
  `PYBKT_REQUIRE_CPP=1` to fail loudly when building from source (default 0, as zpardos asked in #65);
  update "Installing Dependencies for Fast C++ Inferencing (Optional - for OS X and Linux)".
- Users on Python < 3.10 keep getting 1.4.3 from pip; say so in the PR.

**Release request** · comment on #65 once A1 and A2 are merged (text in `ISSUE_DRAFTS.md`). Key points:
tag without a `v` (like `1.4.3`); release notes must say that #70 and #72 change fitted values.

**B1 · C++ safety** · `[same]` · branch `fix/cpp-e-step-heap-scratch` · from `34e3d21`
- Port `34e3d21`: per-student scratch arrays move from the stack to `std::vector` (a student with 200,000
  answers segfaults today; OpenMP worker threads have smaller stacks); `delete` → `delete[]`; the leak;
  `parallel=False` no longer calls the process-wide `omp_set_num_threads(1)`.
- Also delete the dead `#include <alloca.h>` in `predict_onestep_states.cpp:5` and
  `synthetic_data_helper.cpp:13` (nothing calls `alloca`). That finishes the MSVC prerequisites in code.
- Move the forward-only `E_step.predict` entry point that `34e3d21` adds into C2 instead, so B1 is purely
  a safety change.
- Verify: `tests/test_long_sequences.py` (from `34e3d21`) passes; harness passes; outputs bit-identical to
  master on the harness fixtures; `g++ -std=c++17 -fsyntax-only -pedantic-errors -Wvla -Wno-write-strings`
  reports no errors (it reports five variable-length arrays on master).

**Issues to post in wave 1** (owner posts): #65 follow-up, Rust discussion, single-tree proposal, #54
comment. See `ISSUE_DRAFTS.md`.

### Wave 2

**A5 · Windows wheels** · `[infra]` · after B1 and a yes in #65
- `setup.py`: an MSVC branch (`/openmp`, `/O2`; no `-fPIC`, `-fopenmp`, `-w`). Check `long` uses
  (32-bit on Windows). Add `windows-latest` to `release.yml`; the wheel test imports `E_step`.
- MSVC supports OpenMP 2.0 only: `parallel for` needs a signed loop index (`predict_onestep_states.cpp:97`
  uses `int`, fine).
- Verify: the PR's own wheel build (`release.yml` runs on pull requests). Expect several CI rounds; nobody
  has a Windows compiler in this environment.

**T1 · One Python source tree** · `[infra]` · only if the single-tree proposal is welcomed
- Today `source-cpp/pyBKT` and `source-py/pyBKT` each hold the same 24 Python files: 17 identical, 7
  different (`EM_fit.py` by 230 lines, `predict_onestep.py` by 67, five others by 2–4). Every Python fix
  is made twice, and #72 was a drift between the copies.
- Target: one `pyBKT/` tree; the compiled `E_step`/`predict_onestep_states` modules optional at import,
  with the NumPy versions as the fallback; `pyBKT.version` still reports which is active; `setup.py`
  builds the extension when it can.
- Do it before B2–B4 if accepted; otherwise B2–B4 edit both copies.

**Bug PRs, one at a time, each with its issue** (repros in this directory; each changes results):

| PR | Bug | Repro | Fix | Results that change |
| --- | --- | --- | --- | --- |
| B5 | Compiled log-likelihood is an integer (`E_step.cpp:341`, since `dba0ddb`, 2021-04) | `integer_loglike.py`: whole-number trace ending −1508, −1507, −1507; EM stops after 15 iterations with `tol=0.005` | `PyFloat_FromDouble`; restores pre-2021 behaviour | compiled fits run longer (24 → 47 iterations on ASSISTments); `num_fits` choice |
| B2 | Skill-name lists joined into an unescaped regex (`data_helper.py:25`); after `fit`, `model.skills` is that list | `regex_skill_names.py`: default `'.*'` fits the skill but predicts all its rows at 0.5; a list skips it in `fit` | `re.escape` each name when `skills` is a list; a string stays a regex (documented API) | predictions and fits for skills whose names contain `+ ( ) * …` (2,339 ASSISTments rows) |
| B4 | Missing `user_id`: `groupby` drops them from `lengths` (`data_helper.py:164`) but the rows stay | `nan_user_ids.py`: fit ignores them; predict returns 1.12 and 0.0 (pure Python) or 0.0 (compiled) | drop with a warning, or raise (maintainer's choice) | only data with missing ids |
| B3 | Ties in `order_id` ordered by quicksort (`data_helper.py:99`), then a stable sort by user (`:117`) | `tied_order_ids.py`: same rows in two orders, parameters differ by 0.14 (compiled) and 0.63 (pure Python) | one stable sort by (user, order_id) that keeps file order for ties | data with ties (29.6% of ASSISTments rows) |
| B7 | multipair keys are array reprs (`data_helper.py:192`) | needs pandas 2 and 3: fit under 2, predict under 3, every pair "not fitted" | keys from values; read old keys when loading | multipair models across pandas versions |

**B6 · Deterministic C++ sums** · `[same]` up to the last bits · after B5 · refers to #39
- Threads add their counts in a `critical` section in arrival order (`E_step.cpp:306`). Store per-block
  partial counts and add them in block order after the parallel region.
- Verify: identical hashes over 20 runs and 1–16 threads (method: `rust/py/determinism.py`; C++ gave 1–3
  distinct results in 20 runs on ASSISTments, 6–7 on synthetic data).

**E1 · Live mastery updates** · `[opt-in]` · after the #55 comment
- `Roster.update` (2021) already updates one student per answer, but it swaps the shared model's prior to
  the student's state, runs the full predict path on the new answers, then restores the prior
  (`Roster.py:577–585`). #55 and #57 ask for a plain one-step function.
- Build a pure function (normalized two-state update: posterior from the answer, then the learn/forget
  transition), plus a vectorized form for many students; make `Roster` use it so it stops mutating the
  model. Keep `Roster`'s API.
- Evidence: `live_bench.py` (2.4 µs per answer in plain Python; matches `predict` to 1.6e-9 on 294k
  answers; the textbook `1 − P(correct)` form loses precision near 0 and 1).
- Verify: equals `predict`'s state predictions on the harness data; equals today's `Roster` outputs.
- Look at #50 (Roster with multigs) while in this code, as a separate PR.

### Wave 3 (cards are shorter; refresh numbers before posting)

| PR | What | Source | Key evidence | Notes |
| --- | --- | --- | --- | --- |
| C2 | forward-only predict | `fda90e7` + `E_step.predict` from `34e3d21` | 1M rows 60 → 14 ms; multigs × 50 templates 3.27 → 0.12 s, half the peak memory | `[same]` |
| C3 | vectorized `convert_data` | `37f57be`, re-applied after B2–B4/B3 | 5M rows/100 skills 31.6 → 2.0 s; multipair 100k 17.2 → 0.06 s | must keep the fixed behaviour; identical output across 46 input shapes |
| C1 | vectorized pure-Python E-step | `np_estep.py` (posterior-form backward pass) | ASSISTments fit 28.7 → 0.54 s (54x), matches C++ to 3.4e-15 | scaled-β form overflows on long sequences; use the posterior form |
| C4 | skip `alpha` during EM; int8 answers, int32 resources; alpha's labelled shape | ROADMAP | 16 B/answer per iteration not written | |
| C7 | read only needed columns | `load_bench.py` | 20M rows 60.8 s / 5.0 GB → 33.0 s / 2.6 GB | keep every column the model type needs |
| C8 | convert once (prepared data); stop sorting the caller's DataFrame in place | `lean_gap.py` | per-call copies cost ~40% of a fast E-step | `fit` reorders the caller's DataFrame today |
| C9 | all `num_fits` restarts in one pass; skills in parallel | ROADMAP | `num_fits=5` takes 4.3x one fit | keep random draws in today's order |
| C6 | sparse multigs, 64-bit indices | ROADMAP | 1.47 GB → ~2 MB (all ASSISTments skills merged) | |
| C5 | serial for small E-steps, parallel across skills | `e2e_pybkt.py`, `omp_overhead.py` | up to 6x under load; 7.9 µs/call idle | **measure first** on an idle machine, then with load |
| D1 | fit diagnostics (warnings) | `degeneracy_audit.py` | 9.1% implausible, 11.8% at 0/1, 74 of 110 skills' restarts disagree | addresses #45 and the #21/#27/#36/#38 pattern |
| D2 | Beta priors, `priors=` | `map_em.py`; Beck & Chang 2007 | stuck at 0/1: 15.3% → 0%; held-out ll slightly better | defaults need choosing |
| D3 | SQUAREM, `accelerate=` | `squarem.py`; Varadhan & Roland 2008 | 1.4–1.7x fewer E-steps; never worse in 48 runs | needs step cap and monotone guard |
| E2 | DuckDB loader extra | `duckdb_convert.py` | 20M rows read + convert 14.3 s vs 60.8 s | optional dependency |
| E3 | exact out-of-core EM | `check_smoothing.py`; Cappé 2011 | equals forward-backward to 2.5e-16 | |
| E4 | online EM (experimental) + `partial_fit` docs | `online_em.py`, `cappe_moulines.py`; Cappé & Moulines 2009 | one pass within 0.0016 of truth (batch 0.0013) | event-level variant not planned |
| E5 | prefix sharing | `dedup.py` | 2.75x less work (ASSISTments), ~25x (short sessions) | counted only, not built |

## 6. What the experiments add up to

The research covered loading, conversion, the E-step in three languages, EM itself, prediction and live
updates. Read together, the experiments point to a few themes; each theme shapes several plan items.

```mermaid
flowchart LR
  subgraph EXP["Experiments"]
    X1["load_bench, load_breakdown,<br/>duckdb_convert, conv_mem"]
    X2["Rust bkt_rs / bkt_lean,<br/>lean_gap, rayon_simd"]
    X3["e2e_pybkt, omp_overhead"]
    X4["np_estep, ab_estep,<br/>check_smoothing, live_bench"]
    X5["degeneracy_audit, map_em,<br/>squarem, squarem_tol"]
    X6["cappe_moulines, cm_real,<br/>online_em, cappe2011_events, dedup"]
    X7["golden checks, parity runs,<br/>repro scripts"]
  end
  subgraph TH["Themes"]
    T1["Load-bound vs<br/>compute-bound work"]
    T2["Parallelize across skills<br/>and restarts, not inside<br/>small E-steps"]
    T3["Numerical form matters"]
    T4["Reproducibility"]
    T5["Plausible parameters"]
    T6["One forward recursion<br/>serves many features"]
    T7["Independent implementations<br/>find bugs"]
  end
  subgraph PL["Plan items"]
    P1["C3 C6 C7 C8 E2"]
    P2["C5 C9 R"]
    P3["C1 E1 D3 E3"]
    P4["B3 B6 R"]
    P5["D1 D2"]
    P6["C2 E1 E3 E4 E5"]
    P7["A2 harness parity tests"]
  end
  X1 --> T1
  X2 --> T1
  X2 --> T2
  X3 --> T2
  X4 --> T3
  X5 --> T3
  X2 --> T4
  X7 --> T4
  X5 --> T5
  X4 --> T6
  X6 --> T6
  X2 --> T7
  X7 --> T7
  T1 --> P1
  T2 --> P2
  T3 --> P3
  T4 --> P4
  T5 --> P5
  T6 --> P6
  T7 --> P7
```

| Theme | What the experiments showed | Consequence for the plan |
| --- | --- | --- |
| Load-bound vs compute-bound work | Reading and converting 20M rows takes 60.8 s; one C++ E-step pass over them takes about 0.3–1.1 s (14–53 ns per answer). A default fit makes many passes (iterations × 5 restarts), so fits are compute-bound while `predict` and other one-pass work is load-bound. The dense multigs matrix (1.47 GB) and per-call copies (~40% of a fast Rust E-step) cost in both | one-pass work: cheaper reading and convert-once (C7, C8, E2); fits: fewer passes (C9, D3) and a faster kernel (C1, R); both: compact layouts (C4, C6) |
| Parallelize at the right level | OpenMP inside small E-steps cost milliseconds per call under load (7.9 µs idle, unconfirmed under controlled load); real data has many small skills; in Rust, rayon's work stealing and plain scoped threads performed alike on these workloads | parallel across skills and restarts (C9), a size threshold (C5, after measuring); work stealing should suit mixed performance/efficiency cores (R; not measured here) |
| Numerical form matters | the scaled-β backward pass overflowed (NaN) on a 3,585-answer student; the textbook live update loses precision; SQUAREM without its step cap jumped to 0/1 boundaries | posterior-form backward pass (C1, E3), normalized live update (E1), SQUAREM safeguards (D3) |
| Reproducibility | C++ parallel sums vary run to run (#39's leftover); tied `order_id` order depends on input order; Rust's fixed-order reduction is bit-identical across threads and ISAs | B6, B3; Rust design (R) |
| Plausible parameters | 9.1% of kept fits implausible, 11.8% at 0/1; restarts disagree in 74 of 110 skills; four closed issues show users confused by exactly this | warnings first (D1), priors as an option (D2) |
| One forward recursion serves many features | forward-only smoothing (Cappé 2011) gives exact E-step counts without a backward pass | the same φ/ρ recursion underlies predict (C2), live updates (E1), out-of-core EM (E3), online EM (E4) and prefix sharing (E5) |
| Independent implementations find bugs | the harness's C++ vs pure-Python parity tests found #70 and #72; the vectorized NumPy E-step confirmed #72's effect (parameters up to 0.085 apart); Rust parity found the mislabelled alpha shape; live updates vs `predict` found the regex and tie bugs | keep a reference implementation in the harness as an oracle (A2), and add parity tests whenever a backend is added |

What was tried and set aside, with reasons, is in `PLANNING.md` section 7 ("Not planned"): event-level
online EM (stable only at α = 0.8), a GPU backend (no benefit at typical sizes), a gradient-based fitter,
a parallel scan over time, swapping the binding layer, Numba/Cython, Polars as a requirement.

## 7. Rust: why SIMD and multithreading belong there

**The framing for the discussion issue:** SIMD and multithreading would help pyBKT a lot, but they are
riskier to build in C and C++ than in Rust. The C++ backend gets the safety and determinism fixes it needs
(B1, B5, B6) and no new SIMD or threading code.

**What SIMD and threads gave** (measured on a 4-vCPU VM, `rust/REPORT.md`):

| | ASSISTments (as_all) | synthetic 5M answers |
| --- | --- | --- |
| C++ serial / 4 OpenMP threads, ns per answer | 47.9 / 28.7 | 53.2 / 14.0 |
| Rust exact serial (bit-identical to C++) | 22.5 (2.1x) | 21.5 (2.5x) |
| Rust SIMD across students, 4 threads | 2.3 (12.7x vs C++ parallel) | 1.45 (9.7x vs C++ parallel; 37x vs serial) |
| 20-iteration fit in Rust, 4 threads, SIMD | 0.022 s (20x vs C++ serial fit) | 0.15 s (39x) |

**Why it is riskier in C/C++, from this codebase's own history:**

| Problem | Where | Status |
| --- | --- | --- |
| Per-student stack arrays: segfault at 200k answers, and MSVC can't compile them | since the 2017 initial commit | B1 fixes |
| `delete` on `new[]` memory; a leak; earlier leak fixes in 2021 | `E_step.cpp:34`, `9ea4a2c`, `66c05d8` | B1 fixes |
| `parallel=False` switches OpenMP off for the whole process | `1721ab7` (2021) | B1 fixes |
| Unordered parallel sums: seeded fits not reproducible | #39; 1–3 distinct results in 20 runs | B6 fixes |
| Loop variables not reset: every template used the first one's parameters | #70 (found by the harness's C++ vs Python parity test) | fixed by #71 |
| OpenMP portability: seven commits for macOS (2020); Windows never built (#32); a 2023 wheel attempt for Windows and macOS was removed the same day | history | A5 asks first |

Adding SIMD in C++ means intrinsics per instruction set or compiler-specific multiversioning, and wheels
built for the manylinux baseline (SSE2) unless dispatch is added by hand. Adding threads means more shared
mutable state that the compiler does not check.

**What Rust gives instead** (from the prototypes):
- `#![forbid(unsafe_code)]` in our crates: the compiler rejects data races and out-of-bounds writes in our
  code.
- Safe SIMD with runtime dispatch (`fearless_simd`): the same binary uses SSE2, AVX2 or AVX-512, and
  results are bit-identical across all of them.
- A fixed-order reduction: bit-identical results for 1–16 threads and across runs.
- Work stealing (rayon) or scoped threads with an atomic chunk counter; both handle uneven student lengths
  and mixed performance/efficiency cores.

**Costs to state plainly:**
- Dependencies still contain `unsafe`: 2,457 lines in `bkt_lean`'s linked crates (pyo3, pyo3-ffi, libc,
  which any Rust extension needs); 2,560 with `fearless_simd`; 3,704 for the numpy + rayon variant. No
  RustSec advisory affects the resolved versions (checked 2026-10-08).
- A Rust toolchain is needed to build from source (wheels hide it from users). `bkt_lean` needs Python
  ≥ 3.11 for the limited-API buffer protocol; 3.10 would need per-version wheels.
- A second backend to keep in parity with C++ and NumPy; the harness's parity tests are the guard.
- Maintainer familiarity with Rust.

**Proposal for the issue:** an optional package (`pybkt-rs`) that pyBKT uses when installed, with the same
API and the harness's parity tests; C++ stays the default compiled backend. Questions for the maintainer:
interest at all; separate package or in-tree optional backend; who reviews Rust changes.

## 8. Setup and verification

```bash
git clone https://github.com/ChristianMurphy/pyBKT && cd pyBKT
git remote add cahlr https://github.com/CAHLR/pyBKT && git fetch cahlr master
git fetch origin test/regression-harness claude/pybkt-performance-research-xqv9jv claude/pybkt-experiments-xqv9jv

# NumPy 2 (compiled build and pure Python) and NumPy 1 (pure Python); NumPy < 2 needs Python <= 3.12
python3.13 -m venv ~/np2 && ~/np2/bin/pip install numpy setuptools pandas pytest scikit-learn requests
python3.12 -m venv ~/np1 && ~/np1/bin/pip install "numpy<2" pandas pytest scikit-learn requests

# merge each fork branch onto upstream master in memory and run its tests three ways
PY2=~/np2/bin/python PY1=~/np1/bin/python experiments/upstream_sync_check.sh
```

For a PR branch, run the same three configurations on its tree:

```bash
~/np2/bin/pip install --no-build-isolation --no-deps --target /tmp/site .   # builds the C++ extension
~/np2/bin/python -c "import sys; sys.path.insert(0, '/tmp/site'); import pyBKT.fit.E_step"   # must succeed
PYTHONPATH=/tmp/site   ~/np2/bin/python -m pytest -q tests          # compiled, NumPy 2
PYTHONPATH=source-py   ~/np2/bin/python -m pytest -q tests          # pure Python, NumPy 2
PYTHONPATH=source-py   ~/np1/bin/python -m pytest -q tests          # pure Python, NumPy 1
```

C++ changes also get the strict syntax check from B1 and a bit-identity check against master on the
harness fixtures. Research scripts need the example data: `experiments/fetch_data.sh` (ASSISTments and
Cognitive Tutor samples); their paths are listed in `README.md` ("Reproduce from scratch").

## 9. Open items and leads

- **Not yet asked:** the evidence branch name (default `evidence/bkt-research`) and its contents
  (`PLANNING.md` section 7, "Evidence branch"); whether to run the harness on every wheel platform.
- **Lead, unverified:** on the tie-heavy data in `tied_order_ids.py`, compiled and pure-Python fits with the
  same seed differ a lot (prior 0.787 vs 0.604, learn 0.373 vs 0.999). Likely the integer stopping rule
  (B5) ending the compiled EM early, plus several optima. Re-run after B5 before blaming anything else.
- **Lead:** `fit` sorts the caller's DataFrame in place (checked on master); fold into C8 or report it.
- **Lead:** `Roster` mutates the shared model's prior during each update (E1 removes this).
- **Waiting on the maintainer:** #65 (release, Windows), the Rust discussion, the single-tree proposal.
