# Upstream issue and comment drafts (CAHLR/pyBKT), by wave

Status: drafts, 2026-10-09. Nothing here has been posted. The owner posts each item, or approves it,
when its wave starts (`HANDOFF.md` section 3). `<PR>` marks numbers to fill in.

Layout, as decided: topic issues with no umbrella; existing threads reused (#65 install, #55/#57 live
updates, #54 housekeeping); one issue per result-changing bug, each posted with its PR ready.

@-mentions (fork owners credited in item 1) notify those people when posted; drop the `@` to credit
without a notification.

Numbers come from this directory (`README.md`, `rust/REPORT.md`, the repro scripts) and
`benchmarks/ROADMAP.md` on the phase 1 branch. Re-check any number that a later PR changes.

---

## Wave 1 (now)

### 1. Comment on #65, when the NumPy 2 fix and the test suite are opened

> Thanks for merging #66 to #69, and the fixes for #70 and #72.
>
> Item 2 is still open: the pure-Python fit fails on NumPy 2 at `EM_fit.py:43`. `<PR>` fixes it with one
> line in each `EM_fit.py`. The same fix appears in forks by @germanprogrammersberlin, @gengyabc and
> @flixstn, so others have been hitting this too.
>
> `<PR>` adds a test suite and a Tests workflow: the compiled build on Python 3.10 to 3.14, and the
> pure-Python build on NumPy 1 and 2. It compares fitted parameters and predictions with stored
> reference values, so a later change that moves results shows up as a diff in review.
>
> `<PR>` adds the README note on `PYBKT_REQUIRE_CPP` you asked for, and updates the install section now
> that wheels are compiled.
>
> **Windows:** would you like compiled Windows wheels? In #32 the answer was that Windows isn't officially
> supported, so I'd like to ask before sending anything. What stops MSVC today is small: the E-step keeps
> per-student arrays on the stack as variable-length arrays (MSVC rejects them, and they also crash the
> process for a student with 200,000 answers), two files include `alloca.h` without using it, and
> `setup.py` passes GCC flags on every platform except macOS. A C++ fix I'm preparing replaces the stack
> arrays; after it, Windows needs an MSVC branch in `setup.py` (`/openmp`) and `windows-latest` in the
> wheel matrix. The release workflow builds wheels on every PR, so the PR would show whether it works.

### 2. Comment on #65, once both PRs are merged: release request

> With `<PR>` and `<PR>` merged, master installs compiled wheels on Linux (manylinux and musllinux,
> x86_64 and aarch64) and macOS (arm64 and x86_64) for Python 3.10 to 3.14, fits on NumPy 2, and imports
> with scikit-learn 1.8 and later. Would you be able to make a release? Two notes:
>
> 1. **Please tag it the way 1.4.3 was tagged, without a `v`** (for example `1.5.0`). Two workflows can
>    publish: `release.yml` runs when a GitHub release is published and uploads the wheels and the sdist;
>    `publish.yml` runs on tags starting with `v` and uploads only an sdist. A `v` tag would run both, and
>    PyPI would likely reject the second sdist upload. If you'd like, I can send a PR that leaves one
>    path.
> 2. **Some fitted values change**, so the release notes should say so:
>    - compiled fits with `multigs` or `multilearn` now use each template's own parameters (#70);
>    - pure-Python fits now count students with a single answer (#72).
>
>    Because results change, a minor version (1.5.0) may suit better than 1.4.4, but that's your call.

### 3. New issue: SIMD and multithreading for the E-step (discussion)

**Title:** Discussion: SIMD and multithreading for fitting, and whether Rust is the safer place to build them

> I'd like your view before building anything here.
>
> **What could be gained.** Fitting spends its time in the E-step's forward-backward pass. Two things
> speed it up a lot: processing several students at once with SIMD, and splitting students across threads
> with a reduction in a fixed order. I prototyped both. On a 4-core VM, against today's C++ E-step:
>
> | | ASSISTments 2009 | synthetic, 5M answers |
> | --- | --- | --- |
> | same algorithm, one thread | 2.1x faster, bit-identical results | 2.5x |
> | SIMD across students, 4 threads | 12.7x faster than C++ with 4 OpenMP threads | 9.7x faster than C++ with 4 threads (37x vs one thread) |
> | 20 EM iterations, end to end | 20x faster than the C++ serial fit | 39x |
>
> Results are bit-identical for 1 to 16 threads and across SSE2, AVX2 and AVX-512.
>
> **Why not in C++.** Both are doable in C++, but they add the kind of code that has been hardest to get
> right in pyBKT's C++ so far:
> - per-student arrays on the stack (a crash at 200,000 answers; MSVC can't compile them), `delete` on
>   `new[]` memory, and leaks (fixed in 2021, and one more now);
> - `parallel=False` switching OpenMP off for the whole process, and parallel sums added in arrival
>   order, which is why seeded fits still differ in the last bits (#39);
> - OpenMP portability: several rounds for macOS in 2020, no Windows build (#32);
> - fork-join threading under load: when other processes keep the cores busy, every OpenMP call waits about
>   8 ms (one scheduler time slice) whatever its size, because the call ends only when every thread has
>   finished. A 1,800-answer call takes 8.0 ms instead of 0.15 ms on one thread. Switching the loop to
>   dynamic scheduling gave the same 8 ms.
>
> SIMD in C++ also means intrinsics per instruction set, or compiler-specific multiversioning, and
> wheels built for the SSE2 baseline unless dispatch is written by hand.
>
> **Why Rust is the safer place.** The prototype forbids `unsafe` in its own code, so the compiler rejects
> data races and out-of-bounds writes there. SIMD goes through a safe library that picks SSE2, AVX2 or
> AVX-512 at run time. The reduction order is fixed, which gives the bit-identical results above. Its
> work-stealing thread pool (rayon) keeps going when another process takes a core: on the same 1,800-answer
> call under the same load it took 56 µs, where OpenMP took 8 ms.
>
> **Costs, plainly.** The Rust dependencies still contain `unsafe` code (mostly pyo3 and libc, which any
> Rust extension for CPython needs; no security advisory affects the versions used). Building from source
> needs a Rust toolchain, though wheels hide that from users. It would be a second compiled backend to
> keep in step with C++ and NumPy; the test suite's backend-parity tests are meant for exactly that.
>
> **Proposal.** An optional package (for example `pybkt-rs`) that pyBKT uses when it's installed, with the
> same API and the same parity tests. It would combine the leaner of the two prototypes (only pyo3 as a
> binding dependency, no copies of the input arrays) with rayon for work stealing. The C++ backend stays
> the default and only gets safety and determinism fixes, no new SIMD or threading code.
>
> ```mermaid
> flowchart LR
>   API["pyBKT API<br/>Model.fit / predict"] --> SEL{"which E-step<br/>is installed?"}
>   SEL -->|pybkt-rs installed| RS["Rust E-step<br/>SIMD + work stealing"]
>   SEL -->|compiled wheel| CPP["C++ E-step<br/>safety and determinism fixes only"]
>   SEL -->|no compiled module| NP["NumPy E-step"]
>   RS --> T["one test suite:<br/>backend parity"]
>   CPP --> T
>   NP --> T
> ```
>
> Questions:
> 1. Is this something you'd want for pyBKT at all?
> 2. If so, a separate package, or an optional backend inside this repository?
> 3. Who should review Rust changes?

### 4. New issue: one Python source tree (proposal)

**Title:** Proposal: one Python source tree, with the compiled E-step optional at import

> `source-cpp/pyBKT` and `source-py/pyBKT` each hold the same 24 Python files. 17 are identical; 7 differ,
> mostly `EM_fit.py` (230 lines) and `predict_onestep.py` (67 lines). `setup.py` installs one tree or the
> other depending on whether the C++ build succeeds.
>
> Two consequences: every Python fix has to be made twice and kept identical by hand, and the copies drift.
> #72 was a drift of that kind: the pure-Python E-step skipped evidence the C++ one counted.
>
> I noticed the `nopy` branch (July 2023) went further and removed the pure-Python tree altogether. Two ways
> forward, and I'm happy with either:
>
> - **(a) One tree with a NumPy fallback.** One `pyBKT/` tree. The compiled modules stay optional; when
>   they're missing, pyBKT imports the NumPy versions instead, and `pyBKT.version` still says which one is
>   active. `setup.py` keeps building the extension when it can. Results don't change; the test suite runs
>   both paths. A vectorized NumPy E-step I'd send separately makes the fallback 20–24x faster than today's.
> - **(b) C++ only, as in `nopy`**, once wheels cover Windows too. The smallest codebase, but users on a
>   platform without a wheel, or whose build fails, would have nothing to fall back on.
>
> I'd lean towards (a) for that reason, but it's your call. Diagram for (a):
>
> ```mermaid
> flowchart LR
>   subgraph Today
>     S1["setup.py"] -->|C++ build works| A1["installs source-cpp/pyBKT:<br/>24 .py files + extension"]
>     S1 -->|C++ build fails| B1["installs source-py/pyBKT:<br/>24 .py files, 7 of them different"]
>   end
>   subgraph Proposed
>     S2["setup.py"] --> C2["installs pyBKT/:<br/>one copy of the .py files"]
>     C2 --> D2{"compiled E-step<br/>importable?"}
>     D2 -->|yes| E2["C++ path"]
>     D2 -->|no| F2["NumPy path"]
>   end
> ```
>
> Several fixes I'd like to send next edit `data_helper.py` and `Model.py`, so doing this first would
> halve those PRs. Would you accept a PR for (a), prefer (b), or rather keep the two trees?

### 5. Comment on #54

> This was fixed by #48 and #56 (`random.randint` now gets an int), so I think this issue can be closed.

---

## Wave 2 (after the release; one bug at a time, each with its PR)

Each bug issue follows the same shape: what happens, a short repro, the cause, the proposed fix, which
results change, and the question for the maintainer. The repro scripts in this directory are the
longer versions.

### 6. Bug: the compiled E-step returns the log-likelihood as an integer

> **What happens.** The compiled E-step returns the total log-likelihood as a whole number. EM's stopping
> rule compares consecutive values, so it compares integers: it stops as soon as two iterations round to
> the same number, and `num_fits` picks the first of several restarts whose log-likelihoods round alike.
> On ASSISTments 2009 (all skills merged), EM stops after 24 iterations where comparing floats takes 47.
>
> **Repro** (compiled build):
> ```python
> import numpy as np, pandas as pd
> from pyBKT.fit import EM_fit
> from pyBKT.generate import random_model_uni
> from pyBKT.util import data_helper
> rng = np.random.default_rng(2)
> rows = [dict(user_id=f"u{u}", skill_name="s", order_id=t, correct=int(rng.random() < 0.3 + 0.08 * t))
>         for u in range(300) for t in range(8)]
> data = data_helper.convert_data(pd.DataFrame(rows), "s")["s"]
> np.random.seed(0)   # random_model_uni uses NumPy's global generator
> _, trace = EM_fit.EM_fit(random_model_uni.random_model_uni(1, 1), data, tol=0.005, maxiter=100, parallel=False)
> print(trace[trace != 0])   # [-1627. -1538. ... -1508. -1507. -1507.]: whole numbers; stops after 15 iterations
> ```
>
> **Cause.** `E_step.cpp:341` returns `PyLong_FromLong(*total_loglike)`. Before the Boost removal in April
> 2021 (`dba0ddb`), the same value was returned as a double.
>
> **Fix.** Return `PyFloat_FromDouble`, which restores the earlier behaviour. `<PR>`
>
> **What changes.** Compiled fits run until the log-likelihood really stops changing, so fitted
> parameters move for many skills, and `num_fits` may pick a different restart. The test suite's
> reference values for the compiled build change accordingly; the PR shows the diff.
>
> Would you like this fixed, or kept behind an option for people reproducing earlier results?

### 7. Bug: skills whose names contain regex characters are never predicted

> **What happens.** With the default `skills='.*'`, a skill such as "Order of Operations +,-,/,* ()
> positive reals" is fitted, but `predict` leaves every one of its rows at the 0.5 placeholder (2,339 rows
> in ASSISTments 2009). Passing the names as a list skips the skill in `fit` without any message.
>
> **Repro:**
> ```python
> import numpy as np, pandas as pd
> from pyBKT.models import Model
> rng = np.random.default_rng(0)
> df = pd.DataFrame([dict(user_id=f"u{u}", skill_name=s, order_id=t, correct=int(rng.random() < 0.6))
>                    for s in ["plain", "a+b (c)"] for u in range(30) for t in range(6)])
> m = Model(seed=0, num_fits=1); m.fit(data=df)
> print((m.predict(data=df).groupby("skill_name").correct_predictions.apply(lambda s: (s == 0.5).sum())))
> # a+b (c): 180 of 180 rows at 0.5;  plain: 0
> ```
>
> **Cause.** `convert_data` treats a string as a regex, which is the documented behaviour, but it also
> joins a list of names with `|` without escaping them (`data_helper.py:25`). After `fit`, `model.skills`
> holds the list of fitted names, so `predict` builds an unescaped pattern from them.
>
> **Fix.** Escape each name when `skills` is a list; a string stays a regex. `<PR>`
>
> **What changes.** Fits and predictions for skills whose names contain `+ ( ) * ? [ ] .` and similar.
> Nothing else.

### 8. Bug: rows with a missing user id are dropped from the fit and get invalid predictions

> **What happens.** Rows whose `user_id` is missing are silently left out of the fit, and `predict` returns
> values for them that aren't probabilities: 1.12 and 0.0 with the pure-Python build, 0.0 with the compiled
> build.
>
> **Repro:**
> ```python
> import numpy as np, pandas as pd
> from pyBKT.models import Model
> df = pd.DataFrame({"user_id": ["a", "a", np.nan, "b", "b", np.nan, "b"], "skill_name": ["s"] * 7,
>                    "order_id": [1, 2, 3, 1, 2, 9, 3], "correct": [1, 0, 1, 1, 1, 0, 0]})
> m = Model(seed=0, num_fits=1); m.fit(data=df.copy())
> print(m.predict(data=df.copy())[["user_id", "correct_predictions"]])
> ```
>
> **Cause.** `convert_data` counts each student's answers with `groupby(user_id)` (`data_helper.py:164`),
> which drops missing ids, but the rows themselves stay at the end of the data.
>
> **Fix options.** (a) drop those rows with a warning, and return NaN predictions for them; (b) raise an
> error naming the column. Which would you prefer? `<PR>`
>
> **What changes.** Only data with missing user ids.

### 9. Bug: answers that share an `order_id` are ordered differently depending on input order

> **What happens.** When a student has several answers with the same `order_id` (common for multi-step
> problems; 29.6% of ASSISTments 2009 rows are in such groups), their order inside the sequence depends
> on the order of the input rows. The same data in a different row order gives different parameters.
>
> **Repro:**
> ```python
> import numpy as np, pandas as pd
> from pyBKT.models import Model
> rng = np.random.default_rng(1)
> rows = []
> for u in range(40):
>     for t in range(8):
>         rows.append(dict(user_id=f"u{u}", skill_name="s", order_id=t, correct=int(rng.random() < 0.7)))
>         rows.append(dict(user_id=f"u{u}", skill_name="s", order_id=t, correct=int(rng.random() < 0.4)))
> df = pd.DataFrame(rows)
> for frame in (df, df.iloc[::-1].reset_index(drop=True)):
>     m = Model(seed=0, num_fits=1); m.fit(data=frame.copy()); print(m.params().values.ravel())
> # compiled: parameters differ by up to 0.14 between the two row orders (pure Python: 0.63)
> ```
>
> **Cause.** `convert_data` sorts by `order_id` with pandas' default quicksort (`data_helper.py:99`), which
> isn't stable, then by user with a stable mergesort (`:117`).
>
> **Fix.** One stable sort by (user, `order_id`) that keeps the input order for ties. `<PR>`
>
> **What changes.** Fits and predictions for data with tied `order_id`s; for the same input file, the result
> is then the same every time.

### 10. Bug: multipair models fitted under pandas 2 don't work under pandas 3

> **What happens.** A multipair model fitted under pandas 2 reports every pair as not fitted when it
> predicts under pandas 3.
>
> **Cause.** Pair keys are built from the printed form of one-row arrays (`data_helper.py:192`,
> `str(df3[i:i+1][col].values)`). Under pandas 3 the values are a pandas string array, whose printed form
> differs.
>
> **Fix.** Build keys from the values themselves, and keep reading the old key format when a saved model
> is loaded. `<PR>`
>
> **What changes.** Only how multipair keys are stored; fitted values stay the same.
>
> (Repro needs pandas 2 and 3 side by side; the PR includes a test with stored old-format keys.)

### 11. Comment on #55 (links #57): one-step mastery updates

> #57 asks for the same thing, so I'll answer both here.
>
> `Roster` already updates a student one answer at a time, but each update swaps the model's prior to the
> student's current state, runs the full prediction path on the new answer, then restores the prior.
> That works, but it isn't the plain function these issues ask for, and it changes the shared model while
> it runs.
>
> I have a prototype of the plain version: given a skill's fitted parameters, the current mastery
> probability and the new answer, it returns the updated mastery in constant time (2.4 µs per answer in
> plain Python), with a batch form for many students. It matches `predict`'s state predictions to 1.6e-9
> on 294k answers. It uses a normalized form of the update that stays accurate near 0 and 1.
>
> ```mermaid
> flowchart LR
>   subgraph Today["Roster.update today"]
>     a1["set the shared model's prior<br/>to the student's mastery"] --> a2["build a data dict,<br/>run the full predict path"] --> a3["restore the prior"]
>   end
>   subgraph Proposed["Proposed"]
>     b2["Roster.update"] --> b1["update(mastery, correct, params):<br/>one exact step, no shared state"]
>   end
> ```
>
> Proposal: add the function, and have `Roster` use it, so `Roster` gets faster and stops changing the
> model. `Roster`'s API stays the same.
>
> Any preference for the name and signature? For example `model.update(skill, mastery, correct)`
> returning the new mastery.

---

### 11a. PR description for B8 (no issue): one thread for small E-step calls

> **What changes for users:** fits on machines where other work keeps the cores busy (shared servers,
> notebooks, parallel jobs) no longer pay about 8 ms per E-step call; idle machines keep OpenMP's speedup
> on large skills. **Class:** [same] (last-bit differences between the serial and parallel sums).
> **Evidence:** with 4 busy processes on 4 cores, a 1,800-answer call took 8.0 ms with OpenMP and 0.15 ms
> on one thread; 22,900 answers 8.0 ms vs 1.3 ms; idle, OpenMP is 1.7–2.9x faster from about 1,800 answers.
> Dynamic scheduling gave the same 8 ms. **Research basis:** none; measurement. **Verify:**
> `omp_controlled.sh` idle and with background load, before and after.

### 11b. Class-value cluster (after reproducing; one comment per cause)

Outline only, because nothing is reproduced yet. Start from the maintainers' unmerged `noerr` branch
(December 2022), which already lets unseen classes fall back instead of raising, sizes the multigs matrix
from the fitted classes (the likely cause of #47), and clips invalid predictions in `evaluate` with a
warning. The PR would rebase it with credit, add repros and tests, and fix one detail (an unseen multigs
class maps to index 1, a real template). Five reports involve multigs/multilearn class values
after `fit`: `Roster` from a multigs model (#29 closed, #50, #52; a traceback in #50 points at
`Roster.process_data` passing `True` instead of class names), `evaluate` raising `IndexError` on
sequence-level splits with multigs (#47), and `crossvalidate` predicting 0.5 where `predict` raises for
unseen multilearn classes (#45). For each cause found: a short repro on public data, the fix PR, and a
comment on the matching issue (on #50 for the `Roster` cause, mentioning #52).

## Wave 3 (outlines; refresh numbers before posting)

### 12. New issue: faster fitting and prediction, same results
List the [same] PRs with one measured line each: forward-only predict (1M rows 60 → 14 ms; multigs with
50 templates 3.27 → 0.12 s, half the peak memory); vectorized `convert_data` (5M rows and 100 skills
31.6 → 2.0 s; multipair 100k rows 17.2 → 0.06 s); vectorized pure-Python E-step (20–24x faster than today's default
parallel pure-Python fit, 32–40x than serial; matches C++ to 3.4e-15; no process pool, which has broken on
Windows before: #11, #51); reading only the needed columns (20M rows 60.8 s / 5.0 GB → 33.0 s / 2.6 GB); converting once
and not reordering the caller's DataFrame; running `num_fits` restarts in one pass (default takes 4.3x
one fit); a compact multigs layout (1.47 GB → ~2 MB) and 64-bit indices; and running skills and restarts in
parallel instead of inside each small E-step (the small-call OpenMP default ships earlier, in wave 2). Ask
which they'd like first.

### 13. New issue: warn about implausible fits
Link #45, #21, #27, #36, #38. Numbers from `degeneracy_audit.py`: 9.1% of kept fits have guess or slip
above 0.5, 11.8% are stuck at 0 or 1, restarts disagree by more than 0.1 in 74 of 110 skills. Propose
warnings only (no result change): implausible parameters, boundary values, restart disagreement, and
predictions that fall back to 0.5. Cite Pardos & Heffernan 2010 and Beck & Chang 2007.

### 14. New issue: optional estimators (priors, SQUAREM, online EM)
Opt-in only. Beta priors (Beck & Chang 2007): fits stuck at 0/1 go from 15.3% to 0% on 48 skills, held-out
log-likelihood slightly better on average. SQUAREM (Varadhan & Roland 2008): 1.4–1.7x fewer EM steps,
never a worse fit in 48 runs; it also needed fewer passes than L-BFGS, Nelder–Mead or differential evolution
on 8 skills. Global search found higher-likelihood but implausible optima (guess > 0.5) on 4 of 8 skills,
which is the case for priors rather than a stronger optimizer. Online EM (Cappé & Moulines 2009), experimental: one pass within 0.0016 of
the true parameters (batch 0.0013), and what today's `partial_fit` does. Ask which, if any, belong in
pyBKT and where docs should live (README section, notebook).

### 15. New issue: very large datasets
Exact fitting from data read in chunks (forward-only smoothing, Cappé 2011; equals forward-backward to
2.5e-16), an optional DuckDB loader (20M rows read and converted in 14.3 s vs 60.8 s), a compact on-disk
format (reload 20M rows in 0.03 s), and sharing work across students with the same answer prefix (2.75x
less work on ASSISTments). Ask whether large-data support is in scope.
