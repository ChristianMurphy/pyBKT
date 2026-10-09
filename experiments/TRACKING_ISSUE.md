# Draft texts for upstream (CAHLR/pyBKT), for the owner to review and post

Status: draft, 2026-10-09. Nothing here has been posted. Numbers come from this directory's `README.md`,
`rust/REPORT.md`, `benchmarks/ROADMAP.md` on the phase 1 branch, and `nan_user_ids.py`.

Notes for the owner (not part of the texts):
- Part 1 continues #65, which already has the maintainer's approval for wheels and optional scikit-learn.
  Part 2 is the new tracking issue and covers everything else. See PLANNING.md section 9, question 1.
- `<PR>` marks PR numbers to fill in once the PRs exist.
- Part 2 links to this fork branch for evidence (PLANNING.md question 3). Drop the link to keep the issue
  self-contained; the numbers are inline either way.
- #50 (Roster with multigs) is not included; it hasn't been reproduced (PLANNING.md question 2).

---

## Part 1a. Comment on #65, when the NumPy 2 fix and the test suite are opened

> Thanks for merging #66 to #69, and the fixes for #70 and #72.
>
> Item 2 is still open: the pure-Python fit fails on NumPy 2 at `EM_fit.py:43`. `<PR>` fixes it with one
> line in each `EM_fit.py`.
>
> `<PR>` adds a test suite and a Tests workflow: the compiled build on Python 3.10 to 3.14, and the
> pure-Python build on NumPy 1 and 2. It compares fitted parameters and predictions with stored
> reference values, so a later change that moves results shows up as a diff in review.
>
> I'll add the README note on `PYBKT_REQUIRE_CPP` you asked for in `<PR>`.

## Part 1b. Comment on #65, once both are merged: release request

> With `<PR>` and `<PR>` merged, master installs compiled wheels on Linux (manylinux and musllinux,
> x86_64 and aarch64) and macOS (arm64 and x86_64) for Python 3.10 to 3.14, fits on NumPy 2, and imports
> with scikit-learn 1.8 and later. Would you be able to make a release? Two notes:
>
> 1. **Please tag it the way 1.4.3 was tagged, without a `v`** (for example `1.5.0`). Two workflows can
>    publish: `release.yml` runs when a GitHub release is published and uploads the wheels and the sdist;
>    `publish.yml` runs on tags starting with `v` and uploads only an sdist. A `v` tag would run both, and
>    PyPI would likely reject the second sdist upload.
> 2. **Some fitted values change**, so the release notes should say so:
>    - compiled fits with `multigs` or `multilearn` now use each template's own parameters (#70);
>    - pure-Python fits now count students with a single answer (#72).
>
>    Because results change, a minor version (1.5.0) may suit better than 1.4.4, but that's your call.
>
> Windows wheels will follow in a separate PR.

---

## Part 2. New issue

**Title:** Proposals: correct results, faster fitting, and live mastery updates (tracking)

> Following #65, I've been studying how pyBKT fits and predicts: where time and memory go, where results
> can come out wrong, and which methods from the BKT and EM literature suit pyBKT. #64 to #73 were the
> first results. This issue lists the rest as small PRs, grouped by what they change for users, so you
> can choose what you want upstream. Each PR would change one thing, include tests, and say whether
> fitted values change.
>
> Measurements are on ASSISTments 2009 and synthetic data, on a 4-core Linux VM. Evidence and the
> commands to reproduce it:
> https://github.com/ChristianMurphy/pyBKT/tree/claude/pybkt-experiments-xqv9jv/experiments
>
> ### 1. Bugs whose fix changes results (your call on each)
>
> | Bug | Effect | Proposed fix |
> | --- | --- | --- |
> | Skill names are used as a regex without escaping | a skill such as "Order of Operations +,-,/,* () positive reals" never matches itself, so `predict` leaves its 2,339 ASSISTments rows at 0.5 | escape the names |
> | Ties in `order_id` are ordered by an unstable sort | 29.6% of ASSISTments rows are in (user, skill) groups with tied `order_id`s, so their order, and the fit, depends on input order | stable sort; ties keep file order |
> | Rows with a missing `user_id` | the fit drops them silently; `predict` returns 1.12 and 0.0 for them (pure Python) or 0.0 (compiled) | drop them with a warning, or raise |
> | The compiled E-step returns the log-likelihood as an integer | the `tol` stop compares whole numbers (24 EM iterations instead of 47 on ASSISTments), and `num_fits` ties go to the first fit | return a float |
> | `multipair` keys are built from array reprs | a model fitted under pandas 2 reports every pair as "not fitted" under pandas 3 | keys that don't depend on pandas, still reading old ones |
>
> For each: fix it, fix it behind an option, or leave it as it is?
>
> ### 2. Same results, safer and faster
>
> - **C++ E-step safety:** a stack array per student crashes on a student with 400k answers; passing
>   `parallel=False` once turns OpenMP off for the rest of the process; `delete` is used on `new[]` memory.
> - **Reproducible parallel fits:** threads add their counts in arrival order, so compiled parallel fits
>   differ in the last bits from run to run. Fixed blocks summed in a fixed order remove that.
> - **Predict from the forward pass**, without a templates × rows matrix: 1M rows 60 ms to 14 ms;
>   multigs with 50 templates 3.27 s to 0.12 s, with half the peak memory.
> - **Vectorized `convert_data`:** 5M rows and 100 skills 31.6 s to 2.0 s; multipair, 100k rows
>   17.2 s to 0.06 s. Output is identical across 46 input shapes.
> - **Vectorized pure-Python E-step** (NumPy, no new dependency): ASSISTments fit 28.7 s to 0.54 s
>   (54x), matching the C++ fit to 3.4e-15. This matters most where no wheel exists.
> - **Read only the needed columns** when given a file: reading and converting 20M rows goes from 60.8 s
>   and 5.0 GB to 33.0 s and 2.6 GB.
> - **Fit restarts together:** with the default `num_fits=5`, a fit takes 4.3x as long as one restart,
>   because each restart is a separate pass over the data. Running the restarts in one pass, and skills
>   in parallel, can keep every restart's result unchanged.
> - **Convert once:** `fit`, `predict` and `evaluate` each convert the DataFrame again (and sort the
>   caller's DataFrame in place). A prepared-data object could convert once and leave the input alone.
> - **Smaller multigs input:** one template index per answer instead of a templates × answers matrix
>   (1.47 GB to about 2 MB for all ASSISTments skills merged), and 64-bit indices for more than 2^31
>   answers in a skill.
>
> ### 3. Live mastery updates (#55, #57)
>
> Both issues ask for a one-step mastery update using fitted parameters, for tutors that need an answer
> per response without refitting. I have a prototype: constant time per answer, 2.4 µs per answer in
> plain Python, and it matches `predict` to 1.6e-9 on 294k answers. It uses a normalized two-state update
> that stays accurate near 0 and 1. Do you have a preference for the API? For example,
> `model.update(skill, mastery, correct)` returning the new mastery, plus a batch form for many students.
>
> The docs could also say what `partial_fit` does today: a warm-started refit on the new data only.
>
> ### 4. Optional, research-based features (would you want any in pyBKT?)
>
> - **Fit diagnostics** (warnings only): on ASSISTments, 9.1% of the fits pyBKT keeps have guess or slip
>   above 0.5, 11.8% are stuck at 0 or 1, and in 74 of 110 skills the `num_fits` restarts disagree by more
>   than 0.1. A warning when predictions fall back to 0.5 would also address #45. (Pardos & Heffernan 2010)
> - **Beta priors on the parameters** (Beck & Chang 2007), `priors=...`: on 48 ASSISTments skills, fits
>   stuck at 0 or 1 go from 15.3% to 0%, and held-out log-likelihood is slightly better on average.
> - **SQUAREM acceleration** (Varadhan & Roland 2008), `accelerate="squarem"`: 1.4 to 1.7x fewer EM steps;
>   in 48 runs it never ended at a worse fit.
> - **Exact fitting of data larger than memory**, using forward-only smoothing (Cappé 2011): same counts
>   as forward-backward (2.5e-16 relative), reading the data in chunks.
> - **Online parameter updates** (Cappé & Moulines 2009), experimental: one pass over synthetic data gets
>   within 0.0016 of the true parameters, against 0.0013 for batch EM.
> - **DuckDB loader** as an optional extra (`pyBKT[duckdb]`): all of `convert_data` in one query, same
>   output; reading and converting 20M rows takes 14.3 s, against 60.8 s today.
>
> ### 5. Release workflow
>
> `publish.yml` and `release.yml` can both publish (details in #65). One option: `release.yml` also runs
> on tag pushes, and `publish.yml` is removed. Would you like a PR, or do you prefer to keep both?
>
> ### 6. For information: a Rust E-step
>
> I also wrote the E-step in Rust, with no `unsafe` code in the crate. It is bit-identical to the C++
> E-step, 2.2x faster on one thread, and 10 to 35x faster with SIMD and threads. It would be a separate,
> optional package, so pyBKT wouldn't need a Rust toolchain. Nothing to decide now.
>
> ### 7. Housekeeping
>
> #54 (`random.randint(0, 1e8)`) was fixed by #48 and #56 and can be closed.
>
> I'd start with section 2 (no result changes) and section 3, and send the section 1 fixes one at a time
> as you decide. Which of these would you like PRs for?
