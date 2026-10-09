# Deep-research run: what was verified

A multi-agent search, fetch and adversarial verification run (59 agents) on: algorithmic, Python and
C/C++ options for fast, memory-lean and correct BKT fitting.

Outcome: almost every source the proxy allowed through was thin. Academic sources (arXiv,
EDM, publishers) were largely blocked. Only these claims survived 3-vote verification, all from the
nanobind project's own benchmark page (nanobind 1.2.0, Cython 0.29.28, Python 3.10.6, clang++ 15,
about 2023):

- nanobind compiles bindings about 2.7-4.4x faster than pybind11 and 1.6-4.4x faster than Cython (vote 3-0).
- nanobind's per-call overhead is about 3x lower than pybind11's for simple functions and about 10x lower
  when class instances are passed; it is roughly the same as Cython's (3-0).
- nanobind's binaries are about 3-5x smaller than pybind11's (2-1).
- All of these numbers come from one machine and one dated set of versions (3-0).

Relevance to pyBKT: the expensive work is one coarse E-step call per EM iteration (and, as
experiments/README.md section 7 shows, OpenMP per-call cost), not binding overhead. A binding swap
would bring smaller wheels and faster builds, not faster fitting, and does not change numerical
results if the kernel and compiler flags are unchanged.

Claims about cibuildwheel/delocate packaging on macOS were refuted 0-3. Nothing about EM
acceleration, online EM, SIMD layout, OpenMP determinism or floating-point reproducibility reached
verification. Those points in the README rest on our own experiments and the papers cited there.
