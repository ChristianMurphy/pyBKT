
### bkt_rs: 20 crates linked into the extension (normal edges, no proc-macros) incl. itself, 30 incl. build deps / proc-macros
| crate | version | unsafe lines (non-comment) | linked into the .so? |
|---|---|---:|---|
| autocfg | 1.5.1 | 0 | no (build-time / proc-macro) |
| bkt_rs | 0.1.0 | 0 | yes |
| crossbeam-deque | 0.8.8 | 41 | yes |
| crossbeam-epoch | 0.9.21 | 128 | yes |
| crossbeam-utils | 0.8.23 | 76 | yes |
| either | 1.19.0 | 2 | yes |
| fearless_simd | 1.1.0 | 103 | yes |
| heck | 0.5.0 | 0 | no (build-time / proc-macro) |
| libc | 0.2.190 | 799 | yes |
| matrixmultiply | 0.3.11 | 95 | yes |
| ndarray | 0.17.2 | 395 | yes |
| num-complex | 0.4.6 | 2 | yes |
| num-integer | 0.1.47 | 0 | yes |
| num-traits | 0.2.19 | 1 | yes |
| numpy | 0.29.0 | 234 | yes |
| once_cell | 1.21.4 | 47 | yes |
| proc-macro2 | 1.0.107 | 6 | no (build-time / proc-macro) |
| pyo3 | 0.29.3 | 1253 | yes |
| pyo3-build-config | 0.29.3 | 2 | no (build-time / proc-macro) |
| pyo3-ffi | 0.29.3 | 358 | yes |
| pyo3-macros | 0.29.3 | 0 | no (build-time / proc-macro) |
| pyo3-macros-backend | 0.29.3 | 53 | no (build-time / proc-macro) |
| quote | 1.0.47 | 0 | no (build-time / proc-macro) |
| rawpointer | 0.2.1 | 20 | yes |
| rayon | 1.12.0 | 64 | yes |
| rayon-core | 1.13.0 | 86 | yes |
| rustc-hash | 2.1.3 | 0 | yes |
| syn | 2.0.119 | 67 | no (build-time / proc-macro) |
| target-lexicon | 0.13.5 | 0 | no (build-time / proc-macro) |
| unicode-ident | 1.0.26 | 2 | no (build-time / proc-macro) |
| **total** | | **3834** (linked: 3704) | |

### bkt_lean (default): 5 crates linked into the extension (normal edges, no proc-macros) incl. itself, 14 incl. build deps / proc-macros
| crate | version | unsafe lines (non-comment) | linked into the .so? |
|---|---|---:|---|
| bkt_lean | 0.1.0 | 0 | yes |
| heck | 0.5.0 | 0 | no (build-time / proc-macro) |
| libc | 0.2.190 | 799 | yes |
| once_cell | 1.21.4 | 47 | yes |
| proc-macro2 | 1.0.107 | 6 | no (build-time / proc-macro) |
| pyo3 | 0.29.3 | 1253 | yes |
| pyo3-build-config | 0.29.3 | 2 | no (build-time / proc-macro) |
| pyo3-ffi | 0.29.3 | 358 | yes |
| pyo3-macros | 0.29.3 | 0 | no (build-time / proc-macro) |
| pyo3-macros-backend | 0.29.3 | 53 | no (build-time / proc-macro) |
| quote | 1.0.47 | 0 | no (build-time / proc-macro) |
| syn | 2.0.119 | 67 | no (build-time / proc-macro) |
| target-lexicon | 0.13.5 | 0 | no (build-time / proc-macro) |
| unicode-ident | 1.0.26 | 2 | no (build-time / proc-macro) |
| **total** | | **2587** (linked: 2457) | |

### bkt_lean +simd: 6 crates linked into the extension (normal edges, no proc-macros) incl. itself, 15 incl. build deps / proc-macros
| crate | version | unsafe lines (non-comment) | linked into the .so? |
|---|---|---:|---|
| bkt_lean | 0.1.0 | 0 | yes |
| fearless_simd | 1.1.0 | 103 | yes |
| heck | 0.5.0 | 0 | no (build-time / proc-macro) |
| libc | 0.2.190 | 799 | yes |
| once_cell | 1.21.4 | 47 | yes |
| proc-macro2 | 1.0.107 | 6 | no (build-time / proc-macro) |
| pyo3 | 0.29.3 | 1253 | yes |
| pyo3-build-config | 0.29.3 | 2 | no (build-time / proc-macro) |
| pyo3-ffi | 0.29.3 | 358 | yes |
| pyo3-macros | 0.29.3 | 0 | no (build-time / proc-macro) |
| pyo3-macros-backend | 0.29.3 | 53 | no (build-time / proc-macro) |
| quote | 1.0.47 | 0 | no (build-time / proc-macro) |
| syn | 2.0.119 | 67 | no (build-time / proc-macro) |
| target-lexicon | 0.13.5 | 0 | no (build-time / proc-macro) |
| unicode-ident | 1.0.26 | 2 | no (build-time / proc-macro) |
| **total** | | **2690** (linked: 2560) | |
