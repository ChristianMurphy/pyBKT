
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
