
#### as_all (449,220 attempts): E-step

ns/attempt = min over 3 rounds x 9 runs (max/min of per-round mins). Minor page faults per call (`ru_minflt`).

| variant | ns/attempt | x C++ serial | x C++ par | minor faults / call |
|---|---:|---:|---:|---:|
| C++ E_step.run parallel=0 | 49.07 (1.04) | 1.00 | 0.59 | 195 |
| C++ E_step.run parallel=1 (4 OpenMP) | 29.10 (1.03) | 1.69 | 1.00 | 195 |
| bkt_rs exact serial, alpha | 21.98 (1.02) | 2.23 | 1.32 | 24 |
| bkt_rs exact serial, no alpha | 21.46 (1.07) | 2.29 | 1.36 | 0 |
| bkt_rs 4 thr, alpha | 7.43 (1.17) | 6.61 | 3.92 | 25 |
| bkt_rs 4 thr, no alpha | 6.57 (1.35) | 7.47 | 4.43 | 6 |
| bkt_rs fearless L4 serial | 6.27 (1.02) | 7.83 | 4.64 | 0 |
| bkt_rs fearless L8 4 thr | 4.10 (1.55) | 11.96 | 7.09 | 40 |
| lean serial, input copy (to_vec) | 21.48 (1.13) | 2.28 | 1.35 | 49 |
| lean serial, input borrow (&[ReadOnlyCell], 0 copy) | 21.22 (1.01) | 2.31 | 1.37 | 0 |
| lean serial, input -> u8 codes per call | 21.51 (1.06) | 2.28 | 1.35 | 12 |
| lean serial, Dataset (converted once) | 21.07 (1.03) | 2.33 | 1.38 | 0 |
| lean 4 thr, input copy | 7.15 (1.35) | 6.86 | 4.07 | 58 |
| lean 4 thr, input -> codes per call | 8.98 (1.13) | 5.46 | 3.24 | 20 |
| lean 4 thr, Dataset | 7.35 (1.27) | 6.67 | 3.96 | 8 |
| lean serial, alpha -> bytearray (a) | 22.58 (1.09) | 2.17 | 1.29 | 244 |
| lean serial, alpha -> np.empty buffer (b) | 22.50 (1.04) | 2.18 | 1.29 | 130 |
| lean serial, borrow input + alpha -> np.empty (b) | 21.74 (1.03) | 2.26 | 1.34 | 25 |
| lean serial, alpha Vec<f64> -> bytes copy (c) | 34.80 (1.03) | 1.41 | 0.84 | 3931 |
| lean 4 thr, alpha -> bytearray (a) | 7.92 (1.40) | 6.20 | 3.67 | 250 |
| lean 4 thr, alpha -> np.empty, main-thread writes (b) | 8.16 (1.25) | 6.01 | 3.57 | 342 |
| lean 4 thr, alpha -> bytes copy (c) | 18.25 (1.01) | 2.69 | 1.59 | 3943 |
| lean lanes plain [f64;4] serial | 12.34 (1.71) | 3.98 | 2.36 | 0 |
| lean lanes plain [f64;8] serial | 14.38 (1.08) | 3.41 | 2.02 | 160 |
| lean lanes plain L4, 4 thr | 5.71 (1.51) | 8.59 | 5.09 | 26 |
| lean lanes plain L8, 4 thr | 6.23 (1.56) | 7.88 | 4.67 | 143 |
| lean +simd fearless L4 serial | 6.19 (1.39) | 7.92 | 4.70 | 0 |
| lean +simd fearless L8 serial | 6.08 (1.09) | 8.07 | 4.78 | 160 |
| lean +simd fearless L4, 4 thr | 4.73 (1.65) | 10.38 | 6.15 | 15 |
| lean +simd fearless L8, 4 thr | 4.35 (1.44) | 11.27 | 6.68 | 148 |

#### as_all: predict (one-step state predictions, 16 bytes/attempt output)

| variant | ns/attempt | x C++ serial | x C++ par | minor faults / call |
|---|---:|---:|---:|---:|
| C++ E_step.predict serial | 41.83 (1.09) | 1.00 | 0.59 | 195 |
| C++ E_step.predict parallel | 24.64 (1.01) | 1.70 | 1.00 | 195 |
| bkt_rs predict serial (numpy out) | 10.20 (1.05) | 4.10 | 2.42 | 81 |
| bkt_rs predict 4 thr | 4.51 (1.83) | 9.27 | 5.46 | 81 |
| lean predict serial -> bytearray (a) | 11.02 (1.01) | 3.80 | 2.24 | 244 |
| lean predict serial -> np.empty (b) | 10.52 (1.05) | 3.97 | 2.34 | 73 |
| lean predict serial -> bytes copy (c) | 22.29 (1.02) | 1.88 | 1.11 | 3931 |
| lean predict 4 thr -> bytearray (a) | 7.82 (1.29) | 5.35 | 3.15 | 251 |
| lean predict 4 thr -> np.empty (b) | 5.42 (1.17) | 7.72 | 4.55 | 482 |
| lean predict 4 thr -> bytes copy (c) | 15.03 (1.21) | 2.78 | 1.64 | 3941 |

#### as_all: full EM fit, 20 iterations (tol=-1)

| variant | seconds per fit | ns/attempt/iteration | x C++ serial fit | x C++ parallel fit |
|---|---:|---:|---:|---:|
| pyBKT EM_fit + C++ E-step, parallel=False | 0.45 (1.10) | 50.04 | 1.00 | 0.65 |
| pyBKT EM_fit + C++ E-step, parallel=True | 0.29 (1.07) | 32.47 | 1.54 | 1.00 |
| pyBKT EM_fit + bkt_rs shim, serial | 0.20 (1.05) | 22.81 | 2.19 | 1.42 |
| pyBKT EM_fit + bkt_rs shim, 4 thr | 0.07 (1.06) | 7.88 | 6.35 | 4.12 |
| bkt_lean.fit serial (exact) | 0.20 (1.10) | 21.74 | 2.30 | 1.49 |
| bkt_lean.fit 4 thr (chunked) | 0.06 (1.18) | 6.86 | 7.30 | 4.73 |
| bkt_lean.fit 4 thr, plain lanes L4 | 0.04 (1.65) | 4.32 | 11.59 | 7.52 |
| bkt_lean.fit +simd fearless L8, serial | 0.06 (1.08) | 6.31 | 7.93 | 5.14 |
| bkt_lean.fit +simd fearless L8, 4 thr | 0.02 (1.59) | 2.45 | 20.46 | 13.28 |

#### synth5m (5,000,000 attempts): E-step

ns/attempt = min over 3 rounds x 9 runs (max/min of per-round mins). Minor page faults per call (`ru_minflt`).

| variant | ns/attempt | x C++ serial | x C++ par | minor faults / call |
|---|---:|---:|---:|---:|
| C++ E_step.run parallel=0 | 54.67 (1.04) | 1.00 | 0.26 | 19532 |
| C++ E_step.run parallel=1 (4 OpenMP) | 14.06 (1.01) | 3.89 | 1.00 | 19532 |
| bkt_rs exact serial, alpha | 21.73 (1.08) | 2.52 | 0.65 | 165 |
| bkt_rs exact serial, no alpha | 20.02 (1.08) | 2.73 | 0.70 | 0 |
| bkt_rs 4 thr, alpha | 6.11 (1.52) | 8.95 | 2.30 | 573 |
| bkt_rs 4 thr, no alpha | 5.03 (1.46) | 10.86 | 2.79 | 1 |
| bkt_rs fearless L4 serial | 6.13 (1.02) | 8.91 | 2.29 | 0 |
| bkt_rs fearless L8 4 thr | 1.48 (1.20) | 36.98 | 9.51 | 1 |
| lean serial, input copy (to_vec) | 20.33 (1.02) | 2.69 | 0.69 | 647 |
| lean serial, input borrow (&[ReadOnlyCell], 0 copy) | 19.92 (1.04) | 2.75 | 0.71 | 945 |
| lean serial, input -> u8 codes per call | 20.01 (1.02) | 2.73 | 0.70 | 241 |
| lean serial, Dataset (converted once) | 19.71 (1.01) | 2.77 | 0.71 | 0 |
| lean 4 thr, input copy | 5.63 (1.11) | 9.72 | 2.50 | 648 |
| lean 4 thr, input -> codes per call | 5.85 (1.03) | 9.35 | 2.41 | 241 |
| lean 4 thr, Dataset | 5.04 (1.07) | 10.85 | 2.79 | 1 |
| lean serial, alpha -> bytearray (a) | 29.32 (1.02) | 1.86 | 0.48 | 20179 |
| lean serial, alpha -> np.empty buffer (b) | 23.31 (1.02) | 2.35 | 0.60 | 761 |
| lean serial, borrow input + alpha -> np.empty (b) | 22.87 (1.13) | 2.39 | 0.61 | 1059 |
| lean serial, alpha Vec<f64> -> bytes copy (c) | 37.45 (1.05) | 1.46 | 0.38 | 39711 |
| lean 4 thr, alpha -> bytearray (a) | 14.44 (1.15) | 3.79 | 0.97 | 20180 |
| lean 4 thr, alpha -> np.empty, main-thread writes (b) | 7.94 (1.01) | 6.88 | 1.77 | 1418 |
| lean 4 thr, alpha -> bytes copy (c) | 19.06 (1.08) | 2.87 | 0.74 | 39712 |
| lean lanes plain [f64;4] serial | 11.89 (1.07) | 4.60 | 1.18 | 0 |
| lean lanes plain [f64;8] serial | 11.53 (1.09) | 4.74 | 1.22 | 0 |
| lean lanes plain L4, 4 thr | 3.40 (1.07) | 16.08 | 4.14 | 1 |
| lean lanes plain L8, 4 thr | 3.04 (1.13) | 17.98 | 4.62 | 1 |
| lean +simd fearless L4 serial | 5.94 (1.12) | 9.20 | 2.37 | 0 |
| lean +simd fearless L8 serial | 4.96 (1.02) | 11.03 | 2.84 | 0 |
| lean +simd fearless L4, 4 thr | 1.66 (1.07) | 32.94 | 8.47 | 2 |
| lean +simd fearless L8, 4 thr | 1.39 (1.10) | 39.39 | 10.13 | 1 |

#### synth5m: predict (one-step state predictions, 16 bytes/attempt output)

| variant | ns/attempt | x C++ serial | x C++ par | minor faults / call |
|---|---:|---:|---:|---:|
| C++ E_step.predict serial | 47.25 (1.08) | 1.00 | 0.28 | 19532 |
| C++ E_step.predict parallel | 13.22 (1.11) | 3.57 | 1.00 | 19532 |
| bkt_rs predict serial (numpy out) | 9.21 (1.07) | 5.13 | 1.44 | 165 |
| bkt_rs predict 4 thr | 3.03 (1.14) | 15.57 | 4.36 | 572 |
| lean predict serial -> bytearray (a) | 16.85 (1.06) | 2.80 | 0.78 | 20179 |
| lean predict serial -> np.empty (b) | 11.17 (1.09) | 4.23 | 1.18 | 761 |
| lean predict serial -> bytes copy (c) | 26.38 (1.03) | 1.79 | 0.50 | 39711 |
| lean predict 4 thr -> bytearray (a) | 10.35 (1.09) | 4.57 | 1.28 | 20180 |
| lean predict 4 thr -> np.empty (b) | 4.55 (1.20) | 10.40 | 2.91 | 2509 |
| lean predict 4 thr -> bytes copy (c) | 15.80 (1.03) | 2.99 | 0.84 | 39712 |

#### synth5m: full EM fit, 20 iterations (tol=-1)

| variant | seconds per fit | ns/attempt/iteration | x C++ serial fit | x C++ parallel fit |
|---|---:|---:|---:|---:|
| pyBKT EM_fit + C++ E-step, parallel=False | 5.82 (1.03) | 58.17 | 1.00 | 0.29 |
| pyBKT EM_fit + C++ E-step, parallel=True | 1.69 (1.49) | 16.94 | 3.43 | 1.00 |
| pyBKT EM_fit + bkt_rs shim, serial | 2.31 (1.02) | 23.14 | 2.51 | 0.73 |
| pyBKT EM_fit + bkt_rs shim, 4 thr | 0.65 (1.29) | 6.47 | 8.99 | 2.62 |
| bkt_lean.fit serial (exact) | 1.99 (1.03) | 19.95 | 2.92 | 0.85 |
| bkt_lean.fit 4 thr (chunked) | 0.52 (1.03) | 5.24 | 11.11 | 3.24 |
| bkt_lean.fit 4 thr, plain lanes L4 | 0.34 (1.06) | 3.39 | 17.17 | 5.00 |
| bkt_lean.fit +simd fearless L8, serial | 0.52 (1.02) | 5.23 | 11.11 | 3.24 |
| bkt_lean.fit +simd fearless L8, 4 thr | 0.15 (1.16) | 1.51 | 38.57 | 11.23 |

#### Per-call overhead on a tiny input (1 student, 5 attempts; 3000 calls per process)

| variant | min µs | median µs | p90 µs |
|---|---:|---:|---:|
| C++ E_step.run serial | 3.3 | 3.6 | 4.3 |
| C++ E_step.run parallel (OpenMP) | 5.2 | 7.9 | 9.2 |
| C++ predict parallel | 4.6 | 5.6 | 7.3 |
| bkt_rs serial, no alpha | 1.9 | 2.1 | 3.6 |
| bkt_rs threads=4 (1 chunk -> inline) | 5.0 | 8.2 | 11.3 |
| bkt_rs serial + alpha (numpy array) | 2.0 | 2.1 | 3.8 |
| lean serial, copy input | 3.4 | 3.6 | 4.1 |
| lean serial, borrow input | 3.4 | 3.6 | 4.0 |
| lean threads=4 (1 chunk -> inline) | 3.4 | 3.7 | 4.4 |
| lean serial + alpha bytearray + np.frombuffer | 3.9 | 4.5 | 5.5 |
| lean serial + alpha into np.empty | 3.7 | 4.2 | 6.8 |
| lean predict -> bytearray + np.frombuffer | 2.1 | 2.3 | 2.8 |
| lean 4 thr, 4 chunks (forces 4 thread spawns; 20 attempts) | 46.9 | 109.8 | 383.0 |

1-min load average during the sweep: min 0.85, median 1.90, max 2.90
