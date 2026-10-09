import json, sys, collections, statistics
recs = [json.loads(l) for l in open(sys.argv[1])]
g = collections.defaultdict(list)
for r in recs: g[(r['fixture'], r['variant'], r['flavor'])].append(r)
def get(fx, v, fl=None):
    for (f2, v2, fl2), rs in g.items():
        if f2 == fx and v2 == v and (fl is None or fl == fl2): return rs
    return None
def best(fx, v, key='ns_min'):
    rs = get(fx, v); return min(r[key] for r in rs) if rs else None
def cell(fx, v, key='ns_min'):
    rs = get(fx, v)
    if not rs: return '-'
    mins = [r[key] for r in rs]
    return f'{min(mins):.2f} ({max(mins)/min(mins):.2f})'
D = {
 'cpp_serial': 'C++ E_step.run parallel=0', 'cpp_par': 'C++ E_step.run parallel=1 (4 OpenMP)',
 'rs_serial': 'bkt_rs exact serial, alpha', 'rs_serial_noalpha': 'bkt_rs exact serial, no alpha', 'rs_t4': 'bkt_rs 4 thr, alpha', 'rs_t4_noalpha': 'bkt_rs 4 thr, no alpha',
 'rs_fearless_L4_serial_noalpha': 'bkt_rs fearless L4 serial', 'rs_fearless_L8_t4_noalpha': 'bkt_rs fearless L8 4 thr',
 'lean_serial_noalpha_copy': 'lean serial, input copy (to_vec)', 'lean_serial_noalpha_borrow': 'lean serial, input borrow (&[ReadOnlyCell], 0 copy)',
 'lean_serial_noalpha_codes': 'lean serial, input -> u8 codes per call', 'lean_serial_noalpha_ds': 'lean serial, Dataset (converted once)',
 'lean_t4_noalpha_copy': 'lean 4 thr, input copy', 'lean_t4_noalpha_codes': 'lean 4 thr, input -> codes per call', 'lean_t4_noalpha_ds': 'lean 4 thr, Dataset',
 'lean_serial_alpha_bytearray': 'lean serial, alpha -> bytearray (a)', 'lean_serial_alpha_numpy': 'lean serial, alpha -> np.empty buffer (b)',
 'lean_serial_alpha_numpy_borrow': 'lean serial, borrow input + alpha -> np.empty (b)', 'lean_serial_alpha_bytes': 'lean serial, alpha Vec<f64> -> bytes copy (c)',
 'lean_t4_alpha_bytearray': 'lean 4 thr, alpha -> bytearray (a)', 'lean_t4_alpha_numpy': 'lean 4 thr, alpha -> np.empty, main-thread writes (b)', 'lean_t4_alpha_bytes': 'lean 4 thr, alpha -> bytes copy (c)',
 'lean_plain_L4_serial_ds': 'lean lanes plain [f64;4] serial', 'lean_plain_L8_serial_ds': 'lean lanes plain [f64;8] serial',
 'lean_plain_L4_t4_ds': 'lean lanes plain L4, 4 thr', 'lean_plain_L8_t4_ds': 'lean lanes plain L8, 4 thr',
 'lean_fearless_L4_serial_ds': 'lean +simd fearless L4 serial', 'lean_fearless_L8_serial_ds': 'lean +simd fearless L8 serial',
 'lean_fearless_L4_t4_ds': 'lean +simd fearless L4, 4 thr', 'lean_fearless_L8_t4_ds': 'lean +simd fearless L8, 4 thr'}
P = {'cpp_predict_serial': 'C++ E_step.predict serial', 'cpp_predict_par': 'C++ E_step.predict parallel', 'rs_predict_serial': 'bkt_rs predict serial (numpy out)', 'rs_predict_t4': 'bkt_rs predict 4 thr',
     'lean_predict_serial_bytearray': 'lean predict serial -> bytearray (a)', 'lean_predict_serial_numpy': 'lean predict serial -> np.empty (b)', 'lean_predict_serial_bytes': 'lean predict serial -> bytes copy (c)',
     'lean_predict_t4_bytearray': 'lean predict 4 thr -> bytearray (a)', 'lean_predict_t4_numpy': 'lean predict 4 thr -> np.empty (b)', 'lean_predict_t4_bytes': 'lean predict 4 thr -> bytes copy (c)'}
F = {'cpp_fit_serial': 'pyBKT EM_fit + C++ E-step, parallel=False', 'cpp_fit_par': 'pyBKT EM_fit + C++ E-step, parallel=True',
     'rs_fit_serial': 'pyBKT EM_fit + bkt_rs shim, serial', 'rs_fit_par': 'pyBKT EM_fit + bkt_rs shim, 4 thr',
     'lean_fit_serial': 'bkt_lean.fit serial (exact)', 'lean_fit_t4': 'bkt_lean.fit 4 thr (chunked)', 'lean_fit_t4_L4_plain': 'bkt_lean.fit 4 thr, plain lanes L4',
     'lean_fit_serial_L8_fearless': 'bkt_lean.fit +simd fearless L8, serial', 'lean_fit_t4_L8_fearless': 'bkt_lean.fit +simd fearless L8, 4 thr'}
for fx in ['as_all', 'synth5m']:
    N = get(fx, 'cpp_serial')[0]['N']
    cs, cp = best(fx, 'cpp_serial'), best(fx, 'cpp_par')
    print(f'\n#### {fx} ({N:,} attempts): E-step\n\nns/attempt = min over 3 rounds x 9 runs (max/min of per-round mins). Minor page faults per call (`ru_minflt`).\n')
    print('| variant | ns/attempt | x C++ serial | x C++ par | minor faults / call |\n|---|---:|---:|---:|---:|')
    for v, desc in D.items():
        b = best(fx, v)
        if b is None: continue
        print(f'| {desc} | {cell(fx, v)} | {cs/b:.2f} | {cp/b:.2f} | {statistics.median(r["minflt_per_call"] for r in get(fx, v)):.0f} |')
    ps, pp = best(fx, 'cpp_predict_serial'), best(fx, 'cpp_predict_par')
    print(f'\n#### {fx}: predict (one-step state predictions, 16 bytes/attempt output)\n')
    print('| variant | ns/attempt | x C++ serial | x C++ par | minor faults / call |\n|---|---:|---:|---:|---:|')
    for v, desc in P.items():
        b = best(fx, v)
        if b is None: continue
        print(f'| {desc} | {cell(fx, v)} | {ps/b:.2f} | {pp/b:.2f} | {statistics.median(r["minflt_per_call"] for r in get(fx, v)):.0f} |')
    fs, fp = best(fx, 'cpp_fit_serial', 't_min'), best(fx, 'cpp_fit_par', 't_min')
    print(f'\n#### {fx}: full EM fit, 20 iterations (tol=-1)\n')
    print('| variant | seconds per fit | ns/attempt/iteration | x C++ serial fit | x C++ parallel fit |\n|---|---:|---:|---:|---:|')
    for v, desc in F.items():
        b = best(fx, v, 't_min')
        if b is None: continue
        print(f'| {desc} | {cell(fx, v, "t_min")} | {best(fx, v):.2f} | {fs/b:.2f} | {fp/b:.2f} |')
T = {'cpp_serial': 'C++ E_step.run serial', 'cpp_par': 'C++ E_step.run parallel (OpenMP)', 'cpp_predict_par': 'C++ predict parallel',
     'rs_serial_noalpha': 'bkt_rs serial, no alpha', 'rs_t4_noalpha': 'bkt_rs threads=4 (1 chunk -> inline)', 'rs_serial': 'bkt_rs serial + alpha (numpy array)',
     'lean_serial_noalpha_copy': 'lean serial, copy input', 'lean_serial_noalpha_borrow': 'lean serial, borrow input', 'lean_t4_noalpha_copy': 'lean threads=4 (1 chunk -> inline)',
     'lean_serial_alpha_bytearray': 'lean serial + alpha bytearray + np.frombuffer', 'lean_serial_alpha_numpy': 'lean serial + alpha into np.empty',
     'lean_predict_serial_bytearray': 'lean predict -> bytearray + np.frombuffer', 'lean_spawn_t4_forced': 'lean 4 thr, 4 chunks (forces 4 thread spawns; 20 attempts)'}
print('\n#### Per-call overhead on a tiny input (1 student, 5 attempts; 3000 calls per process)\n')
print('| variant | min µs | median µs | p90 µs |\n|---|---:|---:|---:|')
for v, desc in T.items():
    rs = get('tiny', v)
    if not rs: continue
    print(f'| {desc} | {min(r["us_min"] for r in rs):.1f} | {statistics.median(r["us_med"] for r in rs):.1f} | {max(r["us_p90"] for r in rs):.1f} |')
loads = [r['load'] for r in recs]
print(f'\n1-min load average during the sweep: min {min(loads):.2f}, median {statistics.median(loads):.2f}, max {max(loads):.2f}')
