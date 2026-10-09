"""One (fixture, variant) measurement in a fresh process -> JSON line.
usage: bench_lean_one.py FIXTURE VARIANT  (PYTHONPATH must expose pyBKT and bkt_rs/bkt_lean)
FIXTURE 'tiny' = one student with 5 attempts (per-call overhead)."""
import sys, json, time, resource, gc, os, copy
import numpy as np

fixture, variant = sys.argv[1], sys.argv[2]
if fixture == 'tiny':
    z = np.load('/tmp/claude-0/exp/as_all.npz')
    f = {k: z[k] for k in ['prior', 'learns', 'forgets', 'guesses', 'slips']}
    n_seq = 4 if 'spawn' in variant else 1
    f['data'] = np.tile(np.array([[1, 2, 2, 1, 2]], np.int32), (1, n_seq)); f['resources'] = np.ones(5 * n_seq, np.int64)
    f['starts'] = np.arange(n_seq, dtype=np.int64) * 5 + 1; f['lengths'] = np.full(n_seq, 5, np.int64)
else:
    z = np.load(f'/tmp/claude-0/exp/{fixture}.npz')
    f = {k: z[k] for k in ['data', 'resources', 'starts', 'lengths', 'prior', 'learns', 'forgets', 'guesses', 'slips']}
N = f['data'].shape[1]
d = {k: f[k] for k in ('data', 'resources', 'starts', 'lengths')}
m = {'prior': float(f['prior']), **{k: f[k] for k in ('learns', 'forgets', 'guesses', 'slips')}}
a = (f['data'], f['resources'], f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])

def model_with_As():
    mm = copy.deepcopy(m); l, fo = mm['learns'], mm['forgets']
    mm['As'] = np.stack([np.array([[1 - l[i], fo[i]], [l[i], 1 - fo[i]]]) for i in range(len(l))])
    return mm

V = {}
kind = 'estep'
if variant.startswith('cpp'):
    from pyBKT.fit import E_step
    V.update({
        'cpp_serial': lambda: E_step.run(d, m, 1, 0, {}),
        'cpp_par': lambda: E_step.run(d, m, 1, 1, {}),
        'cpp_predict_serial': lambda: E_step.predict(d, m, 0, {}),
        'cpp_predict_par': lambda: E_step.predict(d, m, 1, {}),
    })
    if variant.startswith('cpp_fit'):
        import pyBKT.fit.EM_fit as EM
        par = variant.endswith('par')
        V[variant] = lambda: EM.EM_fit(model_with_As(), d, tol=-1, maxiter=20, parallel=par)
elif variant.startswith('rs_'):
    import bkt_rs
    V.update({
        'rs_serial': lambda: bkt_rs.e_step(*a),
        'rs_serial_noalpha': lambda: bkt_rs.e_step(*a, write_alpha=False),
        'rs_t4': lambda: bkt_rs.e_step(*a, threads=4),
        'rs_t4_noalpha': lambda: bkt_rs.e_step(*a, threads=4, write_alpha=False),
        'rs_predict_serial': lambda: bkt_rs.predict(*a),
        'rs_predict_t4': lambda: bkt_rs.predict(*a, threads=4),
        'rs_fearless_L8_t4_noalpha': lambda: bkt_rs.e_step(*a, threads=4, lanes=8, write_alpha=False),
        'rs_fearless_L4_serial_noalpha': lambda: bkt_rs.e_step(*a, lanes=4, write_alpha=False),
    })
    if variant.startswith('rs_fit'):
        import pyBKT.fit.EM_fit as EM, bkt_rs_compat
        EM.E_step = bkt_rs_compat
        par = variant.endswith('par')
        V[variant] = lambda: EM.EM_fit(model_with_As(), d, tol=-1, maxiter=20, parallel=par)
else:
    import bkt_lean
    def ln(**kw):
        out_mode = kw.get('alpha', 'none')
        def fn():
            o = None
            if out_mode == 'numpy':
                o = np.empty((2, N))
            r = bkt_lean.e_step(*a, out=o, **kw)
            al = r[4]
            if out_mode in ('bytearray', 'bytes'):
                al = np.frombuffer(al, np.float64).reshape(2, N)
            return (np.array(r[0]).reshape(-1, 2, 2), np.array(r[1]).reshape(-1, 2, 2), np.array(r[2]).reshape(2, 1), r[3], al)
        return fn
    def lp(**kw):
        om = kw.get('out_mode', 'bytearray')
        def fn():
            if om == 'numpy':
                return bkt_lean.predict(*a, out=np.empty((2, N)), **kw)
            return np.frombuffer(bkt_lean.predict(*a, **kw), np.float64).reshape(2, N)
        return fn
    ds = bkt_lean.Dataset(f['data'], f['resources'], f['starts'], f['lengths']) if '_ds' in variant else None
    def lds(**kw):
        return lambda: bkt_lean.e_step_ds(ds, *a[4:], **kw)
    for th, tn in ((0, 'serial'), (4, 't4')):
        for inp in ('copy', 'borrow', 'codes'):
            if th and inp == 'borrow':
                continue
            V[f'lean_{tn}_noalpha_{inp}'] = ln(threads=th, input=inp)
        V[f'lean_{tn}_noalpha_ds'] = lds(threads=th)
        for am in ('bytearray', 'bytes', 'numpy'):
            V[f'lean_{tn}_alpha_{am}'] = ln(threads=th, alpha=am)
            V[f'lean_predict_{tn}_{am}'] = lp(threads=th, out_mode=am)
        V[f'lean_{tn}_alpha_numpy_borrow'] = ln(threads=th, alpha='numpy', input='borrow') if th == 0 else None
        for imp in ('plain', 'fearless'):
            for L in (4, 8):
                V[f'lean_{imp}_L{L}_{tn}_ds'] = lds(threads=th, lanes=L, lane_impl=imp)
    V['lean_spawn_t4_forced'] = ln(threads=4, chunk_attempts=1)
    if variant.startswith('lean_fit'):
        # lean_fit_<serial|t4>[_L<l>_<imp>][_compat]
        parts = variant.split('_')
        th = 0 if parts[2] == 'serial' else 4
        lanes = int(parts[3][1:]) if len(parts) > 3 and parts[3].startswith('L') else 0
        imp = parts[4] if lanes else 'plain'
        V[variant] = lambda: bkt_lean.fit(*a, max_iter=20, tol=-1, threads=th, lanes=lanes, lane_impl=imp)
if 'fit' in variant:
    kind = 'fit'
if fixture == 'tiny':
    kind = 'tiny'
fn = V[variant]
reps = {'estep': 9, 'fit': 3, 'tiny': 3000}[kind]
gc.collect()
ru0 = resource.getrusage(resource.RUSAGE_SELF)
fn()  # warm-up
ru1 = resource.getrusage(resource.RUSAGE_SELF)
ts = []
flt0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
for _ in range(reps):
    t0 = time.perf_counter(); o = fn(); ts.append(time.perf_counter() - t0); del o
flt1 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
ts.sort()
denom = N * (20 if kind == 'fit' else 1)
print(json.dumps(dict(fixture=fixture, variant=variant, flavor=os.environ.get('FLAVOR', ''), N=N, kind=kind,
                      t_min=ts[0], t_med=ts[len(ts) // 2], ns_min=ts[0] / denom * 1e9, ns_med=ts[len(ts) // 2] / denom * 1e9,
                      us_min=ts[0] * 1e6, us_med=ts[len(ts) // 2] * 1e6, us_p90=ts[int(len(ts) * 0.9)] * 1e6,
                      minflt_per_call=(flt1 - flt0) / reps, rss_over_kib=ru1.ru_maxrss - ru0.ru_maxrss, load=os.getloadavg()[0])))
