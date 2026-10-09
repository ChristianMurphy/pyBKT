"""Time one (fixture, variant) in a fresh process. Prints one JSON line.
usage: bench_one.py FIXTURE VARIANT [REPS]"""
import sys, json, time, resource, gc, os
import numpy as np

fixture, variant = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 9
z = np.load(f'/tmp/claude-0/exp/{fixture}.npz')
f = {k: z[k] for k in ['data', 'resources', 'starts', 'lengths', 'prior', 'learns', 'forgets', 'guesses', 'slips',
                       'ref_trans', 'ref_emission', 'ref_init']}
N = f['data'].shape[1]
d = {k: f[k] for k in ('data', 'resources', 'starts', 'lengths')}
m = {'prior': float(f['prior']), **{k: f[k] for k in ('learns', 'forgets', 'guesses', 'slips')}}

if variant.startswith('cpp'):
    from pyBKT.fit import E_step
else:
    import bkt_rs
data, res = f['data'], f['resources']
if '_i8u16' in variant:
    data, res = data.astype(np.int8), res.astype(np.uint16)
a = (data, res, f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])

def counts_from(out):
    if isinstance(out, dict):
        return out['all_trans_softcounts'], out['all_emission_softcounts'], out['all_initial_softcounts']
    return out[0], out[1], out[2]

V = {
    'cpp_serial':          (lambda: E_step.run(d, m, 1, 0, {}), True),
    'cpp_par':             (lambda: E_step.run(d, m, 1, 1, {}), True),
    'cpp_predict_serial':  (lambda: E_step.predict(d, m, 0, {}), False),
    'cpp_predict_par':     (lambda: E_step.predict(d, m, 1, {}), False),
    'rs_a_serial':         (lambda: bkt_rs.e_step(*a, threads=0), True),
    'rs_b_t1':             (lambda: bkt_rs.e_step(*a, threads=1), True),
    'rs_b_t2':             (lambda: bkt_rs.e_step(*a, threads=2), True),
    'rs_b_t4':             (lambda: bkt_rs.e_step(*a, threads=4), True),
    'rs_c_noalpha_serial': (lambda: bkt_rs.e_step(*a, threads=0, write_alpha=False), True),
    'rs_c_noalpha_t4':     (lambda: bkt_rs.e_step(*a, threads=4, write_alpha=False), True),
    'rs_c_predict_serial': (lambda: bkt_rs.predict(*a, threads=0), False),
    'rs_c_predict_t4':     (lambda: bkt_rs.predict(*a, threads=4), False),
    'rs_d_i8u16_serial':   (lambda: bkt_rs.e_step(*a, threads=0), True),
    'rs_d_i8u16_noalpha_t4': (lambda: bkt_rs.e_step(*a, threads=4, write_alpha=False), True),
}
# SIMD-across-students: rs_e_<impl>_L<lanes>_<serial|t4>[_alpha]
#   impl in fearless | autovec | autovecdispatch | v1mv (v1 build: target_feature multiversioning)
if variant.startswith('rs_e_'):
    _, _, imp, L, mode, *rest = variant.split('_')
    kw = dict(threads=0 if mode == 'serial' else int(mode[1:]), lanes=int(L[1:]), write_alpha=bool(rest))
    if imp != 'v1mv':
        kw['lane_impl'] = {'autovecdispatch': 'autovec_dispatch'}.get(imp, imp)
    V[variant] = (lambda: bkt_rs.e_step(*a, **kw), True)
fn, has_counts = V[variant]
gc.collect()
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux
out = fn()
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
maxrel = None
if has_counts:
    maxrel = 0.0
    for x, r in zip(counts_from(out), (f['ref_trans'], f['ref_emission'], f['ref_init'])):
        maxrel = max(maxrel, float(np.max(np.abs(x - r) / np.abs(r))))
    assert maxrel <= 1e-12, (variant, maxrel)
del out; gc.collect()
ts = []
for _ in range(reps):
    t0 = time.perf_counter(); o = fn(); ts.append(time.perf_counter() - t0); del o
print(json.dumps(dict(fixture=fixture, variant=variant, flavor=os.environ.get('FLAVOR', ''), isa=os.environ.get('BKT_RS_ISA', 'auto'),
                      N=N, times=ts, ns_min=min(ts) / N * 1e9, ns_med=sorted(ts)[len(ts) // 2] / N * 1e9,
                      ns_max=max(ts) / N * 1e9, rss_over_kib=rss1 - rss0, maxrel_vs_cpp_serial=maxrel,
                      load=os.getloadavg()[0])))
