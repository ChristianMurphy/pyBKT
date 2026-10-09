"""Correctness: Rust bkt_rs vs C++ pyBKT E_step (same process)."""
import sys, numpy as np, common, bkt_rs
from pyBKT.fit import E_step

def cpp_fix(alpha_raw):
    # C++ returns a column-major (2,N) Eigen buffer that Python labels (2,N) C-order.
    N = alpha_raw.shape[1]
    return alpha_raw.ravel().reshape(N, 2).T

def cmp(a, b):
    a = np.atleast_1d(np.asarray(a, float)); b = np.atleast_1d(np.asarray(b, float))
    d = np.abs(a - b)
    rel = d / np.maximum(np.abs(b), 1e-300)
    rel[d == 0] = 0
    return dict(bit=bool(np.array_equal(a.view(np.uint64) if a.flags.c_contiguous else np.ascontiguousarray(a).view(np.uint64),
                                         np.ascontiguousarray(b).view(np.uint64))),
                maxabs=float(d.max()), maxrel=float(rel.max()))

def args(f, data=None, res=None):
    return (f['data'] if data is None else data, f['resources'] if res is None else res,
            f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])

def check_run(name, f, out, ref, alpha_ref):
    tr, em, ini, ll, al = out
    rows = []
    for k, x, r in [('trans', tr, ref['all_trans_softcounts']), ('emission', em, ref['all_emission_softcounts']),
                    ('init', ini, ref['all_initial_softcounts'])]:
        rows.append((k, cmp(x, r)))
    rows.append(('loglike(trunc)', dict(bit=int(ll) == ref['total_loglike'], maxabs=abs(int(ll) - ref['total_loglike']), maxrel=0.0)))
    if al is not None:
        rows.append(('alpha', cmp(al, alpha_ref)))
    return rows

def main():
    results = {}
    for name in ['as_all', 'synth5m']:
        f = common.load(name); d, m = common.dm(f)
        ref = E_step.run(d, m, 1, 0, {})
        for k, rk in [('all_trans_softcounts','ref_trans'),('all_emission_softcounts','ref_emission'),('all_initial_softcounts','ref_init'),('alpha','ref_alpha')]:
            assert np.array_equal(ref[k], f[rk]), (name, k)
        alpha_ref = cpp_fix(ref['alpha'])
        ref_pred = E_step.predict(d, m, 0, {})
        variants = {
            'a serial': lambda: bkt_rs.e_step(*args(f), threads=0),
            'b rayon t=1': lambda: bkt_rs.e_step(*args(f), threads=1),
            'b rayon t=4': lambda: bkt_rs.e_step(*args(f), threads=4),
            'c no-alpha serial': lambda: bkt_rs.e_step(*args(f), threads=0, write_alpha=False),
            'c no-alpha t=4': lambda: bkt_rs.e_step(*args(f), threads=4, write_alpha=False),
            'd int8/int32 serial': lambda: bkt_rs.e_step(*args(f, f['data'].astype(np.int8), f['resources'].astype(np.int32)), threads=0),
            'd int8/uint16 t=4': lambda: bkt_rs.e_step(*args(f, f['data'].astype(np.int8), f['resources'].astype(np.uint16)), threads=4),
        }
        print(f'== {name}: C++ serial reference reproduced bit-exactly from E_step.run(d,m,1,0,{{}}); C++ loglike (truncated int) = {ref["total_loglike"]}')
        for vn, fn in variants.items():
            out = fn()
            print(f'  {vn}: rust loglike = {out[3]!r}')
            for k, c in check_run(name, f, out, ref, alpha_ref):
                print(f'    {k:15s} bit={c["bit"]!s:5s} maxabs={c["maxabs"]:.3e} maxrel={c["maxrel"]:.3e}')
        # C++ parallel vs C++ serial, for context
        refp = E_step.run(d, m, 1, 1, {})
        print('  [C++ parallel vs C++ serial]')
        for k in ['all_trans_softcounts','all_emission_softcounts','all_initial_softcounts']:
            c = cmp(refp[k], ref[k]); print(f'    {k:24s} bit={c["bit"]} maxrel={c["maxrel"]:.3e}')
        for vn, th in [('predict serial', 0), ('predict t=4', 4)]:
            p = bkt_rs.predict(*args(f), threads=th)
            c = cmp(p, ref_pred)
            print(f'  {vn}: vs E_step.predict bit={c["bit"]} maxabs={c["maxabs"]:.3e} maxrel={c["maxrel"]:.3e}')
        p8 = bkt_rs.predict(*args(f, f['data'].astype(np.int8), f['resources'].astype(np.uint16)), threads=4)
        print('  predict int8/uint16 t=4 bit-identical to int32/int64:', np.array_equal(p8, bkt_rs.predict(*args(f), threads=4)))

if __name__ == '__main__':
    main()
