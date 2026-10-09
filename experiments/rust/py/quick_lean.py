import time, common, bkt_rs, bkt_lean, sys, numpy as np
def tm(fn, n=9):
    fn(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)
for fx in sys.argv[1:]:
    f=common.load(fx); N=f['data'].shape[1]
    a=(f['data'], f['resources'], f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])
    ds=bkt_lean.Dataset(f['data'], f['resources'], f['starts'], f['lengths'])
    for vn, fn in [('rs noalpha', lambda: bkt_rs.e_step(*a, write_alpha=False)), ('lean copy', lambda: bkt_lean.e_step(*a)),
                   ('lean borrow', lambda: bkt_lean.e_step(*a, input='borrow')), ('lean codes', lambda: bkt_lean.e_step(*a, input='codes')),
                   ('lean ds', lambda: bkt_lean.e_step_ds(ds, *a[4:])),
                   ('rs t4 noalpha', lambda: bkt_rs.e_step(*a, threads=4, write_alpha=False)), ('lean ds t4', lambda: bkt_lean.e_step_ds(ds, *a[4:], threads=4))]:
        print(fx, f'{vn:14s} {tm(fn)/N*1e9:6.2f}')
