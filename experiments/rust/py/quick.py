import time, numpy as np, common, bkt_rs, sys
from pyBKT.fit import E_step
def tm(fn, n=7):
    fn(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)
for name in sys.argv[1:] or ['as_all','synth5m']:
    f=common.load(name); d,m=common.dm(f); N=f['data'].shape[1]
    a=(f['data'], f['resources'], f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])
    for vn, fn in [('cpp serial', lambda: E_step.run(d,m,1,0,{})), ('cpp par', lambda: E_step.run(d,m,1,1,{})),
                   ('rs serial', lambda: bkt_rs.e_step(*a)), ('rs serial noalpha', lambda: bkt_rs.e_step(*a, write_alpha=False)),
                   ('rs t4', lambda: bkt_rs.e_step(*a, threads=4)), ('rs t4 noalpha', lambda: bkt_rs.e_step(*a, threads=4, write_alpha=False)),
                   ('cpp predict serial', lambda: E_step.predict(d,m,0,{})), ('rs predict serial', lambda: bkt_rs.predict(*a))]:
        print(f'{name:8s} {vn:20s} {tm(fn)/N*1e9:7.2f} ns/attempt')
