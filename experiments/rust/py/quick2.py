import time, common, bkt_rs, sys
def tm(fn, n=9):
    fn(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)
f=common.load(sys.argv[1] if len(sys.argv)>1 else 'synth5m'); N=f['data'].shape[1]
a=(f['data'], f['resources'], f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])
for vn, fn in [('serial', lambda: bkt_rs.e_step(*a)), ('serial noalpha', lambda: bkt_rs.e_step(*a, write_alpha=False)), ('predict serial', lambda: bkt_rs.predict(*a)), ('predict t4', lambda: bkt_rs.predict(*a, threads=4))]:
    print(f'{vn:16s} {tm(fn)/N*1e9:6.2f}', end=' | ')
print()
