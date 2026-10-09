import time, hashlib, numpy as np, common, bkt_rs, os
def tm(fn, n=7):
    fn(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)
out=[]
for name in ['as_all','synth5m']:
    f=common.load(name); N=f['data'].shape[1]
    a=(f['data'], f['resources'], f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])
    for L in (2,4,8):
        o=bkt_rs.e_step(*a, threads=4, lanes=L)
        h=hashlib.sha256(b''.join(np.ascontiguousarray(x).tobytes() for x in o[:3])+np.float64(o[3]).tobytes()+o[4].tobytes()).hexdigest()[:12]
        out.append(f"{name} L={L} hash={h} serial={tm(lambda: bkt_rs.e_step(*a, threads=0, lanes=L, write_alpha=False))/N*1e9:.2f} t4={tm(lambda: bkt_rs.e_step(*a, threads=4, lanes=L, write_alpha=False))/N*1e9:.2f}")
print(os.environ.get('BKT_RS_ISA'), bkt_rs.__file__.split('/')[-3]); print('\n'.join(out))
