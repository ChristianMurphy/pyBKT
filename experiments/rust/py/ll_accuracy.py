"""Which loglike summation is more accurate? Recompute every log(norm) from alpha
(identical to C++) and sum with math.fsum (exactly rounded)."""
import math, numpy as np, common, bkt_rs
for name in ['as_all', 'synth5m']:
    f = common.load(name)
    a = (f['data'], f['resources'], f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])
    tr, em, ini, ll_serial, al = bkt_rs.e_step(*a, threads=0)
    ll_chunk = bkt_rs.e_step(*a, threads=4, write_alpha=False)[3]
    ll_lanes = bkt_rs.e_step(*a, threads=4, write_alpha=False, lanes=2)[3]
    g, s = f['guesses'][0], f['slips'][0]; L, F = f['learns'][0], f['forgets'][0]; p = float(f['prior'])
    d = f['data'][0]; N = d.size
    lik0 = np.where(d == 2, g, np.where(d == 1, 1 - g, 1.0)); lik1 = np.where(d == 2, 1 - s, np.where(d == 1, s, 1.0))
    first = np.zeros(N, bool); first[f['starts'] - 1] = True
    x0p = np.empty(N); x1p = np.empty(N)
    x0p[1:], x1p[1:] = al[0, :-1], al[1, :-1]
    p0 = (1 - L) * x0p + F * x1p; p1 = L * x0p + (1 - F) * x1p
    p0[first] = 1 - p; p1[first] = p
    norms = p0 * lik0 + p1 * lik1
    exact = math.fsum(np.log(norms).tolist())
    print(f'{name}: fsum={exact!r}\n  serial(C++ order) err={ll_serial-exact:+.3e}  chunked err={ll_chunk-exact:+.3e}  lanes err={ll_lanes-exact:+.3e}')
