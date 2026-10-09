"""SIMD-across-students: correctness vs C++ serial reference and vs exact fsum loglike,
bit-equality between implementations / ISA levels / thread counts."""
import numpy as np, common, bkt_rs, hashlib, os, json, math
from correctness import cmp
FS = {'as_all': -251007.4258211593, 'synth5m': -3750860.040946558}  # math.fsum reference (ll_accuracy.py)
out = {}
for name in ['as_all', 'synth5m']:
    f = common.load(name)
    a = (f['data'], f['resources'], f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])
    ref = bkt_rs.e_step(*a)  # bit-identical to C++ serial
    for L in (2, 4, 8):
        for imp in ('fearless', 'autovec', 'autovec_dispatch'):
            hs = set()
            for th in (0, 1, 4):
                o = bkt_rs.e_step(*a, threads=th, lanes=L, lane_impl=imp)
                hs.add(hashlib.sha256(b''.join(np.ascontiguousarray(x).tobytes() for x in o[:3]) + np.float64(o[3]).tobytes() + o[4].tobytes()).hexdigest()[:12])
            r = {k: cmp(x, y)['maxrel'] for k, x, y in zip(['trans', 'em', 'init'], o, ref)}
            r['alpha_bit'] = bool(np.array_equal(o[4], ref[4]))
            r['ll_rel_vs_cpp_order'] = abs(o[3] - ref[3]) / abs(ref[3]); r['ll_rel_vs_fsum'] = abs(o[3] - FS[name]) / abs(FS[name])
            r['hashes'] = sorted(hs)
            print(os.environ.get('BKT_RS_ISA', 'auto'), name, L, imp, json.dumps(r))
