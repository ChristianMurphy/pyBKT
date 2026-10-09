"""bkt_lean correctness: vs C++ (serial bit-identical), vs bkt_rs (same chunk plan / lanes => same bits),
every input mode x output mode, thread-count independence, predict, random generic cases."""
import numpy as np, hashlib, common, bkt_lean, bkt_rs
from pyBKT.fit import E_step
from correctness import cpp_fix, cmp
import lean_api as L

def H(o):
    m = hashlib.sha256()
    for x in o[:3]: m.update(np.ascontiguousarray(x).tobytes())
    m.update(np.float64(o[3]).tobytes())
    if o[4] is not None: m.update(np.ascontiguousarray(o[4]).tobytes())
    return m.hexdigest()[:12]

fails = 0
def check(cond, msg):
    global fails
    if not cond:
        fails += 1; print('FAIL', msg)

simd = ['plain'] + (['fearless'] if bkt_lean.HAS_SIMD else [])
print('bkt_lean HAS_SIMD =', bkt_lean.HAS_SIMD, 'level =', bkt_lean.simd_level())
for name in ['as_all', 'synth5m']:
    f = common.load(name); d, m = common.dm(f); N = f['data'].shape[1]
    a = (f['data'], f['resources'], f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])
    ref = E_step.run(d, m, 1, 0, {})
    refc = (ref['all_trans_softcounts'], ref['all_emission_softcounts'], ref['all_initial_softcounts'])
    ref_alpha = cpp_fix(ref['alpha'])
    # exact serial: every input x output mode, int8/uint16 too
    for inp in ('copy', 'borrow', 'codes'):
        for am in ('none', 'bytearray', 'bytes', 'numpy'):
            for dt in ('native', 'narrow'):
                aa = a if dt == 'native' else (f['data'].astype(np.int8), f['resources'].astype(np.uint16)) + a[2:]
                o = L.e_step(*aa, alpha=am, input=inp)
                ok = all(np.array_equal(x, y) for x, y in zip(o[:3], refc)) and int(o[3]) == ref['total_loglike']
                if am != 'none': ok &= np.array_equal(o[4], ref_alpha)
                check(ok, f'{name} serial {inp} {am} {dt}')
    ds = bkt_lean.Dataset(f['data'], f['resources'], f['starts'], f['lengths'])
    o = L.as_arrays(bkt_lean.e_step_ds(ds, *a[4:], alpha='bytearray'), 1, 1, N)
    check(all(np.array_equal(x, y) for x, y in zip(o[:3], refc)) and np.array_equal(o[4], ref_alpha), f'{name} Dataset serial')
    print(f'{name}: exact serial bit-identical to C++ for input copy/borrow/codes/Dataset x alpha none/bytearray/bytes/numpy x int32+int64/int8+uint16')
    # chunked: identical over threads and to bkt_rs chunked
    rs_chunk = bkt_rs.e_step(*a, threads=4)
    hs = set()
    for th in (1, 2, 3, 4, 8):
        for am in ('bytearray', 'numpy', 'bytes'):
            o = L.e_step(*a, threads=th, alpha=am)
            hs.add(H(o))
    check(len(hs) == 1, f'{name} chunked thread/outmode independence {hs}')
    o = L.e_step(*a, threads=4, alpha='bytearray')
    same_rs = H(o) == H(rs_chunk)
    rel = max(cmp(x, y)['maxrel'] for x, y in zip(o[:3], refc))
    check(rel <= 1e-12 and np.array_equal(o[4], ref_alpha), f'{name} chunked vs C++ {rel}')
    print(f'{name}: chunked: 1 hash over threads {{1,2,3,4,8}} x outputs; == bkt_rs chunked bits: {same_rs}; max rel vs C++ {rel:.1e}')
    for imp in simd:
        for Ln in (2, 4, 8):
            hs = {H(L.e_step(*a, threads=th, lanes=Ln, lane_impl=imp, alpha='bytearray')) for th in (0, 1, 4)}
            o = L.e_step(*a, threads=4, lanes=Ln, lane_impl=imp, alpha='bytearray')
            rs = bkt_rs.e_step(*a, threads=4, lanes=Ln)
            rel = max(cmp(x, y)['maxrel'] for x, y in zip(o[:3], refc))
            check(len(hs) == 1 and rel <= 1e-12 and np.array_equal(o[4], ref_alpha), f'{name} lanes {imp} {Ln}')
            print(f'{name}: lanes {imp} L={Ln}: hashes over threads {{0,1,4}} = {len(hs)}, == bkt_rs lanes bits: {H(o) == H(rs)}, max rel vs C++ {rel:.1e}')
    # predict
    pref = E_step.predict(d, m, 0, {})
    for th in (0, 4):
        for om in ('bytearray', 'bytes', 'numpy'):
            for inp in (('copy', 'borrow', 'codes') if th == 0 else ('copy', 'codes')):
                p = L.predict(*a, threads=th, out_mode=om, input=inp)
                check(np.array_equal(p, pref), f'{name} predict {th} {om} {inp}')
    print(f'{name}: predict bit-identical to E_step.predict (threads 0/4, all output and input modes)')

# random generic cases (K>1, R>1, unsorted/gapped layouts -> packed output path), C++ with fixed=-1 arrays
rng = np.random.default_rng(7)
nc = 0
for case in range(200):
    K = int(rng.integers(1, 5)); R = int(rng.integers(1, 5)); S = int(rng.integers(1, 30))
    lengths = rng.integers(1, 60, S).astype(np.int64); gaps = rng.integers(0, 3, S)
    starts = np.zeros(S, np.int64); pos = 0
    for i in range(S):
        pos += gaps[i]; starts[i] = pos + 1; pos += lengths[i]
    N = int(pos + rng.integers(0, 3))
    perm = rng.permutation(S) if case % 2 else np.arange(S)
    starts = starts[perm].copy(); lengths = lengths[perm].copy()
    data = rng.integers(0, 3, (K, N)).astype(np.int32)
    if K > 1: data[rng.random((K, N)) < 0.7] = 0
    res = rng.integers(1, R + 1, N).astype(np.int64)
    learns = rng.random(R); forgets = rng.random(R) * 0.3; guesses = rng.random(K) * 0.5; slips = rng.random(K) * 0.3
    prior = float(rng.random())
    if case % 7 == 0:
        forgets[:] = 0; learns[0] = 0; prior = 0.0 if case % 2 else 1.0; guesses[0] = 0.0; slips[-1] = 0.0
    dd = dict(data=data, resources=res, starts=starts, lengths=lengths)
    mm = dict(prior=prior, learns=learns, forgets=forgets, guesses=guesses, slips=slips)
    fx = dict(learn=-np.ones(R), forget=-np.ones(R), guess=-np.ones(K), slip=-np.ones(K))
    ref = E_step.run(dd, mm, 1, 0, fx)
    covered = np.zeros(N, bool)
    for s, l in zip(starts, lengths): covered[s - 1:s - 1 + l] = True
    aa = (data, res, starts, lengths, prior, learns, forgets, guesses, slips)
    refc = (ref['all_trans_softcounts'], ref['all_emission_softcounts'], ref['all_initial_softcounts'])
    ra = cpp_fix(ref['alpha'])
    for th, inp, am in [(0, 'borrow', 'numpy'), (0, 'copy', 'bytearray'), (0, 'codes', 'bytes'), (3, 'copy', 'bytearray'), (3, 'codes', 'numpy'), (3, 'copy', 'bytes')]:
        o = L.e_step(*aa, threads=th, input=inp, alpha=am, chunk_attempts=50)
        if th == 0:
            ok = all(np.array_equal(x, y) for x, y in zip(o[:3], refc))
        else:
            ok = all(np.allclose(x, y, rtol=1e-12, atol=1e-12) for x, y in zip(o[:3], refc))
        ok &= np.array_equal(o[4][:, covered], ra[:, covered], equal_nan=True)
        nc += 1
        check(ok, f'generic case {case} th={th} {inp} {am}')
    if K == 1:
        for imp in simd:
            for Ln in (2, 4, 8):
                o = L.e_step(*aa, threads=2, lanes=Ln, lane_impl=imp, alpha='bytearray', chunk_attempts=40)
                ok = all(np.allclose(x, y, rtol=1e-12, atol=1e-12) for x, y in zip(o[:3], refc))
                ok &= np.array_equal(o[4][:, covered], ra[:, covered], equal_nan=True)
                nc += 1
                check(ok, f'generic lanes case {case} {imp} {Ln}')
    pref = E_step.predict(dd, mm, 0, fx)
    for th, om in [(0, 'numpy'), (3, 'bytearray')]:
        p = L.predict(*aa, threads=th, out_mode=om, chunk_attempts=50)
        nc += 1
        check(np.array_equal(p[:, covered], pref[:, covered], equal_nan=True), f'generic predict {case}')
print(f'generic random checks: {nc}')
print('TOTAL FAILURES:', fails)
