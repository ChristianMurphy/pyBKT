import hashlib, numpy as np, common, bkt_rs
def h(out):
    m = hashlib.sha256()
    for x in out[:3]: m.update(np.ascontiguousarray(x).tobytes())
    m.update(np.float64(out[3]).tobytes())
    if out[4] is not None: m.update(out[4].tobytes())
    return m.hexdigest()[:16]
for name in ['as_all', 'synth5m']:
    f = common.load(name)
    a = (f['data'], f['resources'], f['starts'], f['lengths'], float(f['prior']), f['learns'], f['forgets'], f['guesses'], f['slips'])
    hs = {}
    for th in (1, 2, 3, 4, 8, 16):
        hs[th] = {h(bkt_rs.e_step(*a, threads=th)) for _ in range(5)}
    allh = set().union(*hs.values())
    print(name, 'chunked: distinct hashes over threads {1,2,3,4,8,16} x 5 runs =', len(allh), allh)
    print(name, 'serial: distinct hashes over 5 runs =', len({h(bkt_rs.e_step(*a, threads=0)) for _ in range(5)}))
    # chunk size changes the reduction tree (documented): show it
    print(name, 'chunk 16384 vs 65536 identical?', h(bkt_rs.e_step(*a, threads=4, chunk_attempts=16384)) == h(bkt_rs.e_step(*a, threads=4)))
    # C++ parallel run-to-run
    from pyBKT.fit import E_step
    d, m = common.dm(f)
    def hc():
        r = E_step.run(d, m, 1, 1, {})
        return hashlib.sha256(b''.join(np.ascontiguousarray(r[k]).tobytes() for k in ['all_trans_softcounts','all_emission_softcounts','all_initial_softcounts'])).hexdigest()[:16]
    print(name, 'C++ parallel: distinct count hashes over 20 runs =', len({hc() for _ in range(20)}))
