"""Summarize results/bench_v2.jsonl into the markdown tables used in REPORT.md."""
import json, sys, collections, statistics
recs = [json.loads(l) for fn in sys.argv[1:] for l in open(fn)]
g = collections.defaultdict(list)
for r in recs:
    g[(r['fixture'], r['variant'], r['flavor'], r['isa'])].append(r)
def best(fx, v, fl='default', isa='auto'):
    rs = g.get((fx, v, fl, isa))
    return min(x['ns_min'] for x in rs) if rs else None
def cell(fx, v, fl='default', isa='auto'):
    rs = g.get((fx, v, fl, isa))
    if not rs: return '-'
    mins = [r['ns_min'] for r in sorted(rs, key=lambda r: r['round'])]
    return f'{min(mins):.2f} ({max(mins)/min(mins):.2f})'
def rss(fx, v, fl='default'):
    rs = g.get((fx, v, fl, 'auto'))
    return f"{statistics.median(r['rss_over_kib'] for r in rs)/1024:.1f}" if rs else '-'
desc = {
 'cpp_serial': 'C++ E_step.run(...,parallel=0)', 'cpp_par': 'C++ E_step.run(...,parallel=1), 4 OpenMP threads',
 'cpp_predict_serial': 'C++ E_step.predict serial', 'cpp_predict_par': 'C++ E_step.predict parallel',
 'rs_a_serial': '(a) exact serial port', 'rs_b_t1': '(b) chunked rayon, 1 thread', 'rs_b_t2': '(b) chunked rayon, 2 threads',
 'rs_b_t4': '(b) chunked rayon, 4 threads', 'rs_c_noalpha_serial': '(c) no alpha, serial', 'rs_c_noalpha_t4': '(c) no alpha, 4 threads',
 'rs_c_predict_serial': '(c) predict, serial', 'rs_c_predict_t4': '(c) predict, 4 threads',
 'rs_d_i8u16_serial': '(d) int8 data + uint16 resources, serial', 'rs_d_i8u16_noalpha_t4': '(d) int8/uint16, no alpha, 4 threads'}
for fx in ['as_all', 'synth5m']:
    N = g[(fx, 'cpp_serial', 'default', 'auto')][0]['N']
    cs, cp = best(fx, 'cpp_serial'), best(fx, 'cpp_par')
    cps, cpp_ = best(fx, 'cpp_predict_serial'), best(fx, 'cpp_predict_par')
    print(f'\n#### {fx} ({N:,} attempts): C++ and exact-path variants\n')
    print('ns/attempt = best per-process min over 3 rounds x 9 runs; (x) = max/min of the 3 per-round mins.\n')
    print('| variant | safe, portable build | x C++ serial | x C++ par | safe, target-cpu=native | v1 (unsafe), portable | safe/unsafe time ratio | peak RSS over inputs, MiB |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for v in desc:
        b = best(fx, v)
        pred = 'predict' in v
        bs, bp = (cps, cpp_) if pred else (cs, cp)
        u = best(fx, v, 'v1_unsafe_default')
        ratio = f'{b/u:.2f}' if u else '-'
        print(f'| {desc[v]} | {cell(fx, v)} | {bs/b:.2f} | {bp/b:.2f} | {cell(fx, v, "native")} | {cell(fx, v, "v1_unsafe_default")} | {ratio} | {rss(fx, v)} |')
    print(f'\n#### {fx}: SIMD across students (no alpha unless noted; runtime dispatch picked AVX-512 on this CPU)\n')
    print('| impl | L | serial ns/att | x C++ serial | 4 threads ns/att | x C++ par | native build serial | native build 4 thr |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for imp in ('fearless', 'autovecdispatch', 'autovec', 'v1mv'):
        for L in (2, 4, 8):
            fl = 'v1_unsafe_default' if imp == 'v1mv' else 'default'
            s, t = best(fx, f'rs_e_{imp}_L{L}_serial', fl), best(fx, f'rs_e_{imp}_L{L}_t4', fl)
            if s is None: continue
            name = {'fearless': 'fearless_simd f64xL, dispatch!', 'autovecdispatch': 'plain [f64;L] inside dispatch!',
                    'autovec': 'plain [f64;L], no dispatch (SSE2 baseline)', 'v1mv': 'v1 plain arrays + unsafe #[target_feature]'}[imp]
            print(f'| {name} | {L} | {cell(fx, f"rs_e_{imp}_L{L}_serial", fl)} | {cs/s:.2f} | {cell(fx, f"rs_e_{imp}_L{L}_t4", fl)} | {cp/t:.2f} | '
                  f'{cell(fx, f"rs_e_{imp}_L{L}_serial", "native")} | {cell(fx, f"rs_e_{imp}_L{L}_t4", "native")} |')
    v = 'rs_e_fearless_L4_t4_alpha'
    print(f'| fearless, L=4, 4 threads, *with* alpha | 4 | | | {cell(fx, v)} | {cp/best(fx, v):.2f} | | |')
    print(f'\n#### {fx}: fearless lanes, serial, ISA capped with BKT_RS_ISA (portable build)\n')
    print('| ISA | L=2 | L=4 | L=8 |\n|---|---:|---:|---:|')
    for isa in ('sse2', 'sse4_2', 'avx2', 'auto'):
        print(f'| {isa if isa != "auto" else "avx512 (auto)"} | ' + ' | '.join(cell(fx, f'rs_e_fearless_L{L}_serial', 'default', isa) for L in (2, 4, 8)) + ' |')
loads = [r['load'] for r in recs]
print(f'\n1-min load average during the runs: min {min(loads):.2f}, median {statistics.median(loads):.2f}, max {max(loads):.2f}')
