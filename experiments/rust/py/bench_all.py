"""Driver: every (round, fixture, variant, flavor) in a fresh process.
usage: bench_all.py OUT.jsonl [ROUNDS]"""
import subprocess, sys, os, json
out = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
py = os.path.join(root, 'venv/bin/python')
cpp = ['cpp_serial', 'cpp_par', 'cpp_predict_serial', 'cpp_predict_par']
exact = ['rs_a_serial', 'rs_b_t1', 'rs_b_t2', 'rs_b_t4', 'rs_c_noalpha_serial', 'rs_c_noalpha_t4',
         'rs_c_predict_serial', 'rs_c_predict_t4', 'rs_d_i8u16_serial', 'rs_d_i8u16_noalpha_t4']
lanes = [f'rs_e_{imp}_L{L}_{m}' for imp in ('fearless', 'autovec', 'autovecdispatch') for L in (2, 4, 8) for m in ('serial', 't4')]
jobs = []
for fx in ['as_all', 'synth5m']:
    jobs += [(fx, v, 'default', '') for v in cpp]
    for fl in ('default', 'native', 'v1_unsafe_default'):
        jobs += [(fx, v, fl, '') for v in exact]
    jobs += [(fx, v, 'default', '') for v in lanes]
    jobs += [(fx, 'rs_e_fearless_L4_t4_alpha', 'default', '')]
    jobs += [(fx, f'rs_e_{imp}_L{L}_{m}', 'native', '') for imp in ('fearless', 'autovec') for L in (4, 8) for m in ('serial', 't4')]
    jobs += [(fx, f'rs_e_v1mv_L{L}_{m}', 'v1_unsafe_default', '') for L in (4, 8) for m in ('serial', 't4')]
    jobs += [(fx, f'rs_e_fearless_L{L}_serial', 'default', isa) for isa in ('sse2', 'sse4_2', 'avx2') for L in (2, 4, 8)]
with open(out, 'a') as fo:
    for r in range(rounds):
        for fx, v, fl, isa in jobs:
            env = dict(os.environ, PYTHONPATH=os.path.join(root, 'inst', fl), FLAVOR=fl)
            env.pop('BKT_RS_ISA', None)
            if isa: env['BKT_RS_ISA'] = isa
            p = subprocess.run([py, os.path.join(here, 'bench_one.py'), fx, v], env=env, capture_output=True, text=True, cwd=here)
            if p.returncode != 0:
                print('FAILED', fx, v, fl, p.stderr[-2000:], flush=True); continue
            rec = json.loads(p.stdout.strip().splitlines()[-1]); rec['round'] = r
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            print(r, fx, v, fl, isa, f"{rec['ns_min']:.2f} ns/att", flush=True)
