"""Driver for the bkt_lean comparison; every measurement in a fresh process.
usage: bench_lean_all.py OUT.jsonl [ROUNDS]"""
import subprocess, sys, os, json
out = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
py = os.path.join(root, 'venv/bin/python')
cpp = ['cpp_serial', 'cpp_par', 'cpp_predict_serial', 'cpp_predict_par', 'cpp_fit_serial', 'cpp_fit_par']
rs = ['rs_serial', 'rs_serial_noalpha', 'rs_t4', 'rs_t4_noalpha', 'rs_predict_serial', 'rs_predict_t4',
      'rs_fearless_L4_serial_noalpha', 'rs_fearless_L8_t4_noalpha', 'rs_fit_serial', 'rs_fit_par']
lean = ['lean_serial_noalpha_copy', 'lean_serial_noalpha_borrow', 'lean_serial_noalpha_codes', 'lean_serial_noalpha_ds',
        'lean_t4_noalpha_copy', 'lean_t4_noalpha_codes', 'lean_t4_noalpha_ds',
        'lean_serial_alpha_bytearray', 'lean_serial_alpha_bytes', 'lean_serial_alpha_numpy', 'lean_serial_alpha_numpy_borrow',
        'lean_t4_alpha_bytearray', 'lean_t4_alpha_bytes', 'lean_t4_alpha_numpy',
        'lean_predict_serial_bytearray', 'lean_predict_serial_bytes', 'lean_predict_serial_numpy',
        'lean_predict_t4_bytearray', 'lean_predict_t4_bytes', 'lean_predict_t4_numpy',
        'lean_plain_L4_serial_ds', 'lean_plain_L8_serial_ds', 'lean_plain_L4_t4_ds', 'lean_plain_L8_t4_ds',
        'lean_fit_serial', 'lean_fit_t4', 'lean_fit_t4_L4_plain']
lean_simd = ['lean_fearless_L4_serial_ds', 'lean_fearless_L8_serial_ds', 'lean_fearless_L4_t4_ds', 'lean_fearless_L8_t4_ds',
             'lean_fit_serial_L8_fearless', 'lean_fit_t4_L8_fearless']
tiny = [('cpp_serial', 'default'), ('cpp_par', 'default'), ('cpp_predict_par', 'default'), ('rs_serial_noalpha', 'default'),
        ('rs_t4_noalpha', 'default'), ('rs_serial', 'default'), ('lean_serial_noalpha_copy', 'lean'), ('lean_serial_noalpha_borrow', 'lean'),
        ('lean_t4_noalpha_copy', 'lean'), ('lean_serial_alpha_bytearray', 'lean'), ('lean_serial_alpha_numpy', 'lean'),
        ('lean_predict_serial_bytearray', 'lean'), ('lean_spawn_t4_forced', 'lean')]
jobs = []
for fx in ['as_all', 'synth5m']:
    jobs += [(fx, v, 'default') for v in cpp + rs]
    jobs += [(fx, v, 'lean') for v in lean]
    jobs += [(fx, v, 'lean_simd') for v in lean_simd]
jobs += [('tiny', v, fl) for v, fl in tiny]
with open(out, 'a') as fo:
    for r in range(rounds):
        for fx, v, fl in jobs:
            env = dict(os.environ, PYTHONPATH=os.path.join(root, 'inst', fl), FLAVOR=fl)
            p = subprocess.run([py, os.path.join(here, 'bench_lean_one.py'), fx, v], env=env, capture_output=True, text=True, cwd=here)
            if p.returncode != 0:
                print('FAILED', fx, v, fl, p.stderr[-1500:], flush=True); continue
            rec = json.loads(p.stdout.strip().splitlines()[-1]); rec['round'] = r
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            print(r, fx, v, fl, f"{rec['ns_min']:.2f} ns", flush=True)
