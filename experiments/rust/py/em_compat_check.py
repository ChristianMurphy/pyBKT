"""Run pyBKT's own EM_fit loop (20 iterations, tol=-1) with the C++ E-step and
with the bkt_rs shim (serial) and compare the fitted parameters bitwise."""
import numpy as np, common, copy
import pyBKT.fit.EM_fit as EM
import bkt_rs_compat
from pyBKT.fit import E_step as CPP
f = common.load('as_all'); d, m = common.dm(f)
l, fo = m['learns'], m['forgets']
m['As'] = np.stack([np.array([[1 - l[i], fo[i]], [l[i], 1 - fo[i]]]) for i in range(len(l))])
res = {}
for name, mod in [('cpp', CPP), ('rust', bkt_rs_compat)]:
    EM.E_step = mod
    model, lls = EM.EM_fit(copy.deepcopy(m), d, tol=-1, maxiter=20, parallel=False)
    res[name] = (model, lls)
EM.E_step = CPP
for k in ['prior', 'learns', 'forgets', 'guesses', 'slips', 'As', 'emissions', 'pi_0']:
    a, b = np.asarray(res['cpp'][0][k]), np.asarray(res['rust'][0][k])
    print(f'{k:8s} bit-identical={np.array_equal(a, b)} maxabs={np.max(np.abs(a-b)):.2e}')
print('loglik C++ (int-truncated) last 3:', res['cpp'][1][-3:].ravel(), ' rust (float):', res['rust'][1][-3:].ravel())
