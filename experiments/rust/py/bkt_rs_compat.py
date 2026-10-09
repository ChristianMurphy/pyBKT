"""Drop-in shim with the pyBKT.fit.E_step interface, backed by bkt_rs.
Differences from the C++ (deliberate, see REPORT.md):
  * parameters are taken per resource / per subpart (C++ without `fixed` uses
    learns[0]/forgets[0]/guesses[0]/slips[0] for every index);
  * total_loglike is a float (C++ truncates it to int via PyLong_FromLong);
  * alpha is a real (2, N) array (C++ returns a column-major buffer labelled (2, N)).
"""
import numpy as np
import bkt_rs

def _eff(model, fixed, key_model, key_fixed):
    v = np.asarray(model[key_model], dtype=np.float64)
    if key_fixed in fixed:
        fx = np.asarray(fixed[key_fixed], dtype=np.float64)
        v = np.where(fx >= 0, fx, v)
    return np.ascontiguousarray(v)

def _args(data, model, fixed):
    prior = float(fixed['prior']) if 'prior' in fixed else float(model['prior'])
    return (data['data'], data['resources'], np.asarray(data['starts'], np.int64), np.asarray(data['lengths'], np.int64),
            prior, _eff(model, fixed, 'learns', 'learn'), _eff(model, fixed, 'forgets', 'forget'),
            _eff(model, fixed, 'guesses', 'guess'), _eff(model, fixed, 'slips', 'slip'))

def run(data, model, num_outputs, parallel, fixed, write_alpha=True):
    tr, em, ini, ll, alpha = bkt_rs.e_step(*_args(data, model, fixed), threads=4 if parallel else 0, write_alpha=write_alpha)
    return {'all_trans_softcounts': tr, 'all_emission_softcounts': em, 'all_initial_softcounts': ini,
            'alpha': alpha, 'total_loglike': ll}

def predict(data, model, parallel, fixed):
    a = _args(data, model, fixed)
    return bkt_rs.predict(*a, threads=4 if parallel else 0,
                          pred_learns=np.ascontiguousarray(model['learns'], dtype=np.float64),
                          pred_forgets=np.ascontiguousarray(model['forgets'], dtype=np.float64))
