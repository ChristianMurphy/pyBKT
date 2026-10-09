import numpy as np
from pyBKT.util import check_data
from pyBKT.fit import E_step
from pyBKT.fit import M_step
import os

SOLVERS = ('em', 'squarem')

def EM_fit(model, data, tol = None, maxiter = None, parallel = True, fixed = {}, solver = 'em'):

    if tol is None: 
        tol = 1e-3
    if maxiter is None:
        maxiter = 100
    if solver not in SOLVERS:
        raise ValueError("solver must be one of: " + ", ".join(SOLVERS))
    if solver == 'squarem':
        return squarem_fit(model, data, tol, maxiter, parallel, fixed)

    num_subparts = data["data"].shape[0] #mmm the first dimension of data represents each subpart?? interesting.
    num_resources = len(model["learns"])
    log_likelihoods = np.zeros((maxiter, 1))

    for i in range(maxiter):
        result = E_step.run(data, model, 1, int(parallel), fixed)
        for j in range(num_resources):
            result['all_trans_softcounts'][j] = result['all_trans_softcounts'][j].transpose()
        for j in range(num_subparts):
            result['all_emission_softcounts'][j] = result['all_emission_softcounts'][j].transpose()

        log_likelihoods[i][0] = result['total_loglike']

        if(i > 1 and abs(log_likelihoods[i][0] - log_likelihoods[i-1][0]) < tol):
            break

        model = M_step.run(model, result['all_trans_softcounts'], result['all_emission_softcounts'], result['all_initial_softcounts'], fixed)

    return(model, log_likelihoods[:i+1])

def squarem_fit(model, data, tol, maxiter, parallel, fixed):
    """ EM accelerated with SQUAREM, scheme SqS3 (Varadhan and Roland 2008).

    Each cycle takes two EM steps, extrapolates along them, and takes one EM step
    from the extrapolated point. The step length is capped, and the cap grows
    after each accepted full-length step. When the log-likelihood after the cycle
    is below the one after its first EM step, the cycle falls back to the second
    plain EM step and the cap resets. maxiter bounds the number of E-steps.
    """
    num_subparts = data["data"].shape[0]
    num_resources = len(model["learns"])
    log_likelihoods = []
    returned = None

    def em_map(params):
        nonlocal returned
        if params is not returned: # the M-step that returned params left them in the model
            set_params(model, params)
        result = E_step.run(data, model, 1, int(parallel), fixed)
        for j in range(num_resources):
            result['all_trans_softcounts'][j] = result['all_trans_softcounts'][j].transpose()
        for j in range(num_subparts):
            result['all_emission_softcounts'][j] = result['all_emission_softcounts'][j].transpose()
        log_likelihoods.append(np.asarray(result['total_loglike']).item())
        M_step.run(model, result['all_trans_softcounts'], result['all_emission_softcounts'], result['all_initial_softcounts'], fixed = fixed)
        returned = get_params(model)
        return returned, log_likelihoods[-1]

    eps, step_min, step_max = 1e-6, 1.0, 1.0
    params0 = get_params(model)
    params1, ll0 = em_map(params0)
    while len(log_likelihoods) + 4 <= maxiter:
        params2, ll1 = em_map(params1)
        r = params1 - params0
        v = params2 - 2 * params1 + params0
        v_norm = np.sqrt(v @ v)
        alpha = -np.sqrt(r @ r) / v_norm if v_norm > 0 else -step_min
        alpha = max(min(alpha, -step_min), -step_max)
        # Parameters that EM leaves unchanged, such as fixed ones or forgets in a
        # model without forgetting, keep their exact value instead of being clipped.
        moving = (r != 0) | (v != 0)
        params = np.where(moving, np.clip(params0 - 2 * alpha * r + alpha ** 2 * v, eps, 1 - eps), params2)
        params, _ = em_map(params)
        params_next, ll = em_map(params)
        if not ll >= ll1: # also true for a NaN log-likelihood
            params, step_max = params2, step_min
            params_next, ll = em_map(params)
        elif alpha == -step_max:
            step_max *= 4
        converged = abs(ll - ll0) < tol
        params0, params1, ll0 = params, params_next, ll
        if converged:
            break

    set_params(model, params0)
    return(model, np.array(log_likelihoods).reshape((-1, 1)))

def get_params(model):
    return np.concatenate((model['learns'], model['forgets'], model['guesses'], model['slips'], [model['prior']]))

def set_params(model, params):
    num_resources, num_subparts = len(model['learns']), len(model['guesses'])
    learns, forgets = params[:num_resources], params[num_resources:2 * num_resources]
    guesses, slips = params[2 * num_resources:-1 - num_subparts], params[-1 - num_subparts:-1]
    model['learns'], model['forgets'], model['guesses'], model['slips'] = learns, forgets, guesses, slips
    As = np.empty((num_resources, 2, 2))
    As[:, 0, 0], As[:, 0, 1], As[:, 1, 0], As[:, 1, 1] = 1 - learns, forgets, learns, 1 - forgets
    emissions = np.empty((num_subparts, 2, 2))
    emissions[:, 0, 0], emissions[:, 0, 1], emissions[:, 1, 0], emissions[:, 1, 1] = 1 - guesses, guesses, slips, 1 - slips
    model['As'], model['emissions'] = As, emissions
    model['prior'] = params[-1]
    model['pi_0'] = np.array([[1 - params[-1]], [params[-1]]])
