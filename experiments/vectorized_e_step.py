"""Prototype: the pure Python E-step with the forward-backward pass vectorized across sequences.

Same signature and return value as source-py/pyBKT/fit/EM_fit.py run(). The loop is over
time steps, and each step updates every sequence that is still running at once, so the
Python-level iteration count is the longest sequence length, not the total number of responses.
Evidence for an issue, not a PR: no multiprocessing, no normalizeLengths.
"""
import numpy as np


def run(data, model, trans_softcounts=None, emission_softcounts=None, init_softcounts=None, num_outputs=1, parallel=True, fixed={}):
    alldata = data["data"]
    bigT, num_subparts = len(alldata[0]), len(alldata)
    allresources, starts, lengths = np.asarray(data["resources"]), np.asarray(data["starts"]) - 1, np.asarray(data["lengths"])
    learns, forgets, guesses, slips = model["learns"], model["forgets"], model["guesses"], model["slips"]
    prior, num_resources = model["prior"], len(learns)

    # Parameter setup copied from the pure Python run().
    if 'prior' in fixed:
        prior = fixed['prior']
    initial_distn = np.array([1 - prior, prior], dtype=float)
    if 'learns' in fixed:
        learns = learns * (fixed['learns'] < 0) + fixed['learns'] * (fixed['learns'] >= 0)
    if 'forgets' in fixed:
        forgets = forgets * (fixed['forgets'] < 0) + fixed['forgets'] * (fixed['forgets'] >= 0)
    As = np.empty((2, 2 * num_resources))
    As[0, 0::2], As[0, 1::2] = 1 - learns, forgets
    As[1, 0::2], As[1, 1::2] = learns, 1 - forgets
    if 'guesses' in fixed:
        guesses = fixed['guesses'] * (fixed['guesses'] < 0) + fixed['guesses'] * (fixed['guesses'] >= 0)
    if 'slips' in fixed:
        slips = fixed['slips'] * (fixed['slips'] < 0) + fixed['slips'] * (fixed['slips'] >= 0)
    Bn = np.empty((2, 2 * num_subparts))
    Bn[0, 0::2], Bn[0, 1::2] = 1 - guesses, guesses
    Bn[1, 0::2], Bn[1, 1::2] = slips, 1 - slips

    # Transition entries A[i, j] for the resource at each position, as flat arrays.
    resource_index = np.asarray(allresources, dtype=np.int64) - 1
    a00, a01 = As[0, 0::2][resource_index], As[0, 1::2][resource_index]
    a10, a11 = As[1, 0::2][resource_index], As[1, 1::2][resource_index]

    # Emission likelihood of every response at once.
    likelihood0, likelihood1 = np.ones(bigT), np.ones(bigT)
    for n in range(num_subparts):
        row = alldata[n]
        for value, column in ((1, 2 * n), (2, 2 * n + 1)):
            selected = row == value
            if Bn[0, column] != 0:
                likelihood0[selected] *= Bn[0, column]
            if Bn[1, column] != 0:
                likelihood1[selected] *= Bn[1, column]

    # Longest sequences first, so the sequences still running at step t are a prefix.
    order = np.argsort(-lengths, kind="stable")
    sequence_starts, sequence_lengths = starts[order], lengths[order]
    max_length = int(sequence_lengths[0]) if len(sequence_lengths) else 0
    running = np.searchsorted(-sequence_lengths, -np.arange(max_length + 1), side="left")  # running[t] = count with length > t

    # Forward pass: one vectorized step per time index across all running sequences.
    alpha0, alpha1, norms = np.empty(bigT), np.empty(bigT), np.ones(bigT)
    position = sequence_starts
    f0, f1 = initial_distn[0] * likelihood0[position], initial_distn[1] * likelihood1[position]
    norm = f0 + f1
    norms[position], alpha0[position], alpha1[position] = norm, f0 / norm, f1 / norm
    for t in range(1, max_length):
        position = sequence_starts[:running[t]] + t
        previous = position - 1
        p0, p1 = alpha0[previous], alpha1[previous]
        f0 = (a00[previous] * p0 + a01[previous] * p1) * likelihood0[position]
        f1 = (a10[previous] * p0 + a11[previous] * p1) * likelihood1[position]
        norm = f0 + f1
        norms[position], alpha0[position], alpha1[position] = norm, f0 / norm, f1 / norm
    loglike = np.log(norms).sum()

    # Backward pass. pair[i, j] = A[i, j] * alpha_t[j] * gamma_t+1[i] / (A @ alpha_t)[i]
    gamma0, gamma1 = np.empty(bigT), np.empty(bigT)
    pair = np.zeros((4, bigT))
    last = sequence_starts + sequence_lengths - 1
    gamma0[last], gamma1[last] = alpha0[last], alpha1[last]
    for t in range(max_length - 2, -1, -1):
        position = sequence_starts[:running[t + 1]] + t
        q0, q1 = alpha0[position], alpha1[position]
        b00, b01, b10, b11 = a00[position], a01[position], a10[position], a11[position]
        with np.errstate(divide="ignore", invalid="ignore"):
            r0 = gamma0[position + 1] / (b00 * q0 + b01 * q1)
            r1 = gamma1[position + 1] / (b10 * q0 + b11 * q1)
            step = np.nan_to_num(np.array([b00 * q0 * r0, b01 * q1 * r0, b10 * q0 * r1, b11 * q1 * r1]), copy=False)
        pair[:, position] = step
        gamma0[position], gamma1[position] = step[0] + step[2], step[1] + step[3]

    trans_by_resource = np.stack([np.bincount(resource_index, weights=pair[k], minlength=num_resources) for k in range(4)], axis=1).reshape(num_resources, 2, 2)
    all_trans_softcounts = trans_by_resource.transpose(1, 0, 2).reshape(2, 2 * num_resources)
    gamma = np.vstack([gamma0, gamma1])
    alpha = np.vstack([alpha0, alpha1])
    all_emission_softcounts = np.zeros((2, 2 * num_subparts))
    for n in range(num_subparts):
        row = alldata[n]
        for value in (1, 2):
            all_emission_softcounts[:, 2 * n + value - 1] = gamma[:, row == value].sum(axis=1)
    all_initial_softcounts = gamma[:, sequence_starts].sum(axis=1).reshape(2, 1)

    # Output layout copied from the pure Python run().
    result = {}
    result["total_loglike"] = np.array([[loglike]])
    result["all_trans_softcounts"] = np.reshape(all_trans_softcounts.flatten(order='F'), (num_resources, 2, 2), order='C')
    result["all_emission_softcounts"] = np.reshape(all_emission_softcounts.flatten(order='F'), (num_subparts, 2, 2), order='C')
    result["all_initial_softcounts"] = all_initial_softcounts
    result["alpha_out"] = alpha.flatten(order='F').reshape(alpha.shape, order='C')
    return result
