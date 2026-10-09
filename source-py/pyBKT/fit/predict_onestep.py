#########################################
# predict_onestep.py                    #
# predict_onestep                       #
#                                       #
# @author Anirudhan Badrinath           #
# @author Christian Garay               #
# Last edited: 20 March 2020            #
#########################################

import numpy as np
from pyBKT.fit.EM_fit import run as E_step_run

# correct_emission_predictions is a  num_subparts x T array, where element
# (i,t) is predicted probability that answer to subpart i at time t+1 is correct
def run(model, data, parallel = True):

    num_subparts = data["data"].shape[0]  # mmm the first dimension of data represents each subpart?? interesting.
    num_resources = len(model["learns"])

    trans_softcounts = np.zeros((num_resources, 2, 2))
    emission_softcounts = np.zeros((num_subparts, 2, 2))
    init_softcounts = np.zeros((2, 1))

    result = {}
    result['all_trans_softcounts'] = trans_softcounts
    result['all_emission_softcounts'] = emission_softcounts
    result['all_initial_softcounts'] = init_softcounts

    result = E_step_run(data, model, result['all_trans_softcounts'], result['all_emission_softcounts'], result['all_initial_softcounts'], 1, parallel)
    for j in range(num_resources):
        result['all_trans_softcounts'][j] = result['all_trans_softcounts'][j].transpose()
    for j in range(num_subparts):
        result['all_emission_softcounts'][j] = result['all_emission_softcounts'][j].transpose()
    state_predictions = predict_onestep_states(data, model, result['alpha_out'])
    return (correct_predictions(model, data, state_predictions), state_predictions)

def correct_predictions(model, data, state_predictions):
    """P(correct) for each attempt, under the guess and slip of the subpart that attempt used."""
    alldata = data["data"]
    # each attempt is filed under one subpart (a row of data); an attempt with no response uses the first
    subpart = 0 if alldata.shape[0] == 1 else (alldata != 0).argmax(axis = 0)
    guesses = np.asarray(model["guesses"])[subpart]
    not_slips = (1 - np.asarray(model["slips"]))[subpart]
    return guesses * state_predictions[0] + not_slips * state_predictions[1]

def predict_onestep_states(data, model, forward_messages):
    alldata, allresources, starts, lengths, learns, forgets, guesses, slips, prior = \
            data["data"], data["resources"], data["starts"], data["lengths"], \
            model["learns"], model["forgets"], model["guesses"], model["slips"], \
            model["prior"]
    bigT, num_subparts, num_sequences, num_resources = \
            len(alldata[0]), len(alldata), len(starts), len(learns)

    initial_distn = np.array([1 - prior, prior])

    As = np.empty((2, 2 * num_resources))
    interleave(As[0], 1 - learns, forgets)
    interleave(As[1], learns, 1 - forgets)

    fd_temp = np.ravel(forward_messages)

    # outputs
    all_predictions = np.empty((2 * bigT, ))
    for sequence_index in range(num_sequences):
        sequence_start = starts[sequence_index] - 1
        T = lengths[sequence_index]
        forward_messages = fd_temp[2 * sequence_start: 2 * (sequence_start + T)].reshape((2, T), order = 'F')
        predictions = all_predictions[2 * sequence_start: 2 * (sequence_start + T)].reshape((2, T), order = 'F')

        predictions[:, 0] = initial_distn
        for t in range(T - 1):
            resources_temp = allresources[sequence_start+t]
            k = 2 * (resources_temp - 1)
            predictions[:, t + 1] = As[0: 2, k: k + 2].dot(forward_messages[:, t])

    return all_predictions.reshape((2, bigT), order = 'F')

def interleave(m, v1, v2):
        m[0::2], m[1::2] = v1, v2
