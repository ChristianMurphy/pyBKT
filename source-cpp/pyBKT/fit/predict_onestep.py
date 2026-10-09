import numpy as np
from pyBKT.fit import E_step

# Returns (correct_predictions, state_predictions). correct_predictions[t] is the
# predicted probability that attempt t is answered correctly, given the attempts
# before it; state_predictions is 2 x T, the probability of each knowledge state.
def run(model, data, parallel = True, fixed = {}):
    # the forward pass alone, skipping the backward pass and expected counts
    state_predictions = E_step.predict(data, model, int(parallel), fixed)
    return (correct_predictions(model, data, state_predictions), state_predictions)

def correct_predictions(model, data, state_predictions):
    """P(correct) for each attempt, under the guess and slip of the subpart that attempt used."""
    alldata = data["data"]
    # each attempt is filed under one subpart (a row of data); an attempt with no response uses the first
    subpart = 0 if alldata.shape[0] == 1 else (alldata != 0).argmax(axis = 0)
    guesses = np.asarray(model["guesses"])[subpart]
    not_slips = (1 - np.asarray(model["slips"]))[subpart]
    return guesses * state_predictions[0] + not_slips * state_predictions[1]
