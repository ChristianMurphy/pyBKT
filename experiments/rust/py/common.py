import numpy as np
def load(name):
    z = np.load(f'/tmp/claude-0/exp/{name}.npz')
    return {k: z[k] for k in z.files}
def dm(f):
    d = {k: f[k] for k in ('data','resources','starts','lengths')}
    m = {'prior': float(f['prior']), **{k: f[k] for k in ('learns','forgets','guesses','slips')}}
    return d, m
