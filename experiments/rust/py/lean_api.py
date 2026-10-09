"""Thin helpers turning bkt_lean's raw outputs into numpy arrays (zero copy for alpha)."""
import numpy as np
import bkt_lean

def as_arrays(out, R, K, N, out_arr=None):
    tr, em, ini, ll, al = out
    tr = np.array(tr).reshape(R, 2, 2); em = np.array(em).reshape(K, 2, 2); ini = np.array(ini).reshape(2, 1)
    if al is not None and not isinstance(al, np.ndarray):
        al = np.frombuffer(al, dtype=np.float64).reshape(2, N)
    return tr, em, ini, ll, al

def e_step(data, resources, starts, lengths, prior, learns, forgets, guesses, slips, alpha='bytearray', **kw):
    K, N = data.shape if data.ndim == 2 else (1, data.size)
    out = None
    if alpha == 'numpy':
        out = kw.pop('out', None)
        if out is None:
            out = np.empty((2, N))
    o = bkt_lean.e_step(data, resources, starts, lengths, prior, learns, forgets, guesses, slips, alpha=alpha, out=out, **kw)
    return as_arrays(o, len(learns), K, N)

def predict(data, resources, starts, lengths, prior, learns, forgets, guesses, slips, out_mode='bytearray', **kw):
    N = data.shape[-1]
    if out_mode == 'numpy':
        out = kw.pop('out', None)
        if out is None:
            out = np.empty((2, N))
        return bkt_lean.predict(data, resources, starts, lengths, prior, learns, forgets, guesses, slips, out_mode='numpy', out=out, **kw)
    return np.frombuffer(bkt_lean.predict(data, resources, starts, lengths, prior, learns, forgets, guesses, slips, out_mode=out_mode, **kw), dtype=np.float64).reshape(2, N)
