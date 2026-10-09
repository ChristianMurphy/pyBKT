"""Live inference: update P(known) for (student, skill) as each answer arrives, and emit the
prediction for the next answer. Exact BKT forward filtering with fixed parameters.
Compares: per-event Python (floats, dict state), per-event NumPy (small arrays), and batched
NumPy over a micro-batch of events (gather/update/scatter on a state array; events of the same
student within a batch are applied in rounds so order is preserved). Checks all agree with
pyBKT's predict over the full history."""
import time, numpy as np, pandas as pd
from pyBKT.models import Model

df = pd.read_csv("/tmp/claude-0/data/as.csv", low_memory=False, encoding="latin")
df = df[df["original"] == 1].dropna(subset=["skill_name"]).sort_values("order_id").reset_index(drop=True)
m = Model(seed=0, num_fits=1, parallel=False); m.fit(data=df.copy())
ref = m.predict(data=df.copy()).loc[df.index, "correct_predictions"].to_numpy()

skills = {s: i for i, s in enumerate(m.fit_model)}
P = np.array([[m.fit_model[s]["prior"], m.fit_model[s]["learns"][0], m.fit_model[s]["forgets"][0],
               m.fit_model[s]["guesses"][0], m.fit_model[s]["slips"][0]] for s in skills])
sk = df["skill_name"].map(skills).to_numpy(); us = df["user_id"].to_numpy(); y = df["correct"].to_numpy()
N = len(df)

def py_events():
    state, out = {}, np.empty(N)
    for i in range(N):
        s = sk[i]; key = (us[i], s)
        prior, learn, forget, guess, slip = P[s]
        k = state.get(key, prior)
        out[i] = k * (1 - slip) + (1 - k) * guess
        # normalise the two state weights; 1 - P(correct) cancels badly when k is near 1
        if y[i] == 1: a1, a0 = k * (1 - slip), (1 - k) * guess
        else: a1, a0 = k * slip, (1 - k) * (1 - guess)
        post = a1 / (a1 + a0)
        state[key] = post * (1 - forget) + (1 - post) * learn
    return out

def np_batched(batch=4096):
    keys, sid = np.unique(np.stack([us, sk], 1), axis=0, return_inverse=True)
    sid = sid.ravel()
    state = P[keys[:, 1], 0].copy()
    out = np.empty(N)
    for b in range(0, N, batch):
        idx = np.arange(b, min(b + batch, N))
        while len(idx):
            # first occurrence of each student in the remaining batch goes this round (keeps per-student order)
            _, first = np.unique(sid[idx], return_index=True)
            r = idx[np.sort(first)]
            s, k = sk[r], state[sid[r]]
            g, sl, le, fo = P[s, 3], P[s, 4], P[s, 1], P[s, 2]
            pc = k * (1 - sl) + (1 - k) * g
            out[r] = pc
            a1 = np.where(y[r] == 1, k * (1 - sl), k * sl)
            a0 = np.where(y[r] == 1, (1 - k) * g, (1 - k) * (1 - g))
            post = a1 / (a1 + a0)
            state[sid[r]] = post * (1 - fo) + (1 - post) * le
            idx = np.setdiff1d(idx, r, assume_unique=True)
    return out

for name, fn in [("python per-event", py_events), ("numpy batched x4096", np_batched)]:
    t = time.perf_counter(); out = fn(); el = time.perf_counter() - t
    print(f"{name:22s} {el:.2f}s  {el/N*1e9:6.0f} ns/event  {N/el/1e6:.2f} M events/s  max|diff vs pyBKT predict| {np.max(np.abs(out-ref)):.2e}")
