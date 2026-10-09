import sys, time, numpy as np, pandas as pd
from bkt_np import *
from compare_online import Streamer, simulate_stream, init_params, batch_fit
truth = dict(prior=0.3, learn=0.15, forget=0.0, guess=0.2, slip=0.1)
events = simulate_stream(40000, 12, truth)
p0 = init_params(3)
cps = sorted({int(len(events) * f) for f in (0.03, 0.1, 0.3, 1.0)})
err = lambda p: max(abs(p[k] - truth[k]) for k in ("prior", "learn", "guess", "slip"))
rows = []
for disc in (1.0, 1 - 1e-4, 1 - 3e-5, 1 - 1e-5):
    s = Streamer(p0, 100, discount=disc)
    for n, (u, y) in enumerate(events, 1):
        s.event(u, y)
        if n in cps:
            rows.append(dict(discount=disc, events=n, err=err(m_step(s.S, p_old=s.p))))
# two-pass variant: stream once to get params, then a second streaming pass (fresh student state) starting from them
s = Streamer(p0, 100)
for u, y in events: s.event(u, y)
s2 = Streamer(m_step(s.S, p_old=s.p), 100)
for u, y in events: s2.event(u, y)
rows.append(dict(discount="2 passes", events=len(events), err=err(m_step(s2.S, p_old=s2.p))))
print(pd.DataFrame(rows).pivot(index="events", columns="discount", values="err").round(4).to_string())
