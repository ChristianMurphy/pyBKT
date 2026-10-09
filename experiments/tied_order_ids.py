"""K6: tied order_id values. Works with either pyBKT build (pure Python needs NumPy 1, see #65).

  python tied_order_ids.py

Found on upstream master cc1682e (2026-10-09). convert_data sorts by order_id with pandas' default quicksort
(data_helper.py, added 2021-02-10), then by user_id with a stable mergesort (added 2021-02-11). Rows that share
a (user, order_id) pair keep whatever order quicksort leaves, which depends on the input row order. In ASSISTments
2009, 29.6% of rows are in (user, skill) groups with tied order_ids (several rows per multi-skill problem).
Here the same rows in two input orders give different fitted parameters.
"""
import warnings
import numpy as np
import pandas as pd
from pyBKT.models import Model

warnings.simplefilter("ignore")
rng = np.random.default_rng(1)
rows = []
for u in range(40):
    for t in range(8):
        # each step has two answers with the same order_id (as in a multi-step problem), often one right, one wrong
        rows.append(dict(user_id=f"u{u}", skill_name="s", order_id=t, correct=int(rng.random() < 0.7)))
        rows.append(dict(user_id=f"u{u}", skill_name="s", order_id=t, correct=int(rng.random() < 0.4)))
df = pd.DataFrame(rows)
fits = {}
for label, frame in [("file order", df), ("rows reversed", df.iloc[::-1].reset_index(drop=True))]:
    m = Model(seed=0, num_fits=1)
    m.fit(data=frame.copy())
    fits[label] = m.params().values.ravel()
    print(f"{label:14s}", np.round(fits[label], 6))
print("max parameter difference:", float(np.abs(fits["file order"] - fits["rows reversed"]).max()))
