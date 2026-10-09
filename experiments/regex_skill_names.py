"""K5: skill names containing regex characters. Works with either pyBKT build (pure Python needs NumPy 1, see #65).

  python regex_skill_names.py

Found on upstream master cc1682e (2026-10-09). `skills` is a regex when given as a string (default '.*'), and a
list of names is joined with '|' without escaping (data_helper.py, convert_data). After fit, model.skills holds the
list of fitted names, so predict matches every skill against an unescaped regex.
Observed: with the default, the skill is fitted but every one of its rows is predicted at the 0.5 placeholder;
with skills=[...names...], fit skips the skill silently.
The fix should escape names only when they come as a list; a string stays a regex (documented API).
"""
import warnings
import numpy as np
import pandas as pd
from pyBKT.models import Model

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
rows = [dict(user_id=f"u{u}", skill_name=s, order_id=t, correct=int(rng.random() < 0.6))
        for s in ["plain", "Order of Operations +,-,/,* () positive reals"] for u in range(30) for t in range(6)]
df = pd.DataFrame(rows)
for label, kw in [("default skills='.*'", {}), ("skills=[names]", {"skills": list(df.skill_name.unique())})]:
    m = Model(seed=0, num_fits=1)
    m.fit(data=df.copy(), **kw)
    p = m.predict(data=df.copy())
    at_half = p.groupby("skill_name").correct_predictions.apply(lambda s: int((s == 0.5).sum())).to_dict()
    print(f"{label}: fitted {sorted(m.fit_model)}; rows predicted at exactly 0.5 per skill: {at_half}")
