"""K8: rows with a missing user id. Run with pyBKT's pure-Python build on NumPy 1 (NumPy 2 hits #65 first):

  PYTHONPATH=<pyBKT>/source-py python nan_user_ids.py      # or with a compiled pyBKT installed

Found on upstream master cc1682e (2026-10-09). convert_data counts lengths with groupby(user_id), which
drops missing ids, but keeps those rows at the end of the data array. The fit ignores them silently
(parameters equal a fit without them), and predict returns invalid values for them.
"""
import warnings
import numpy as np
import pandas as pd
from pyBKT.models import Model

warnings.simplefilter("ignore")
df = pd.DataFrame({"user_id": ["a", "a", np.nan, "b", "b", np.nan, "b"], "skill_name": ["s"] * 7,
                   "order_id": [1, 2, 3, 1, 2, 9, 3], "correct": [1, 0, 1, 1, 1, 0, 0], "template_id": ["t"] * 7})
with_nan = Model(seed=0, num_fits=1)
with_nan.fit(data=df.copy())
without = Model(seed=0, num_fits=1)
without.fit(data=df[df.user_id.notna()].copy())
print("same parameters with and without the missing-id rows:",
      np.allclose(with_nan.params().values.ravel(), without.params().values.ravel()))
print(with_nan.predict(data=df.copy())[["user_id", "order_id", "correct", "correct_predictions", "state_predictions"]])
# Observed, pure Python on NumPy 1: the two missing-id rows get correct_predictions 1.12137 (above 1) and 0.0.
# Observed, compiled build on NumPy 2: both rows get 0.0. Neither is the 0.5 used elsewhere for unknown rows.
