"""K1: the compiled E-step returns the log-likelihood as an integer. Needs the compiled pyBKT build.

  python integer_loglike.py

Found on upstream master cc1682e (2026-10-09): E_step.cpp returns PyLong_FromLong(*total_loglike). Before the Boost
removal (commit dba0ddb, 2021-04-14) it returned the double. EM_fit's stopping rule compares consecutive values, so
it compares whole numbers: EM stops once two iterations round to the same integer, and num_fits picks the first of
restarts whose log-likelihoods round alike. On merged ASSISTments EM stops at 24 iterations; comparing floats takes 47.
"""
import warnings
import numpy as np
import pandas as pd
from pyBKT.fit import EM_fit, E_step
from pyBKT.generate import random_model_uni
from pyBKT.util import data_helper

warnings.simplefilter("ignore")
rng = np.random.default_rng(2)
rows = [dict(user_id=f"u{u}", skill_name="s", order_id=t, correct=int(rng.random() < 0.3 + 0.08 * t))
        for u in range(300) for t in range(8)]
data = data_helper.convert_data(pd.DataFrame(rows), "s")["s"]
np.random.seed(0)                       # random_model_uni draws from NumPy's global generator
start = random_model_uni.random_model_uni(1, 1)
result = E_step.run(data, start, 1, 0, {})   # (data, model, num_outputs, parallel, fixed)
print("total_loglike:", result["total_loglike"], type(result["total_loglike"]).__name__)
fitted, trace = EM_fit.EM_fit(start, data, tol=0.005, maxiter=100, parallel=False)
values = trace[trace != 0].ravel()
print("EM log-likelihood trace:", values.tolist())
print("iterations:", len(values), "| all whole numbers:", bool(np.all(values == np.round(values))))
