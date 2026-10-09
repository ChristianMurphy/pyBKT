"""Time data conversion, EM and prediction on simulated data of a chosen size.

Run from outside the repository root so the installed build is imported:

    python benchmarks/bench.py --rows 1000000 5000000
    python benchmarks/bench.py --rows 1000000 --model multipair --skills 10

Peak memory is the process's high-water mark, so run one size per process
(the default) when comparing memory.
"""
import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

MODEL_TYPES = {
    "default": [False, False, False, False],
    "multilearn": [True, False, False, False],
    "multiprior": [False, True, False, False],
    "multipair": [False, False, True, False],
    "multigs": [False, False, False, True],
}


def simulate(rows, attempts, skills, templates, seed=0):
    """BKT-simulated attempts: rows // attempts students, each answering one skill."""
    rng = np.random.default_rng(seed)
    students = rows // attempts
    known = rng.random(students) < 0.3
    correct = np.empty((students, attempts), dtype=np.int64)
    for t in range(attempts):
        r = rng.random(students)
        correct[:, t] = np.where(known, r >= 0.1, r < 0.2)
        known |= rng.random(students) < 0.15
    n = students * attempts
    return pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(students), attempts),
            "skill_name": np.repeat(rng.integers(0, skills, students), attempts).astype(str),
            "order_id": np.arange(n),
            "template_id": rng.integers(0, templates, n).astype(str),
            "problem_id": rng.integers(0, templates, n).astype(str),
            "correct": correct.ravel(),
        }
    )


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_one(args):
    from pyBKT.fit import EM_fit, predict_onestep
    from pyBKT.generate import random_model_uni
    from pyBKT.util import data_helper

    df = simulate(args.rows, args.attempts, args.skills, args.templates)
    out = {"rows": args.rows, "model": args.model, "skills": args.skills, "mb_input": peak_mb()}

    t = time.perf_counter()
    datas = data_helper.convert_data(df, ".*", model_type=MODEL_TYPES[args.model], defaults={"multipair": "problem_id"})
    out["s_convert"] = time.perf_counter() - t
    out["mb_convert"] = peak_mb()

    em, predict = 0.0, 0.0
    for data in datas.values():
        start = random_model_uni.random_model_uni(
            len(data["resource_names"]), len(data["gs_names"]), rand=np.random.RandomState(0)
        )
        t = time.perf_counter()
        fitted, _ = EM_fit.EM_fit(start, data, tol=-1, maxiter=args.iterations, parallel=True)
        em += time.perf_counter() - t
        t = time.perf_counter()
        predict_onestep.run(fitted, data)
        predict += time.perf_counter() - t
    out["ms_em_iteration"] = 1000 * em / args.iterations
    out["s_predict"] = predict
    out["mb_peak"] = peak_mb()
    print(json.dumps(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[1_000_000])
    parser.add_argument("--attempts", type=int, default=20, help="attempts per student")
    parser.add_argument("--skills", type=int, default=1)
    parser.add_argument("--templates", type=int, default=3)
    parser.add_argument("--model", choices=MODEL_TYPES, default="default")
    parser.add_argument("--iterations", type=int, default=10, help="EM iterations to time")
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single:
        args.rows = args.rows[0]
        return run_one(args)
    # one process per size, so each peak-memory reading is its own
    options = ["--attempts", args.attempts, "--skills", args.skills, "--templates", args.templates,
               "--model", args.model, "--iterations", args.iterations]
    for rows in args.rows:
        subprocess.run([sys.executable, __file__, "--single", "--rows", str(rows)] + list(map(str, options)), check=True)

if __name__ == "__main__":
    main()
