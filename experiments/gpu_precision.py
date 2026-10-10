"""Would a float32 GPU kernel fit BKT correctly? (planning round 11)

WebGPU (WGSL) and Metal have no f64, and consumer NVIDIA GPUs run f64 at 1/32 to 1/64 of f32 speed, so a
practical GPU E-step would compute in f32. This models how such a kernel is written: one thread per
student keeps its own running counts and log-likelihood in f32 along its sequence, then the per-student
partials are reduced across students, either on the GPU in f32 ("f32") or on the host in f64
("f32+f64sum", downloading 11 numbers per student). The reference is np_estep.estep in f64.

  python gpu_precision.py estep            # one E-step on the fixtures: counts and log-likelihood error
  python gpu_precision.py em               # EM fits with pyBKT's stopping rule (|change in ll| < 0.005)
  python gpu_precision.py underflow        # one 992-answer sequence: f32 underflow, and the log-odds fix
Fixtures: /tmp/claude-0/exp/{as_all,synth5m}.npz (make_fixture.py) and the 8 largest ASSISTments skills.
"""
import sys
import numpy as np
from np_estep import estep as estep64


def estep32(data, starts, lengths, prior, learn, forget, guess, slip, host_f64_sum):
    f = np.float32
    order = np.argsort(-lengths, kind="stable")
    Ls = lengths[order]; st = starts[order] - 1
    S, Tmax = len(Ls), int(Ls[0])
    n_active = np.searchsorted(-Ls, -np.arange(1, Tmax + 1), side="right")
    A = np.array([[1 - learn, learn], [forget, 1 - forget]], f)
    Bx = np.array([[1.0, 1.0], [1 - guess, slip], [guess, 1 - slip]], f)    # by code: missing, wrong, right
    pi = np.array([1 - prior, prior], f)
    alpha = np.empty((Tmax, S, 2), f); codes = np.empty((Tmax, S), np.int8)
    ll_s = np.zeros(S, f)                                  # per-student running log-likelihood, f32
    for t in range(Tmax):
        n = n_active[t]
        c = data[st[:n] + t]; codes[t, :n] = c
        a = (pi * Bx[c]) if t == 0 else (alpha[t - 1, :n] @ A) * Bx[c]
        s = a.sum(1, dtype=f)
        alpha[t, :n] = a / s[:, None]
        ll_s[:n] += np.log(s)
    trans_s = np.zeros((S, 2, 2), f); emit_s = np.zeros((S, 3, 2), f); init_s = np.zeros((S, 2), f)
    gamma = np.empty((S, 2), f)
    for t in range(Tmax - 1, -1, -1):
        n = n_active[t]
        ending = Ls[:n] == t + 1
        if t + 1 < Tmax:
            cont = ~ending
            pred = alpha[t, :n] @ A
            with np.errstate(invalid="ignore", divide="ignore"):
                xi = alpha[t, :n, :, None] * A[None] * (gamma[:n] / pred)[:, None, :]
            xi = np.nan_to_num(xi, nan=0.0).astype(f)
            trans_s[:n][cont] += xi[cont]
            g = np.where(cont[:, None], xi.sum(2, dtype=f), alpha[t, :n])
        else:
            g = alpha[t, :n].copy()
        gamma[:n] = g
        emit_s[np.arange(n), codes[t, :n]] += g
        if t == 0:
            init_s[:n] = g
    acc = np.float64 if host_f64_sum else f
    trans = trans_s.sum(0, dtype=acc); emit = emit_s.sum(0, dtype=acc); init = init_s.sum(0, dtype=acc)
    ll = ll_s.sum(dtype=acc)
    return trans.astype(np.float64), emit[1:].astype(np.float64), init.astype(np.float64), float(ll)


MODES = {"f64": lambda *a: estep64(*a),
         "f32": lambda *a: estep32(*a, host_f64_sum=False),
         "f32+f64sum": lambda *a: estep32(*a, host_f64_sum=True)}


def m_step(trans, emit, init):
    return dict(prior=init[1] / init.sum(), learn=trans[0, 1] / trans[0].sum(), forget=0.0,
                guess=emit[1, 0] / emit[:, 0].sum(), slip=emit[0, 1] / emit[:, 1].sum())


def datasets():
    for name in ("as_all", "synth5m"):
        z = np.load(f"/tmp/claude-0/exp/{name}.npz")
        p = dict(prior=float(z["prior"]), learn=float(z["learns"][0]), forget=float(z["forgets"][0]),
                 guess=float(z["guesses"][0]), slip=float(z["slips"][0]))
        yield name, (z["data"][0].astype(np.int8), z["starts"], z["lengths"]), p


def run_estep():
    print("dataset,answers,mode,max_rel_err_counts,ll_f64,ll_abs_err,ll_rel_err")
    for name, (d, s, L), p in datasets():
        args = (d, s, L, p["prior"], p["learn"], p["forget"], p["guess"], p["slip"])
        ref = MODES["f64"](*args)
        rv = np.concatenate([ref[0].ravel(), ref[1].ravel(), ref[2]])
        for mode in ("f32", "f32+f64sum"):
            r = MODES[mode](*args)
            v = np.concatenate([r[0].ravel(), r[1].ravel(), r[2]])
            err = np.max(np.abs(v - rv) / np.abs(rv))
            print(f"{name},{int(L.sum())},{mode},{err:.1e},{ref[3]:.4f},{abs(r[3] - ref[3]):.4f},{abs(r[3] - ref[3]) / abs(ref[3]):.1e}",
                  flush=True)


def fit(mode, data, p, tol=0.005, maxit=200):
    """pyBKT's EM loop: stop when the log-likelihood (as this mode computes it) changes by less than tol."""
    prev, trace = None, []
    for it in range(1, maxit + 1):
        trans, emit, init, ll = MODES[mode](*data, p["prior"], p["learn"], p["forget"], p["guess"], p["slip"])
        trace.append(ll)
        if prev is not None and abs(ll - prev) < tol:
            return p, it, trace
        prev, p = ll, m_step(trans, emit, init)
    return p, maxit, trace


def run_em():
    sys.path.insert(0, ".")
    from optimizers import load_skills
    from compare_em import init_params
    sets = [(n, d, p) for n, d, p in datasets()]
    sets += [(skill, data, None) for skill, data, _ in load_skills(8)]
    print("dataset,answers,mode,iterations,stopped_by_tol,final_ll_f64,max_param_diff_vs_f64,ll_diff_vs_f64_fit")
    for name, data, _ in sets:
        p0 = init_params(0)
        ref_p, ref_it, _ = fit("f64", data, dict(p0))
        ll_of = lambda q: estep64(*data, q["prior"], q["learn"], q["forget"], q["guess"], q["slip"])[3]
        ref_ll = ll_of(ref_p)
        print(f"{name},{int(data[2].sum())},f64,{ref_it},{ref_it < 200},{ref_ll:.4f},0,0", flush=True)
        for mode in ("f32", "f32+f64sum"):
            q, it, _ = fit(mode, data, dict(p0))
            dp = max(abs(q[k] - ref_p[k]) for k in ("prior", "learn", "guess", "slip"))
            qll = ll_of(q)
            print(f"{name},{int(data[2].sum())},{mode},{it},{it < 200},{qll:.4f},{dp:.1e},{qll - ref_ll:.4f}", flush=True)


def run_underflow():
    """The longest sequence of "Conversion of Fraction Decimals Percents" (992 answers), forward pass only:
    f64 and f32 with normalized state probabilities (as the CPU kernels), and f32 with a log-odds state."""
    sys.path.insert(0, ".")
    from optimizers import load_skills
    from compare_em import init_params
    gen = load_skills(3); next(gen); next(gen)
    skill, (codes, starts, lengths), _ = next(gen)
    i = int(np.argmax(lengths)); d = codes[starts[i] - 1: starts[i] - 1 + lengths[i]]
    p = init_params(0)
    for f in (np.float64, np.float32):
        A = np.array([[1 - p["learn"], p["learn"]], [0, 1]], f)
        Bx = np.array([[1, 1], [1 - p["guess"], p["slip"]], [p["guess"], 1 - p["slip"]]], f)
        a = np.array([1 - p["prior"], p["prior"]], f) * Bx[d[0]]; ll = 0.0; zero_at = None
        for t in range(len(d)):
            if t:
                a = (a @ A) * Bx[d[t]]
            s_ = a.sum(dtype=f); a = a / s_; ll += float(np.log(s_))
            if zero_at is None and a[0] == 0:
                zero_at = t
        print(f"{skill}, {len(d)} answers, {f.__name__} normalized state: ll {ll:.5f}; P(unknown) exactly 0 from answer {zero_at}")
    f = np.float32
    learn, guess, slip, prior = f(p["learn"]), f(p["guess"]), f(p["slip"]), f(p["prior"])
    ek = {1: slip, 2: f(1) - slip}; eu = {1: f(1) - guess, 2: guess}
    sig = lambda x: f(1) / (f(1) + np.exp(-x))
    lo = f(np.log(prior) - np.log1p(-prior)); ll = f(0)
    for t, o in enumerate(d):
        if t:
            lo = (lo + np.log1p(learn * np.exp(-lo))) if lo > 0 else np.log(learn + np.exp(lo))
            lo = lo - np.log1p(-learn)
        ll += np.log(sig(lo) * ek[int(o)] + sig(-lo) * eu[int(o)])
        lo = lo + np.log(ek[int(o)]) - np.log(eu[int(o)])
    print(f"{skill}, {len(d)} answers, float32 log-odds state: ll {float(ll):.5f}")


if __name__ == "__main__":
    {"estep": run_estep, "em": run_em, "underflow": run_underflow}[sys.argv[1]]()
