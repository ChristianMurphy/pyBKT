"""Which optimizer for BKT? EM vs SQUAREM vs L-BFGS vs Nelder-Mead vs differential evolution (planning round 8).

Cost is counted in E-step passes over the data, which is what every method pays per likelihood (and, for
gradient methods, per gradient) evaluation in any language: optimizer overhead is negligible next to a pass.

The gradient comes free with the E-step. For an HMM, Fisher's identity gives the gradient of the
log-likelihood as the gradient of EM's expected complete-data log-likelihood at the current parameters,
so the expected counts S from one forward-backward pass give, in logit coordinates u = logit(p):
    d ll / d u_prior = n(known at t=1)            - prior * n(first answers)
    d ll / d u_learn = n(unknown -> known)        - learn * n(transitions from unknown)
    d ll / d u_guess = n(correct while unknown)   - guess * n(answers while unknown)
    d ll / d u_slip  = n(incorrect while known)   - slip  * n(answers while known)
(forget fixed at 0, as pyBKT's default). check_gradient() verifies this against finite differences.

  python optimizers.py check            # gradient self-check on one skill
  python optimizers.py run N [--de]     # N skills with >= 1,000 answers (as in squarem_tol.py); --de adds DE
  python optimizers.py extra N          # full BFGS and per-answer-scaled L-BFGS, scored against run's results
  python optimizers.py bounded N        # EM and L-BFGS-B with guess, slip <= 0.5
Writes results/optimizers.csv (one row per skill x start x method) and prints a summary.
"""
import sys, time
import numpy as np, pandas as pd
from scipy import optimize
from bkt_np import pad, fb_counts, m_step
from np_estep import estep
from squarem import squarem as squarem_fit, to_vec, to_p, KEYS
from compare_em import init_params


class Evals:
    """One E-step pass = np_estep.estep (the vectorized kernel planned for pyBKT's pure-Python build, C1),
    rearranged into bkt_np's count layout. Counts passes and records the log-likelihood after each one."""

    def __init__(self, data):
        self.codes, self.starts, self.lengths = data
        self.n, self.trace = 0, []

    def counts(self, p):
        trans, emit, init, ll = estep(self.codes, self.starts, self.lengths, p["prior"], p["learn"], p["forget"],
                                      p["guess"], p["slip"])
        return np.concatenate([init, trans.ravel(), emit.T.ravel()]), float(ll)   # emit is (answer, state)

    def em(self, p):
        S, ll = self.counts(p)
        self.n += 1
        self.trace.append(ll)
        return m_step(S, p_old=p), ll

    def ll_grad(self, v):
        p = to_p(v)
        S, ll = self.counts(p)
        self.n += 1
        self.trace.append(ll)
        x = np.array([p[k] for k in KEYS])
        g = np.array([S[1] - x[0] * (S[0] + S[1]),        # prior
                      S[3] - x[1] * (S[2] + S[3]),        # learn: trans[0, 1] vs trans[0, :]
                      S[7] - x[2] * (S[6] + S[7]),        # guess: emit[unknown, correct]
                      S[8] - x[3] * (S[8] + S[9])])       # slip:  emit[known, incorrect]
        return ll, g


def em_fit(c, p, tol, maxe):
    prev = -np.inf
    while c.n < maxe:
        q, ll = c.em(p)
        if abs(ll - prev) < tol:
            return p, ll
        p, prev = q, ll
    return p, ll


def lbfgs_fit(c, p, tol, maxe):
    f = lambda v: tuple(-a for a in c.ll_grad(v))
    r = optimize.minimize(f, to_vec(p), jac=True, method="L-BFGS-B",
                          options=dict(maxfun=maxe, ftol=tol / 1e4, gtol=1e-8, maxls=40))
    return to_p(r.x), -r.fun


def bfgs_fit(c, p, tol, maxe):
    """Full BFGS (a 4 x 4 inverse-Hessian estimate) in logit space, objective scaled per answer."""
    N = float(c.lengths.sum())
    f = lambda v: tuple(-a / N for a in c.ll_grad(v))
    r = optimize.minimize(f, to_vec(p), jac=True, method="BFGS", options=dict(maxiter=maxe, gtol=1e-9))
    return to_p(r.x), -r.fun * N


def lbfgs_scaled_fit(c, p, tol, maxe):
    """L-BFGS as lbfgs_fit, but with the objective scaled per answer (the fix found for the bounded run)."""
    N = float(c.lengths.sum())
    f = lambda v: tuple(-a / N for a in c.ll_grad(v))
    r = optimize.minimize(f, to_vec(p), jac=True, method="L-BFGS-B",
                          options=dict(maxfun=maxe, ftol=1e-15, gtol=1e-10, maxls=40))
    return to_p(r.x), -r.fun * N


def hybrid_fit(c, p, tol, maxe, em_steps=5):
    for _ in range(em_steps):
        p, _ = c.em(p)
    return lbfgs_fit(c, p, tol, maxe - em_steps)


def lbfgs_bounded_fit(c, p, tol, maxe, cap=0.5):
    """L-BFGS-B with guess and slip bounded at `cap` (logit 0 for 0.5): the plausibility constraint as a box."""
    f = lambda v: tuple(-a for a in c.ll_grad(v))
    lim = float(np.log(cap) - np.log1p(-cap))
    v0 = np.minimum(to_vec(p), [7, 7, lim, lim])
    r = optimize.minimize(f, v0, jac=True, method="L-BFGS-B", bounds=[(-7, 7), (-7, 7), (-7, lim), (-7, lim)],
                          options=dict(maxfun=maxe, ftol=tol / 1e4, gtol=1e-8, maxls=40))
    return to_p(r.x), -r.fun


def em_projected_fit(c, p, tol, maxe, cap=0.5):
    """EM whose M-step result is clipped to guess, slip <= cap (a projected M-step)."""
    prev = -np.inf
    while c.n < maxe:
        q, ll = c.em(p)
        q = dict(q, guess=min(q["guess"], cap), slip=min(q["slip"], cap))
        if abs(ll - prev) < tol:
            return p, ll
        p, prev = q, ll
    return p, ll


def nelder_mead_fit(c, p, tol, maxe):
    f = lambda v: -c.ll_grad(v)[0]
    r = optimize.minimize(f, to_vec(p), method="Nelder-Mead", options=dict(maxfev=maxe, xatol=1e-6, fatol=tol))
    return to_p(r.x), -r.fun


def de_fit(c, seed, maxe):
    """Differential evolution over the logit box [-7, 7]^4 (probabilities 0.001 to 0.999), 40 candidates per
    generation as in the Basin example; no local polish, so the count is DE's own."""
    f = lambda v: -c.ll_grad(v)[0]
    r = optimize.differential_evolution(f, [(-7, 7)] * 4, popsize=10, maxiter=max(1, maxe // 40 - 1), tol=1e-10,
                                        polish=False, seed=seed, init="latinhypercube")
    return to_p(r.x), -r.fun


def load_skills(n):
    df = pd.read_csv("/tmp/claude-0/data/as.csv", encoding="latin", low_memory=False)
    df = df[df["original"] == 1].dropna(subset=["skill_name"]).sort_values("order_id")
    vc = df.skill_name.value_counts()
    for skill in list(vc[vc >= 1000].index[:n]):
        d = df[df.skill_name == skill]
        seqs = [g.correct.to_numpy() for _, g in d.groupby("user_id", sort=False)]
        lengths = np.array([len(q) for q in seqs], np.int64)
        starts = np.concatenate([[1], 1 + np.cumsum(lengths[:-1])]).astype(np.int64)     # pyBKT: 1-based
        codes = (np.concatenate(seqs) + 1).astype(np.int8)                                 # 1 wrong, 2 right
        yield skill, (codes, starts, lengths), pad(seqs)


def check_gradient():
    skill, data, (Y, L) = next(load_skills(1))
    c = Evals(data)
    p = init_params(0)
    S_np, ll_np = c.counts(p); S_ref, ll_ref = fb_counts(Y, L, p)
    print(f"{skill}: np_estep vs bkt_np.fb_counts, max rel diff of counts "
          f"{np.max(np.abs(S_np - S_ref) / np.maximum(np.abs(S_ref), 1e-12)):.1e}, log-likelihood diff {abs(ll_np - ll_ref):.1e}")
    for seed in range(3):
        v = to_vec(init_params(seed))
        ll, g = c.ll_grad(v)
        fd = np.array([(c.ll_grad(v + h)[0] - c.ll_grad(v - h)[0]) / 2e-6 for h in np.eye(4) * 1e-6])
        print(f"{skill}, start {seed}: analytic {np.round(g, 4)}  finite-diff {np.round(fd, 4)}  "
              f"max rel err {np.max(np.abs(g - fd) / np.maximum(np.abs(fd), 1e-9)):.1e}")


def first_within(trace, target, delta):
    """E-step passes until the log-likelihood first gets within delta of target (None if never)."""
    for i, ll in enumerate(trace, 1):
        if ll >= target - delta:
            return i
    return None


CAP, DE_CAP = 1500, 2000     # passes; the "passes to reach within delta of the best" measures don't need the tail


def run(n, with_de):
    rows = []
    for skill, data, _ in load_skills(n):
        t_skill = time.time()
        for seed in range(3):
            p0 = init_params(seed)
            methods = [("em", lambda c: em_fit(c, p0, 1e-8, CAP)),
                       ("squarem", lambda c: squarem_fit(c, p0, tol=1e-8, maxe=CAP)),
                       ("lbfgs", lambda c: lbfgs_fit(c, p0, 1e-8, CAP)),
                       ("em5+lbfgs", lambda c: hybrid_fit(c, p0, 1e-8, CAP)),
                       ("nelder_mead", lambda c: nelder_mead_fit(c, p0, 1e-8, CAP))]
            if with_de and seed == 0:
                methods.append(("de", lambda c: de_fit(c, seed, DE_CAP)))
            for name, fn in methods:
                c = Evals(data)
                t = time.perf_counter(); p, ll = fn(c); dt = time.perf_counter() - t
                print(f"  {skill[:30]} start {seed} {name}: {c.n} passes, ll {ll:.3f}, {dt:.0f}s", file=sys.stderr, flush=True)
                rows.append(dict(skill=skill, answers=int(data[2].sum()), seed=seed, method=name, esteps=c.n, ll=ll,
                                 hit_cap=c.n >= (DE_CAP if name == "de" else CAP) - 2,
                                 best_seen=max(c.trace), seconds=dt, trace=c.trace, **{k: p[k] for k in KEYS}))
        print(f"{skill}: {time.time() - t_skill:.0f}s", file=sys.stderr, flush=True)
    o = pd.DataFrame(rows)
    o["skill_best"] = o.groupby("skill").best_seen.transform("max")
    o["start_best"] = o.groupby(["skill", "seed"]).best_seen.transform("max")
    for d in (0.005, 1e-4):
        o[f"to_skill_best_{d}"] = [first_within(t, b, d) for t, b in zip(o.trace, o.skill_best)]
        o[f"to_start_best_{d}"] = [first_within(t, b, d) for t, b in zip(o.trace, o.start_best)]
    o["gap_to_skill_best"] = o.skill_best - o.best_seen
    o["plausible"] = (o.guess < 0.5) & (o.slip < 0.5)
    o.drop(columns="trace").to_csv("results/optimizers.csv", index=False)
    g = o.groupby("method")
    print(pd.DataFrame({
        "runs": g.size(),
        "total_esteps_to_converge": g.esteps.sum(),
        "median_esteps_to_start_best(0.005)": g["to_start_best_0.005"].median(),
        "median_esteps_to_start_best(1e-4)": g["to_start_best_0.0001"].median(),
        "never_within_1e-4_of_start_best": g["to_start_best_0.0001"].apply(lambda s: int(s.isna().sum())),
        "median_gap_to_skill_best": g.gap_to_skill_best.median(),
        "runs_within_0.005_of_skill_best": g.gap_to_skill_best.apply(lambda s: int((s <= 0.005).sum())),
        "plausible": g.plausible.sum(),
        "hit_cap": g.hit_cap.sum(),
    }).round(4).to_string())


def run_extra(n):
    """BFGS and scaled L-BFGS on the same skills and starts; scored against the per-start best in
    results/optimizers.csv together with these runs."""
    ref = pd.read_csv("results/optimizers.csv")
    rows = []
    for skill, data, _ in load_skills(n):
        for seed in range(3):
            p0 = init_params(seed)
            for name, fn in (("bfgs", lambda c: bfgs_fit(c, p0, 1e-8, CAP)),
                             ("lbfgs_scaled", lambda c: lbfgs_scaled_fit(c, p0, 1e-8, CAP))):
                c = Evals(data)
                p, ll = fn(c)
                rows.append(dict(skill=skill, seed=seed, method=name, esteps=c.n, ll=ll, best_seen=max(c.trace),
                                 trace=c.trace, **{k: p[k] for k in KEYS}))
        print(skill, file=sys.stderr, flush=True)
    o = pd.DataFrame(rows)
    start_best = pd.concat([ref[["skill", "seed", "best_seen"]], o[["skill", "seed", "best_seen"]]]
                           ).groupby(["skill", "seed"]).best_seen.max()
    o["start_best"] = [start_best[(a, b)] for a, b in zip(o.skill, o.seed)]
    o["to_start_best_0.005"] = [first_within(t, b, 0.005) for t, b in zip(o.trace, o.start_best)]
    o["gap_to_start_best"] = o.start_best - o.best_seen
    o.drop(columns="trace").to_csv("results/optimizers_extra.csv", index=False)
    old = ref[ref.method.isin(["em", "squarem", "lbfgs", "em5+lbfgs"])]
    both = pd.concat([old[["method", "esteps", "to_start_best_0.005"]], o[["method", "esteps", "to_start_best_0.005"]]])
    g = both.groupby("method")
    print(pd.DataFrame({"runs": g.size(), "total_passes_to_converge": g.esteps.sum(),
                        "median_passes_to_within_0.005": g["to_start_best_0.005"].median(),
                        "never_within_0.005": g["to_start_best_0.005"].apply(lambda s: int(s.isna().sum()))}).to_string())
    print("\nnew runs farther than 0.005 from their start's best:")
    print(o[o.gap_to_start_best > 0.005][["skill", "seed", "method", "esteps", "gap_to_start_best"]].to_string(index=False))


def run_bounded(n):
    """Same skills and starts as run(): EM and L-BFGS with guess, slip <= 0.5. Compared against the unconstrained
    rows in results/optimizers.csv (run that first)."""
    ref = pd.read_csv("results/optimizers.csv")
    rows = []
    for skill, data, _ in load_skills(n):
        for seed in range(3):
            p0 = init_params(seed)
            for name, fn in (("em_projected", lambda c: em_projected_fit(c, p0, 1e-8, CAP)),
                             ("lbfgs_bounded", lambda c: lbfgs_bounded_fit(c, p0, 1e-8, CAP))):
                c = Evals(data)
                p, ll = fn(c)
                rows.append(dict(skill=skill, seed=seed, method=name, esteps=c.n, ll=ll, **{k: p[k] for k in KEYS}))
        print(skill, file=sys.stderr, flush=True)
    o = pd.DataFrame(rows)
    o["at_bound"] = (o.guess > 0.4999) | (o.slip > 0.4999)
    best = ref.groupby("skill").best_seen.max()
    plaus = ref[ref.plausible].groupby("skill").best_seen.max()
    o["below_unconstrained_best"] = best.reindex(o.skill).values - o.ll
    o["vs_best_plausible_found"] = o.ll - plaus.reindex(o.skill).values
    o.to_csv("results/optimizers_bounded.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.round({"ll": 3, "prior": 3, "learn": 3, "guess": 3, "slip": 3, "below_unconstrained_best": 2,
                   "vs_best_plausible_found": 2}).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check_gradient()
    elif sys.argv[1] == "extra":
        run_extra(int(sys.argv[2]))
    elif sys.argv[1] == "bounded":
        run_bounded(int(sys.argv[2]))
    else:
        run(int(sys.argv[2]), "--de" in sys.argv)
