"""Compare batch EM against incremental / stepwise / streaming EM variants on ASSISTments.

Per skill: students split 80/20 by user_id; train each method on the train students,
then score one-step-ahead predictions (filtering) on held-out students.
Event-stream methods consume train attempts in global time order (order_id).
"""
import sys, time, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from bkt_np import *

rng_init = np.random.RandomState(0)


def init_params(seed):
    r = np.random.RandomState(seed)
    # same distribution as pyBKT's random_model_uni (prior/learn uniform, guess/slip <= 0.4 roughly)
    return dict(prior=r.uniform(.05, .6), learn=r.uniform(.05, .4), forget=0.0, guess=r.uniform(.05, .35), slip=r.uniform(.05, .35))


def score(Y, L, p):
    pr, lab = predict(Y, L, p)
    pr = np.clip(pr, 1e-9, 1 - 1e-9)
    ll = np.mean(lab * np.log(pr) + (1 - lab) * np.log(1 - pr))
    auc = roc_auc_score(lab, pr) if 0 < lab.mean() < 1 else np.nan
    return dict(auc=auc, rmse=float(np.sqrt(np.mean((pr - lab) ** 2))), ll=float(ll))


# ---------- methods: each returns (params, info) ----------

def m_batch(train, p0):
    Y, L = pad([s for _, s in train])
    p, hist = batch_em(Y, L, p0, iters=500, tol=1e-6)
    return p, dict(passes=len(hist))


def student_batches(train, size):
    for i in range(0, len(train), size):
        yield train[i : i + size]


def m_naive_accumulate(train, p0, size=50, epochs=1):
    """User-proposed mini-batch EM: E-step on each new batch under current params, add to running totals, M-step."""
    p, S = dict(p0), np.zeros(D)
    for _ in range(epochs):
        for b in student_batches(train, size):
            Y, L = pad([s for _, s in b])
            Sb, _ = fb_counts(Y, L, p)
            S += Sb
            p = m_step(S, p_old=p)
    return p, dict(passes=epochs)


def m_stepwise(train, p0, size=50, alpha=0.7, epochs=1):
    """Stepwise / online EM (Cappé & Moulines 2009; Liang & Klein 2009): S <- (1-eta) S + eta * S_batch/attempts."""
    p, S, k = dict(p0), None, 0
    for _ in range(epochs):
        for b in student_batches(train, size):
            Y, L = pad([s for _, s in b])
            Sb, _ = fb_counts(Y, L, p)
            Sb = Sb / L.sum()
            eta = (k + 2) ** -alpha
            S = Sb if S is None else (1 - eta) * S + eta * Sb
            k += 1
            p = m_step(S, p_old=p)
    return p, dict(passes=epochs)


def m_incremental(train, p0, size=50, epochs=3):
    """Incremental EM (Neal & Hinton 1998): keep each batch's stats, replace them on revisit. Converges to batch EM's fixed point."""
    p = dict(p0)
    batches = list(student_batches(train, size))
    keep = [None] * len(batches)
    S = np.zeros(D)
    for _ in range(epochs):
        for i, b in enumerate(batches):
            Y, L = pad([s for _, s in b])
            Sb, _ = fb_counts(Y, L, p)
            if keep[i] is not None:
                S -= keep[i]
            keep[i] = Sb
            S += Sb
            p = m_step(S, p_old=p)
    return p, dict(passes=epochs)


def m_stream(events, p0, every=1, discount=1.0, pseudo=1.0):
    """Event-level streaming EM with exact forward-only smoothing per student.

    Each event updates that student's (phi, rho); the global statistics hold the sum of
    every student's current smoothed statistics (old contribution swapped for new).
    Parameters are re-estimated every `every` events. Optional discount < 1 forgets old evidence
    (Mongillo & Deneve 2008 style); pseudo-counts keep early M-steps sane.
    """
    p = dict(p0)
    A, B = A_of(p), B_of(p)
    students, contrib = {}, {}
    S = np.zeros(D)
    for n, (u, y) in enumerate(events):
        st = students.get(u)
        if st is None:
            st = students[u] = StudentStream()
            old = 0.0
        else:
            old = contrib[u]
        st.update(int(y), A, B, p["prior"])
        new = st.stats()
        if discount < 1:
            S *= discount
            # discounted contributions are no longer exact per-student sums; just add the delta
        S += new - old
        contrib[u] = new
        if (n + 1) % every == 0:
            p = m_step(S, p_old=p, pseudo=pseudo)
            A, B = A_of(p), B_of(p)
    return m_step(S, p_old=p, pseudo=pseudo), dict(passes=1, state_floats_per_student=2 + 2 * D + D)


def m_filter_only(events, p0, every=1, pseudo=1.0):
    """Approximate online EM using only filtered (not smoothed) one-lag expectations per event.
    The 'use the current mastery probability' shortcut; biased for transitions."""
    p = dict(p0)
    A, B = A_of(p), B_of(p)
    phi = {}
    S = np.zeros(D)
    for n, (u, y) in enumerate(events):
        y = int(y)
        e = np.ones(2) if y < 0 else B[:, y]
        f = phi.get(u)
        if f is None:
            a = np.array([1 - p["prior"], p["prior"]]) * e
            post = a / a.sum()
            S[0:2] += post
        else:
            joint = f[:, None] * A * e[None, :]
            joint /= joint.sum()
            S[2:6] += joint.ravel()
            post = joint.sum(0)
        if y >= 0:
            S[6 + y] += post[0]
            S[8 + y] += post[1]
        phi[u] = post
        if (n + 1) % every == 0:
            p = m_step(S, p_old=p, pseudo=pseudo)
            A, B = A_of(p), B_of(p)
    return m_step(S, p_old=p, pseudo=pseudo), dict(passes=1)


def main():
    df = pd.read_csv("/tmp/claude-0/data/as.csv", encoding="latin", low_memory=False)
    df = df[df["original"] == 1].dropna(subset=["skill_name"])
    df = df.sort_values("order_id")
    top = df["skill_name"].value_counts()
    skills = list(top[top >= 3000].index[: int(sys.argv[1]) if len(sys.argv) > 1 else 12])
    seeds = [0, 1, 2]
    rows = []
    for skill in skills:
        d = df[df["skill_name"] == skill]
        users = d["user_id"].unique()
        test_users = set(users[np.random.RandomState(42).rand(len(users)) < 0.2])
        tr, te = d[~d["user_id"].isin(test_users)], d[d["user_id"].isin(test_users)]
        # students ordered by first appearance (as a stream would deliver them)
        train = [(u, g["correct"].to_numpy()) for u, g in tr.groupby("user_id", sort=False)]
        test = [g["correct"].to_numpy() for _, g in te.groupby("user_id", sort=False)]
        Yt, Lt = pad(test)
        events = list(zip(tr["user_id"].to_numpy(), tr["correct"].to_numpy()))
        for seed in seeds:
            p0 = init_params(seed)
            methods = {
                "batch_em": lambda: m_batch(train, p0),
                "naive_accum_b50": lambda: m_naive_accumulate(train, p0, 50),
                "stepwise_a0.7_b50": lambda: m_stepwise(train, p0, 50, 0.7),
                "stepwise_a0.7_b50_x3": lambda: m_stepwise(train, p0, 50, 0.7, epochs=3),
                "stepwise_a1.0_b50": lambda: m_stepwise(train, p0, 50, 1.0),
                "incremental_b50_x1": lambda: m_incremental(train, p0, 50, 1),
                "incremental_b50_x3": lambda: m_incremental(train, p0, 50, 3),
                "stream_smoothed_every100": lambda: m_stream(events, p0, 100),
                "stream_filter_only_every100": lambda: m_filter_only(events, p0, 100),
            }
            for name, fn in methods.items():
                t = time.perf_counter()
                p, info = fn()
                el = time.perf_counter() - t
                rows.append(dict(skill=skill, seed=seed, method=name, seconds=el, n_train=len(events),
                                 **{k: float(v) for k, v in p.items()}, **score(Yt, Lt, p), **info))
        print(skill, len(events), "train attempts", file=sys.stderr)
    out = pd.DataFrame(rows)
    out.to_csv("/tmp/claude-0/exp/compare_em.csv", index=False)
    base = out[out.method == "batch_em"].set_index(["skill", "seed"])
    out = out.join(base[["ll", "auc", "learn", "guess", "slip", "prior"]], on=["skill", "seed"], rsuffix="_batch")
    for c in ["learn", "guess", "slip", "prior"]:
        out["d_" + c] = (out[c] - out[c + "_batch"]).abs()
    out["d_ll"] = out["ll"] - out["ll_batch"]
    out["d_auc"] = out["auc"] - out["auc_batch"]
    summ = out.groupby("method").agg(heldout_ll=("ll", "mean"), d_ll_vs_batch=("d_ll", "mean"), worst_d_ll=("d_ll", "min"),
                                     auc=("auc", "mean"), d_auc=("d_auc", "mean"),
                                     d_learn=("d_learn", "mean"), d_guess=("d_guess", "mean"), d_slip=("d_slip", "mean"), d_prior=("d_prior", "mean"),
                                     passes=("passes", "mean"), seconds=("seconds", "mean"))
    pd.set_option("display.width", 250)
    print(summ.sort_values("d_ll_vs_batch", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
