"""Online-update experiments.

exp warm  (ASSISTments): batch-fit on the first half of each skill's train attempts (by time),
          then process the second half online; compare to a batch refit on all train data.
exp synth (simulated, known truth): one long time-interleaved stream from a cold start;
          parameter error vs events seen.
"""
import sys, time
import numpy as np, pandas as pd
from bkt_np import *
from compare_em import score, init_params


class Streamer:
    """Exact forward-only smoothed statistics for many students; global S = sum of their current stats."""

    def __init__(self, p, every=100, discount=1.0, update=True):
        self.p, self.every, self.discount, self.update = dict(p), every, discount, update
        self.A, self.B = A_of(p), B_of(p)
        self.st, self.contrib, self.when = {}, {}, {}
        self.S = np.zeros(D)
        self.n = 0

    def event(self, u, y):
        st = self.st.get(u)
        if st is None:
            st = self.st[u] = StudentStream()
            old = 0.0
        else:
            old = self.contrib[u]
        st.update(int(y), self.A, self.B, self.p["prior"])
        new = st.stats()
        if self.discount < 1:
            self.S *= self.discount
            if st is not None and u in self.when:
                # the old contribution has decayed along with S since it was added
                old = old * self.discount ** (self.n + 1 - self.when[u])
            self.when[u] = self.n + 1
        self.S += new - old
        self.contrib[u] = new
        self.n += 1
        if self.update and self.n % self.every == 0:
            self.set(m_step(self.S, p_old=self.p))

    def set(self, p):
        self.p, self.A, self.B = p, A_of(p), B_of(p)


class FilterOnly:
    def __init__(self, p, every=100):
        self.p, self.every = dict(p), every
        self.A, self.B = A_of(p), B_of(p)
        self.phi, self.S, self.n = {}, np.zeros(D), 0

    def event(self, u, y):
        y = int(y)
        e = np.ones(2) if y < 0 else self.B[:, y]
        f = self.phi.get(u)
        if f is None:
            a = np.array([1 - self.p["prior"], self.p["prior"]]) * e
            post = a / a.sum()
            self.S[0:2] += post
        else:
            joint = f[:, None] * self.A * e[None, :]
            joint /= joint.sum()
            self.S[2:6] += joint.ravel()
            post = joint.sum(0)
        if y >= 0:
            self.S[6 + y] += post[0]
            self.S[8 + y] += post[1]
        self.phi[u] = post
        self.n += 1
        if self.n % self.every == 0:
            self.p = m_step(self.S, p_old=self.p)
            self.A, self.B = A_of(self.p), B_of(self.p)


def seqs_of(events):
    by = {}
    for u, y in events:
        by.setdefault(u, []).append(y)
    return [np.array(v) for v in by.values()]


def batch_fit(events, p0, iters=500):
    Y, L = pad(seqs_of(events))
    p, hist = batch_em(Y, L, p0, iters=iters, tol=1e-6)
    return p, len(hist)


def exp_warm(n_skills):
    df = pd.read_csv("/tmp/claude-0/data/as.csv", encoding="latin", low_memory=False)
    df = df[df["original"] == 1].dropna(subset=["skill_name"]).sort_values("order_id")
    vc = df["skill_name"].value_counts()
    skills = list(vc[vc >= 3000].index[:n_skills])
    rows = []
    for skill in skills:
        d = df[df["skill_name"] == skill]
        users = d["user_id"].unique()
        test_users = set(users[np.random.RandomState(42).rand(len(users)) < 0.2])
        tr, te = d[~d["user_id"].isin(test_users)], d[d["user_id"].isin(test_users)]
        Yt, Lt = pad([g["correct"].to_numpy() for _, g in te.groupby("user_id", sort=False)])
        events = list(zip(tr["user_id"].to_numpy(), tr["correct"].to_numpy()))
        half = len(events) // 2
        first, second = events[:half], events[half:]
        p0 = init_params(0)
        t = time.perf_counter(); p_half, it_half = batch_fit(first, p0); t_half = time.perf_counter() - t
        t = time.perf_counter(); p_all, it_all = batch_fit(events, p_half); t_all = time.perf_counter() - t
        res = {"refit_batch_all (warm from half)": (p_all, it_all * len(events), t_all), "frozen_half": (p_half, 0, 0.0)}

        # a few EM iterations on all data from the half model: the cheap periodic refit
        for k in (1, 3, 10):
            Y, L = pad(seqs_of(events))
            p = p_half
            t = time.perf_counter()
            for _ in range(k):
                p = m_step(fb_counts(Y, L, p)[0], p_old=p)
            res[f"periodic_refit_{k}iter"] = (p, k * len(events), time.perf_counter() - t)

        for every in (100, 1000):
            for disc in (1.0,):
                s = Streamer(p_half, every=every, update=False)
                for u, y in first:          # rebuild retained per-student state under the half model
                    s.event(u, y)
                s.update = True
                t = time.perf_counter()
                for u, y in second:
                    s.event(u, y)
                res[f"stream_smoothed_every{every}"] = (m_step(s.S, p_old=s.p), len(second), time.perf_counter() - t)
        # naive mini-batch accumulation: stats of the first half (under p_half) + new events' stats
        # computed once under the parameters current when they arrive, never recomputed.
        # Approximated at event granularity with the same smoother but frozen past contributions:
        s = Streamer(p_half, every=100, update=False)
        for u, y in first:
            s.event(u, y)
        frozen_S = s.S.copy()
        s2 = Streamer(p_half, every=100)
        s2.S = frozen_S.copy()
        s2.st, s2.contrib = s.st, {u: s.contrib[u] for u in s.contrib}
        t = time.perf_counter()
        for u, y in second:
            s2.event(u, y)
        res["stream_smoothed_retained_every100(same as above)"] = (m_step(s2.S, p_old=s2.p), len(second), time.perf_counter() - t)

        f = FilterOnly(p_half, every=100)
        # filter-only needs no retained history beyond phi; replay first half without updates for state
        f.every = 10**12
        for u, y in first:
            f.event(u, y)
        f.every = 100
        t = time.perf_counter()
        for u, y in second:
            f.event(u, y)
        res["filter_only_every100"] = (m_step(f.S, p_old=f.p), len(second), time.perf_counter() - t)

        for name, (p, work, secs) in res.items():
            rows.append(dict(skill=skill, method=name, attempts_processed=work, seconds=secs, **{k: float(v) for k, v in p.items()}, **score(Yt, Lt, p)))
        print(skill, len(events), file=sys.stderr)
    out = pd.DataFrame(rows)
    ref = out[out.method.str.startswith("refit_batch_all")].set_index("skill")
    out = out.join(ref[["ll", "auc", "learn", "guess", "slip", "prior"]], on="skill", rsuffix="_ref")
    for c in ["learn", "guess", "slip", "prior"]:
        out["d_" + c] = (out[c] - out[c + "_ref"]).abs()
    out["d_ll"], out["d_auc"] = out["ll"] - out["ll_ref"], out["auc"] - out["auc_ref"]
    out.to_csv("/tmp/claude-0/exp/compare_online_warm.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.groupby("method").agg(heldout_ll=("ll", "mean"), d_ll=("d_ll", "mean"), worst_d_ll=("d_ll", "min"), d_auc=("d_auc", "mean"),
                                    d_learn=("d_learn", "mean"), d_guess=("d_guess", "mean"), d_slip=("d_slip", "mean"), d_prior=("d_prior", "mean"),
                                    work=("attempts_processed", "mean"), seconds=("seconds", "mean")).sort_values("d_ll", ascending=False).round(4).to_string())


def simulate_stream(n_students, mean_len, truth, seed=0):
    """Students arrive over time; attempts interleave. Returns events in time order."""
    rng = np.random.default_rng(seed)
    ev = []
    for s in range(n_students):
        L = max(1, rng.poisson(mean_len))
        known = rng.random() < truth["prior"]
        t0 = rng.random()
        for k in range(L):
            correct = (rng.random() >= truth["slip"]) if known else (rng.random() < truth["guess"])
            ev.append((t0 + 0.02 * k + 0.001 * rng.random(), s, int(correct)))
            if not known:
                known = rng.random() < truth["learn"]
    ev.sort()
    return [(u, y) for _, u, y in ev]


def exp_synth(n_students):
    truth = dict(prior=0.3, learn=0.15, forget=0.0, guess=0.2, slip=0.1)
    events = simulate_stream(n_students, 12, truth)
    p0 = init_params(3)
    checkpoints = sorted({int(len(events) * f) for f in (0.01, 0.03, 0.1, 0.3, 1.0)})
    rows = []
    def err(p):
        return max(abs(p[k] - truth[k]) for k in ("prior", "learn", "guess", "slip"))
    for name, mk in [("stream_smoothed_every100", lambda: Streamer(p0, 100)), ("stream_smoothed_every1000", lambda: Streamer(p0, 1000)),
                     ("filter_only_every100", lambda: FilterOnly(p0, 100))]:
        s = mk(); t = time.perf_counter()
        for n, (u, y) in enumerate(events, 1):
            s.event(u, y)
            if n in checkpoints:
                p = m_step(s.S, p_old=s.p)
                rows.append(dict(method=name, events=n, max_abs_err=err(p), seconds=time.perf_counter() - t, **{k: round(float(v), 4) for k, v in p.items()}))
        print(name, file=sys.stderr)
    for n in checkpoints:
        t = time.perf_counter()
        p, it = batch_fit(events[:n], p0)
        rows.append(dict(method="batch_em_on_prefix", events=n, max_abs_err=err(p), seconds=time.perf_counter() - t, **{k: round(float(v), 4) for k, v in p.items()}))
    out = pd.DataFrame(rows)
    out.to_csv("/tmp/claude-0/exp/compare_online_synth.csv", index=False)
    pd.set_option("display.width", 250)
    print("truth", truth, "init", {k: round(v, 3) for k, v in p0.items()})
    print(out.pivot(index="events", columns="method", values="max_abs_err").round(4).to_string())
    print(out[out.events == max(checkpoints)].round(4).to_string())


if __name__ == "__main__":
    {"warm": exp_warm, "synth": exp_synth}[sys.argv[1]](int(sys.argv[2]))
