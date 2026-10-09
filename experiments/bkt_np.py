"""Small NumPy reference implementation of standard BKT (one skill, one template),
used to prototype batch, incremental, stepwise and streaming EM.

Observations: 1 correct, 0 incorrect, -1 missing (likelihood 1, like pyBKT's 0 code).
States: 0 unknown, 1 known. Parameters: dict prior, learn, forget, guess, slip.

Sufficient statistics vector (length 10):
  [0:2]  init      E[x_1 = k]
  [2:6]  trans     E[#(x_{t-1}=i, x_t=j)] flattened (i, j)
  [6:10] emit      E[#(x_t=k, y_t=o)]     flattened (k, o)
"""
import numpy as np

D = 10


def A_of(p):
    return np.array([[1 - p["learn"], p["learn"]], [p["forget"], 1 - p["forget"]]])  # A[i, j] = P(j | i)


def B_of(p):
    return np.array([[1 - p["guess"], p["guess"]], [p["slip"], 1 - p["slip"]]])  # B[k, o] = P(o | k)


def pad(seqs):
    """List of 1-D int arrays -> (S, L) int8 matrix padded with -2, and lengths."""
    lengths = np.array([len(s) for s in seqs])
    Y = np.full((len(seqs), lengths.max()), -2, dtype=np.int8)
    for i, s in enumerate(seqs):
        Y[i, : len(s)] = s
    return Y, lengths


def lik(Y, B):
    """(S, L, 2) emission likelihood; 1 for missing (-1) and padding (-2)."""
    out = np.ones(Y.shape + (2,))
    for o in (0, 1):
        out[Y == o] = B[:, o]
    return out


def m_step(S, fix_forget=True, p_old=None, pseudo=0.0):
    """M-step. A ratio with a zero denominator (no evidence yet) keeps p_old's value, or 0.5."""
    init, trans, emit = S[0:2] + pseudo, S[2:6].reshape(2, 2) + pseudo, S[6:10].reshape(2, 2) + pseudo
    old = p_old or {}

    def ratio(num, den, key):
        return num / den if den > 0 else old.get(key, 0.5)

    return {
        "prior": ratio(init[1], init.sum(), "prior"),
        "learn": ratio(trans[0, 1], trans[0].sum(), "learn"),
        "forget": 0.0 if fix_forget else ratio(trans[1, 0], trans[1].sum(), "forget"),
        "guess": ratio(emit[0, 1], emit[0].sum(), "guess"),
        "slip": ratio(emit[1, 0], emit[1].sum(), "slip"),
    }


def fb_counts(Y, lengths, p, per_student=False):
    """Batch E-step by scaled forward-backward, vectorized across students.
    Returns (S, loglik) or (S_per_student (n, D), loglik_per_student)."""
    A, B = A_of(p), B_of(p)
    n, L = Y.shape
    E = lik(Y, B)
    valid = np.arange(L)[None, :] < lengths[:, None]
    alpha = np.empty((n, L, 2))
    c = np.ones((n, L))
    a = np.array([1 - p["prior"], p["prior"]])[None, :] * E[:, 0]
    c[:, 0] = a.sum(1)
    alpha[:, 0] = a / c[:, 0:1]
    for t in range(1, L):
        a = (alpha[:, t - 1] @ A) * E[:, t]
        ct = a.sum(1)
        v = valid[:, t]
        c[:, t] = np.where(v, ct, 1.0)
        alpha[:, t] = np.where(v[:, None], a / np.where(v, ct, 1.0)[:, None], alpha[:, t - 1])
    # Backward pass in posterior form, as pyBKT's C++ E-step does: work with gamma (<= 1), not scaled beta.
    # The textbook scaled beta overflows on long sequences when "known" is absorbing (forget = 0):
    # beta(unknown) grew past 1e300 on a 3,585-answer ASSISTments student, giving inf * 0 = NaN.
    #   xi_t(i, j) = alpha_t(i) A(i, j) gamma_{t+1}(j) / (alpha_t A)(j),   gamma_t(i) = sum_j xi_t(i, j)
    Sx = np.zeros((n, D))
    last = lengths - 1
    gamma = np.empty((n, L, 2))
    gamma[np.arange(n), last] = alpha[np.arange(n), last]
    for t in range(L - 2, -1, -1):
        v = t < last                                  # students whose sequence continues past t
        pred = alpha[:, t] @ A                        # (alpha_t A)(j)
        with np.errstate(invalid="ignore", divide="ignore"):
            xi = alpha[:, t][:, :, None] * A[None] * (gamma[:, t + 1] / pred)[:, None, :]
        xi = np.nan_to_num(xi, nan=0.0)                # pyBKT: pair != pair -> 0
        Sx[:, 2:6] += np.where(v[:, None], xi.reshape(n, 4), 0)
        gamma[:, t] = np.where(v[:, None], xi.sum(2), gamma[:, t])
    valid = np.arange(L)[None, :] <= last[:, None]
    Sx[:, 0:2] = gamma[:, 0]
    for o in (0, 1):
        m = (Y == o) & valid
        Sx[:, 6 + o] = (gamma[:, :, 0] * m).sum(1)
        Sx[:, 8 + o] = (gamma[:, :, 1] * m).sum(1)
    ll = np.log(c).sum(1)
    if per_student:
        return Sx, ll
    return Sx.sum(0), ll.sum()


def batch_em(Y, lengths, p, iters=200, tol=1e-6, fix_forget=True):
    hist = []
    for _ in range(iters):
        S, ll = fb_counts(Y, lengths, p)
        hist.append(ll)
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol:
            break
        p = m_step(S, fix_forget)
    return p, hist


def predict(Y, lengths, p):
    """One-step-ahead P(correct) for each valid attempt (filtering), vectorized. Returns flat preds, flat labels."""
    A, B = A_of(p), B_of(p)
    n, L = Y.shape
    E = lik(Y, B)
    f = np.tile([1 - p["prior"], p["prior"]], (n, 1))
    preds = np.empty((n, L))
    for t in range(L):
        preds[:, t] = f @ B[:, 1]
        a = f * E[:, t]
        a /= a.sum(1, keepdims=True)
        f = a @ A
    valid = (np.arange(L)[None, :] < lengths[:, None]) & (Y >= 0)
    return preds[valid], Y[valid].astype(float)


class StudentStream:
    """Exact forward-only smoothing of BKT sufficient statistics for one student (Cappé 2011 style).

    Keeps only the filter phi (2) and rho (2 x D): rho[x] = E[S_1:t | x_t = x, y_1:t].
    With fixed parameters, stats() after the last attempt equals the forward-backward counts.
    """

    __slots__ = ("phi", "rho")

    def __init__(self):
        self.phi = None
        self.rho = None

    def update(self, y, A, B, prior):
        e = np.ones(2) if y < 0 else B[:, y]
        if self.phi is None:
            a = np.array([1 - prior, prior]) * e
            c = a.sum()
            self.phi = a / c
            rho = np.zeros((2, D))
            rho[0, 0] = rho[1, 1] = 1.0
        else:
            pred = self.phi @ A                       # P(x_t | y_1:t-1)
            a = pred * e
            c = a.sum()
            # r[x', x] = P(x_{t-1} = x' | x_t = x, y_1:t-1)
            r = self.phi[:, None] * A / pred[None, :]
            rho = np.zeros((2, D))
            for x in (0, 1):
                rho[x] = r[0, x] * self.rho[0] + r[1, x] * self.rho[1]
                rho[x, 2 + 0 * 2 + x] += r[0, x]
                rho[x, 2 + 1 * 2 + x] += r[1, x]
            self.phi = a / c
        if y >= 0:
            rho[0, 6 + y] += 1.0
            rho[1, 8 + y] += 1.0
        self.rho = rho
        return np.log(c)

    def stats(self):
        return self.phi @ self.rho
