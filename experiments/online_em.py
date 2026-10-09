"""Online EM for BKT, written against the papers it implements.

BKT is a two-state HMM. Its complete-data likelihood is an exponential family whose sufficient
statistics S are the 10 expected counts in bkt_np (init[2], trans[2x2], emit[2x2]), and whose
M-step theta_bar(s) is the closed-form ratio in bkt_np.m_step. That puts BKT inside
Assumption 1 of Cappé & Moulines (2009), with:

  s_bar(Y; theta) = E_theta[S(X) | Y]     the E-step for one observation Y
  theta_bar(s)    = argmax_theta l(s; theta)   the M-step (eq. 13), bkt_np.m_step

Two notions of "observation" give two algorithms:

1. StudentLevelOnlineEM: one observation Y is one student's complete answer sequence for a skill.
   Students are independent, so this is exactly the setting of Cappé & Moulines (2009),
   "Online EM Algorithm for Latent Data Models", JRSS-B 71(3), arXiv:0712.4273:
     eq. (15)  s_{n+1} = s_n + gamma_{n+1} (s_bar(Y_{n+1}; theta_n) - s_n),  theta_{n+1} = theta_bar(s_{n+1})
     Thm 5     gamma_n = gamma0 * n^-alpha, alpha in (1/2, 1], so that sum gamma = inf, sum gamma^2 < inf
     Sec. 2.3  alpha = 1, gamma0 = 1 (gamma_n = 1/n) is Neal & Hinton's incremental EM in online mode
     Asm 1(c), Sec. 4   no M-step until s_n is inside S (they skip the first 20 observations)
     eq. (29)  Polyak-Ruppert averaging of theta_n for alpha in (1/2, 1)
     Sec. 3.1  stability by truncating/projecting s_n to a compact subset of S
               (here: optional pseudo-counts, which keep every ratio inside (0, 1))
   s_bar is computed exactly by forward-backward on the one sequence (bkt_np.fb_counts).
   A sequence must be complete before it is processed.

2. EventLevelOnlineEM: an observation is one answer, and a student's answers arrive interleaved
   with other students'. Within one student the observations are not independent (HMM), so
   eq. (15) does not apply directly. This uses:
     - the forward smoothing of additive functionals (bkt_np.StudentStream; the recursion used by
       Cappé (2011), "Online EM algorithm for hidden Markov models", JCGS 20(3), arXiv:0908.2359).
       It gives E[S_1:t | y_1:t] exactly from 22 numbers per student. Checked numerically
       against forward-backward (check_smoothing.py: max rel diff 2.5e-16). The recursion has NOT
       been checked line by line against the 2011 paper's text (arXiv was unreachable from here).
     - global statistics = sum over students of their current smoothed statistics, with each
       student's old contribution swapped for its new one on every event (incremental EM in the
       sense of Neal & Hinton), and an optional exponential discount lambda as in
       Mongillo & Denève (2008), Neural Computation 20(7), for drifting data.
   Cappé (2011) itself runs one long sequence with a decreasing step size; the multi-student
   bookkeeping here is our adaptation, not taken from a paper.

Both classes take and return bkt_np-style parameter dicts (prior, learn, forget, guess, slip).
"""
import numpy as np
from bkt_np import D, A_of, B_of, m_step, fb_counts, pad, StudentStream

KEYS = ("prior", "learn", "guess", "slip")


class StudentLevelOnlineEM:
    """Cappé & Moulines (2009) eq. (15) with one student sequence per observation."""

    def __init__(self, p0, alpha=0.75, gamma0=1.0, burn_in=20, pseudo=0.0, average_from=None):
        self.p = dict(p0)
        self.alpha, self.gamma0, self.burn_in, self.pseudo = alpha, gamma0, burn_in, pseudo
        self.s = None          # running statistics, per-student scale
        self.n = 0             # observations (students) seen
        self.average_from = average_from   # start Polyak-Ruppert averaging after this many observations
        self._avg, self._avg_n = None, 0

    def gamma(self, n):
        return min(1.0, self.gamma0 * n ** -self.alpha)      # Thm 5 step-size schedule

    def update(self, sequences):
        """Process a mini-batch of complete sequences (1-D arrays of 0/1, -1 missing).
        A batch of m sequences counts as m observations with one combined step."""
        Y, L = pad(sequences)
        s_bar = fb_counts(Y, L, self.p)[0] / len(sequences)   # mean of s_bar(Y_i; theta_n) over the batch
        self.n += len(sequences)
        g = self.gamma(self.n // len(sequences) if len(sequences) > 1 else self.n)
        self.s = s_bar if self.s is None else self.s + g * (s_bar - self.s)   # eq. (15), first line
        if self.n >= self.burn_in:                                             # Asm 1(c): wait for s in S
            self.p = m_step(self.s, p_old=self.p, pseudo=self.pseudo)          # eq. (15), second line
            if self.average_from is not None and self.n >= self.average_from:  # eq. (29)
                v = np.array([self.p[k] for k in KEYS])
                self._avg_n += 1
                self._avg = v if self._avg is None else self._avg + (v - self._avg) / self._avg_n
        return self.p

    def averaged(self):
        if self._avg is None:
            return dict(self.p)
        return dict(self.p, **dict(zip(KEYS, self._avg)))


class EventLevelOnlineEM:
    """Per-answer updates with exact forward smoothing per student (see module docstring, part 2)."""

    def __init__(self, p0, m_step_every=100, discount=1.0, pseudo=0.0):
        self.p = dict(p0)
        self.A, self.B = A_of(p0), B_of(p0)
        self.every, self.discount, self.pseudo = m_step_every, discount, pseudo
        self.students, self.contrib, self.when = {}, {}, {}
        self.S = np.zeros(D)
        self.n = 0

    def predict(self, student):
        """P(correct) for the student's next answer (filtering; this is pyBKT's one-step prediction)."""
        st = self.students.get(student)
        known = self.p["prior"] if st is None else (st.phi @ self.A)[1]
        return known * (1 - self.p["slip"]) + (1 - known) * self.p["guess"]

    def update(self, student, y):
        st = self.students.get(student)
        old = 0.0
        if st is None:
            st = self.students[student] = StudentStream()
        else:
            old = self.contrib[student]
        st.update(int(y), self.A, self.B, self.p["prior"])
        new = st.stats()
        self.n += 1
        if self.discount < 1:
            self.S *= self.discount
            if student in self.when:   # the stored contribution decayed along with S
                old = old * self.discount ** (self.n - self.when[student])
            self.when[student] = self.n
        self.S += new - old
        self.contrib[student] = new
        if self.n % self.every == 0:
            self.p = m_step(self.S, p_old=self.p, pseudo=self.pseudo)
            self.A, self.B = A_of(self.p), B_of(self.p)
        return self.p


if __name__ == "__main__":
    # Self-checks tying the code to the equations.
    rng = np.random.default_rng(0)
    seqs = [rng.choice([0, 1], size=rng.integers(2, 30)) for _ in range(400)]
    p0 = dict(prior=0.3, learn=0.2, forget=0.0, guess=0.25, slip=0.1)

    # (a) Sec. 2.3: alpha=1, gamma0=1 is incremental EM in online mode: s_n is the plain average
    #     of s_bar(Y_i; theta_{i-1}). Check by recomputing that average directly.
    em = StudentLevelOnlineEM(p0, alpha=1.0, gamma0=1.0, burn_in=1)
    p, acc = dict(p0), np.zeros(D)
    for i, s in enumerate(seqs, 1):
        Y, L = pad([s]); acc += fb_counts(Y, L, p)[0]
        p = m_step(acc / i, p_old=p)
        em.update([s])
    print("eq.15 with gamma_n=1/n equals running-average incremental EM:",
          max(abs(em.p[k] - p[k]) for k in KEYS) < 1e-12)

    # (b) With M-steps disabled, EventLevelOnlineEM's statistics equal the batch E-step exactly.
    ev = EventLevelOnlineEM(p0, m_step_every=10**12)
    order = [(i, int(y)) for i, s in enumerate(seqs) for y in s]
    rng.shuffle(order)
    # keep each student's answers in order while interleaving students
    pos = {i: 0 for i in range(len(seqs))}
    for i, _ in order:
        ev.update(i, seqs[i][pos[i]]); pos[i] += 1
    Y, L = pad(seqs)
    S_batch = fb_counts(Y, L, p0)[0]
    print("event-level smoothed statistics equal forward-backward (interleaved students):",
          np.max(np.abs(ev.S - S_batch) / np.maximum(1.0, np.abs(S_batch))) < 1e-12)
