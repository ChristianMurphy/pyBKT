//! BKT E-step kernels with no dependencies beyond std (no `unsafe`).
//! The exact path mirrors pyBKT/source-cpp/pyBKT/fit/E_step.cpp operation for
//! operation, so the serial path is bit-identical to the C++ serial path.
//!
//! A(i,j) = P(state i at t+1 | state j at t); A = [[1-learn, forget], [learn, 1-forget]].
//! Count layouts are the raw C++ buffer layouts as Python sees them:
//!   trans[r][j][i] (from j, to i), emission[n][o][i] (obs o, state i), init[i].

use std::cell::Cell;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex};

// ------------------------------------------------------------------ inputs

pub trait Obs {
    /// 0 = no response, 1 = incorrect (any nonzero value other than 2), 2 = correct
    fn code(&self) -> usize;
}
macro_rules! impl_obs {
    ($($t:ty),*) => {$(
        impl Obs for $t {
            #[inline(always)]
            fn code(&self) -> usize { (*self != 0) as usize + (*self == 2) as usize }
        }
    )*};
}
impl_obs!(i8, u8, i16, i32, i64);

pub trait Res {
    fn val(&self) -> i64;
}
macro_rules! impl_res {
    ($($t:ty),*) => {$(
        impl Res for $t {
            #[inline(always)]
            fn val(&self) -> i64 { *self as i64 }
        }
    )*};
}
impl_res!(u8, u16, i16, i32, u32, i64);

#[derive(Clone, Debug)]
pub struct Params {
    pub prior: f64,
    pub learns: Vec<f64>,
    pub forgets: Vec<f64>,
    pub guesses: Vec<f64>,
    pub slips: Vec<f64>,
}

pub struct Model {
    /// per resource: [a00, a01, a10, a11]
    pub a: Vec<[f64; 4]>,
    /// per resource, used only for predictions (C++ As_model)
    pub a_pred: Vec<[f64; 4]>,
    /// per subpart, per obs code: factor for state 0/1 (B entry 0 stored as 1.0, as C++ skips it)
    pub lik: Vec<[[f64; 2]; 3]>,
    pub init: [f64; 2],
}

impl Model {
    pub fn new(p: &Params, pred_learns: &[f64], pred_forgets: &[f64]) -> Model {
        let mk = |l: f64, f: f64| [1.0 - l, f, l, 1.0 - f];
        let a = p.learns.iter().zip(&p.forgets).map(|(&l, &f)| mk(l, f)).collect();
        let a_pred = pred_learns.iter().zip(pred_forgets).map(|(&l, &f)| mk(l, f)).collect();
        let nz = |x: f64| if x != 0.0 { x } else { 1.0 };
        let lik = p
            .guesses
            .iter()
            .zip(&p.slips)
            .map(|(&g, &s)| [[1.0, 1.0], [nz(1.0 - g), nz(s)], [nz(g), nz(1.0 - s)]])
            .collect();
        Model { a, a_pred, lik, init: [1.0 - p.prior, p.prior] }
    }
    pub fn from_params(p: &Params) -> Model {
        Model::new(p, &p.learns, &p.forgets)
    }
}

pub struct Input<'a, D, R> {
    pub data: &'a [D], // (K, N) row-major
    pub k: usize,
    pub n: usize,
    pub res: &'a [R],
    pub starts: &'a [i64],
    pub lengths: &'a [i64],
}

impl<D: Obs, R: Res> Input<'_, D, R> {
    pub fn validate(&self, num_res: usize) -> Result<(), String> {
        if self.data.len() != self.k * self.n {
            return Err("data size mismatch".into());
        }
        if self.starts.len() != self.lengths.len() {
            return Err("starts and lengths differ in length".into());
        }
        for (&s, &l) in self.starts.iter().zip(self.lengths) {
            if s < 1 || l < 1 || (s - 1 + l) as u64 > self.n as u64 {
                return Err(format!("sequence start={s} length={l} out of range"));
            }
        }
        if num_res > 1 {
            if self.res.len() < self.n {
                return Err("resources shorter than data".into());
            }
            for r in self.res {
                let v = r.val();
                if v < 1 || v > num_res as i64 {
                    return Err(format!("resource {v} out of range 1..={num_res}"));
                }
            }
        }
        Ok(())
    }
}

impl<D, R> Input<'_, D, R> {
    #[inline(always)]
    pub fn span(&self, si: usize) -> (usize, usize) {
        ((self.starts[si] - 1) as usize, self.lengths[si] as usize)
    }
    /// sequences laid out in index order without overlap (chunks can own disjoint output spans)
    pub fn ordered_disjoint(&self) -> bool {
        self.starts.windows(2).zip(self.lengths).all(|(w, &l)| w[0] + l <= w[1])
    }
    /// [lo, hi) output span owned by each chunk (valid when ordered_disjoint or one chunk)
    pub fn chunk_spans(&self, chunks: &[Range<usize>]) -> Vec<(usize, usize)> {
        if chunks.len() <= 1 {
            return vec![(0, self.n)];
        }
        (0..chunks.len())
            .map(|ci| {
                let lo = if ci == 0 { 0 } else { self.span(chunks[ci].start).0 };
                let hi = if ci + 1 < chunks.len() { self.span(chunks[ci + 1].start).0 } else { self.n };
                (lo, hi)
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Counts {
    pub trans: Vec<f64>, // R*4, [r][j][i]
    pub em: Vec<f64>,    // K*4, [n][o][i]
    pub init: [f64; 2],
    pub ll: f64,
}

impl Counts {
    pub fn zero(r: usize, k: usize) -> Counts {
        Counts { trans: vec![0.0; 4 * r], em: vec![0.0; 4 * k], init: [0.0; 2], ll: 0.0 }
    }
    pub fn add(&mut self, o: &Counts) {
        for (a, b) in self.trans.iter_mut().zip(&o.trans) {
            *a += b;
        }
        for (a, b) in self.em.iter_mut().zip(&o.em) {
            *a += b;
        }
        self.init[0] += o.init[0];
        self.init[1] += o.init[1];
        self.ll += o.ll;
    }
}

pub fn reduce(parts: &[Counts], nr: usize, k: usize) -> Counts {
    let mut total = Counts::zero(nr, k);
    for p in parts {
        total.add(p);
    }
    total
}

// ------------------------------------------------------------- output sinks

/// callback receiving (start, row0, row1) of one sequence
pub type SeqFn<'a> = dyn FnMut(usize, &[f64], &[f64]) + 'a;

/// Receives the finished alpha / prediction rows of one sequence.
/// One virtual call per *sequence*, not per element.
pub trait Emit {
    fn active(&self) -> bool {
        true
    }
    fn emit(&mut self, st: usize, a0: &[f64], a1: &[f64]);
    /// for buffering emitters: hand the buffered sequences to `f` (on the main thread)
    fn drain(&mut self, _f: &mut SeqFn<'_>) {}
}

pub struct NoEmit;
impl Emit for NoEmit {
    fn active(&self) -> bool {
        false
    }
    fn emit(&mut self, _: usize, _: &[f64], _: &[f64]) {}
}

/// rows of f64 starting at output position `lo`
pub struct F64Emit<'a> {
    pub r0: &'a mut [f64],
    pub r1: &'a mut [f64],
    pub lo: usize,
}
impl Emit for F64Emit<'_> {
    #[inline]
    fn emit(&mut self, st: usize, a0: &[f64], a1: &[f64]) {
        let o = st - self.lo;
        self.r0[o..o + a0.len()].copy_from_slice(a0);
        self.r1[o..o + a1.len()].copy_from_slice(a1);
    }
}

/// rows of native-endian f64 bytes (PyByteArray / Vec<u8>) starting at `lo`
pub struct ByteEmit<'a> {
    pub r0: &'a mut [u8],
    pub r1: &'a mut [u8],
    pub lo: usize,
}
impl Emit for ByteEmit<'_> {
    #[inline]
    fn emit(&mut self, st: usize, a0: &[f64], a1: &[f64]) {
        let o = (st - self.lo) * 8;
        for (d, v) in self.r0[o..o + 8 * a0.len()].chunks_exact_mut(8).zip(a0) {
            d.copy_from_slice(&v.to_ne_bytes());
        }
        for (d, v) in self.r1[o..o + 8 * a1.len()].chunks_exact_mut(8).zip(a1) {
            d.copy_from_slice(&v.to_ne_bytes());
        }
    }
}

/// a writable Python buffer seen as &[Cell<f64>] (GIL-bound, main thread only)
pub struct CellEmit<'a> {
    pub r0: &'a [Cell<f64>],
    pub r1: &'a [Cell<f64>],
}
impl Emit for CellEmit<'_> {
    #[inline]
    fn emit(&mut self, st: usize, a0: &[f64], a1: &[f64]) {
        for (c, &v) in self.r0[st..st + a0.len()].iter().zip(a0) {
            c.set(v);
        }
        for (c, &v) in self.r1[st..st + a1.len()].iter().zip(a1) {
            c.set(v);
        }
    }
}

/// chunk-local buffer, drained on the main thread
#[derive(Default)]
pub struct PackEmit {
    b0: Vec<f64>,
    b1: Vec<f64>,
    seqs: Vec<(usize, usize)>,
}
impl Emit for PackEmit {
    fn emit(&mut self, st: usize, a0: &[f64], a1: &[f64]) {
        self.seqs.push((st, a0.len()));
        self.b0.extend_from_slice(a0);
        self.b1.extend_from_slice(a1);
    }
    fn drain(&mut self, f: &mut SeqFn<'_>) {
        let mut p = 0;
        for &(st, l) in &self.seqs {
            f(st, &self.b0[p..p + l], &self.b1[p..p + l]);
            p += l;
        }
        *self = PackEmit::default();
    }
}

/// Split full output rows into per-chunk emitters for the given spans.
pub fn split_f64<'a>(mut r0: &'a mut [f64], mut r1: &'a mut [f64], spans: &[(usize, usize)]) -> Vec<F64Emit<'a>> {
    let mut v = Vec::with_capacity(spans.len());
    let mut pos = 0;
    for &(lo, hi) in spans {
        let (_, a) = std::mem::take(&mut r0).split_at_mut(lo - pos);
        let (m0, rest0) = a.split_at_mut(hi - lo);
        let (_, b) = std::mem::take(&mut r1).split_at_mut(lo - pos);
        let (m1, rest1) = b.split_at_mut(hi - lo);
        r0 = rest0;
        r1 = rest1;
        pos = hi;
        v.push(F64Emit { r0: m0, r1: m1, lo });
    }
    v
}

pub fn split_bytes<'a>(mut r0: &'a mut [u8], mut r1: &'a mut [u8], spans: &[(usize, usize)]) -> Vec<ByteEmit<'a>> {
    let mut v = Vec::with_capacity(spans.len());
    let mut pos = 0;
    for &(lo, hi) in spans {
        let (_, a) = std::mem::take(&mut r0).split_at_mut(8 * (lo - pos));
        let (m0, rest0) = a.split_at_mut(8 * (hi - lo));
        let (_, b) = std::mem::take(&mut r1).split_at_mut(8 * (lo - pos));
        let (m1, rest1) = b.split_at_mut(8 * (hi - lo));
        r0 = rest0;
        r1 = rest1;
        pos = hi;
        v.push(ByteEmit { r0: m0, r1: m1, lo });
    }
    v
}

// ------------------------------------------------------------- chunk driver

/// Split sequences (in index order) into chunks of >= chunk_attempts attempts.
pub fn plan_chunks(lengths: &[i64], chunk_attempts: usize) -> Vec<Range<usize>> {
    let mut v = Vec::new();
    let mut begin = 0;
    let mut acc = 0usize;
    for (i, &l) in lengths.iter().enumerate() {
        acc += l as usize;
        if acc >= chunk_attempts {
            v.push(begin..i + 1);
            begin = i + 1;
            acc = 0;
        }
    }
    if begin < lengths.len() {
        v.push(begin..lengths.len());
    }
    v
}

pub type BoxEmit<'a> = Box<dyn Emit + Send + 'a>;

/// Run `work(i, emitter_i)` for every chunk i on `threads` scoped std threads
/// (chunks claimed dynamically through an AtomicUsize). Finished chunks are sent
/// to the calling thread, which calls `on_done` on their emitter as they arrive
/// (pipelined) and stores the result at index i: results come back in chunk
/// order regardless of which thread ran what.
pub fn run_threads<'a, T, F>(
    threads: usize,
    emits: Vec<BoxEmit<'a>>,
    work: F,
    on_done: &mut dyn FnMut(&mut dyn Emit),
) -> Vec<T>
where
    T: Send,
    F: Fn(usize, &mut dyn Emit) -> T + Sync,
{
    let n = emits.len();
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    let slots: Vec<Mutex<Option<BoxEmit<'a>>>> = emits.into_iter().map(|e| Mutex::new(Some(e))).collect();
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel::<(usize, T, BoxEmit<'a>)>();
    std::thread::scope(|s| {
        for _ in 0..threads.min(n) {
            let tx = tx.clone();
            let (work, next, slots) = (&work, &next, &slots);
            s.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let mut e = slots[i].lock().expect("poisoned").take().expect("claimed twice");
                let t = work(i, &mut *e);
                if tx.send((i, t, e)).is_err() {
                    break;
                }
            });
        }
        drop(tx);
        for (i, t, mut e) in rx {
            on_done(&mut *e);
            out[i] = Some(t);
        }
    });
    out.into_iter().map(|t| t.expect("chunk result missing")).collect()
}

/// Same contract as run_threads, on the calling thread (no Sync/Send needed).
pub fn run_inline<'a, T>(
    mut emits: Vec<Box<dyn Emit + 'a>>,
    mut work: impl FnMut(usize, &mut dyn Emit) -> T,
    on_done: &mut dyn FnMut(&mut dyn Emit),
) -> Vec<T> {
    let mut out = Vec::with_capacity(emits.len());
    for (i, e) in emits.iter_mut().enumerate() {
        out.push(work(i, &mut **e));
        on_done(&mut **e);
    }
    out
}

// ------------------------------------------------------- exact scalar kernel

#[inline(always)]
pub fn lik_k<D: Obs, R>(inp: &Input<D, R>, m: &Model, pos: usize) -> (f64, f64) {
    let (mut l0, mut l1) = (1.0f64, 1.0f64);
    for n in 0..inp.k {
        let l = &m.lik[n][inp.data[n * inp.n + pos].code()];
        l0 *= l[0];
        l1 *= l[1];
    }
    (l0, l1)
}

pub struct Scratch {
    pub s0: Vec<f64>,
    pub s1: Vec<f64>,
}
impl Scratch {
    pub fn new() -> Self {
        Scratch { s0: Vec::new(), s1: Vec::new() }
    }
    fn get(&mut self, tlen: usize) -> (&mut [f64], &mut [f64]) {
        if self.s0.len() < tlen {
            self.s0.resize(tlen, 0.0);
            self.s1.resize(tlen, 0.0);
        }
        (&mut self.s0[..tlen], &mut self.s1[..tlen])
    }
}
impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Process sequences `seqs`, accumulating into `c` in exactly the C++ order.
#[inline(always)]
fn process<D: Obs, R: Res, const K1: bool, const R1: bool, const EMIT: bool>(
    inp: &Input<D, R>,
    m: &Model,
    seqs: Range<usize>,
    c: &mut Counts,
    emit: &mut dyn Emit,
    sc: &mut Scratch,
) {
    let [i0, i1] = m.init;
    let lk = m.lik[0];
    let a_r1 = m.a[0];
    let mut tr = [0.0f64; 4];
    let mut em = [[0.0f64; 2]; 3];
    if R1 {
        tr.copy_from_slice(&c.trans[0..4]);
    }
    if K1 {
        em[1] = [c.em[0], c.em[1]];
        em[2] = [c.em[2], c.em[3]];
    }
    let mut init = c.init;
    let mut ll = c.ll;

    for si in seqs {
        let (st, tlen) = inp.span(si);
        let (a0, a1) = sc.get(tlen);
        let a1 = &mut a1[..a0.len()];
        let d = &inp.data[st..st + tlen]; // row 0 (the only row when K1)
        let rs = if R1 { &inp.res[..0] } else { &inp.res[st..st + tlen] };
        let lik = |t: usize| -> (f64, f64) {
            if K1 {
                let l = lk[d[t].code()];
                (l[0], l[1])
            } else {
                lik_k(inp, m, st + t)
            }
        };
        let amat = |t: usize| -> [f64; 4] { if R1 { a_r1 } else { m.a[(rs[t].val() - 1) as usize] } };
        // ---- forward
        let (l0, l1) = lik(0);
        let mut x0 = i0 * l0;
        let mut x1 = i1 * l1;
        let mut norm = x0 + x1;
        x0 /= norm;
        x1 /= norm;
        ll += norm.ln();
        a0[0] = x0;
        a1[0] = x1;
        for t in 0..tlen - 1 {
            let a = amat(t);
            let p0 = a[0] * x0 + a[1] * x1;
            let p1 = a[2] * x0 + a[3] * x1;
            let (l0, l1) = lik(t + 1);
            x0 = p0 * l0;
            x1 = p1 * l1;
            norm = x0 + x1;
            x0 /= norm;
            x1 /= norm;
            ll += norm.ln();
            a0[t + 1] = x0;
            a1[t + 1] = x1;
        }
        // ---- backward
        let mut g0 = x0;
        let mut g1 = x1;
        let emit_em = |em: &mut [[f64; 2]; 3], c: &mut Counts, t: usize, g0: f64, g1: f64| {
            if K1 {
                let code = d[t].code();
                if code != 0 {
                    em[code][0] += g0;
                    em[code][1] += g1;
                }
            } else {
                for n in 0..inp.k {
                    let code = inp.data[n * inp.n + st + t].code();
                    if code != 0 {
                        let b = n * 4 + (code - 1) * 2;
                        c.em[b] += g0;
                        c.em[b + 1] += g1;
                    }
                }
            }
        };
        emit_em(&mut em, c, tlen - 1, g0, g1);
        for t in (0..tlen - 1).rev() {
            let a = amat(t);
            let (y0, y1) = (a0[t], a1[t]);
            let p0 = a[0] * y0 + a[1] * y1;
            let p1 = a[2] * y0 + a[3] * y1;
            // pair(i,j) = ((A(i,j) * alpha_j) * gamma_i) / p_i ; NaN -> 0
            let nz = |v: f64| if v.is_nan() { 0.0 } else { v };
            let q00 = nz(a[0] * y0 * g0 / p0);
            let q01 = nz(a[1] * y1 * g0 / p0);
            let q10 = nz(a[2] * y0 * g1 / p1);
            let q11 = nz(a[3] * y1 * g1 / p1);
            let b: &mut [f64] = if R1 {
                &mut tr
            } else {
                let r = (rs[t].val() - 1) as usize;
                &mut c.trans[r * 4..r * 4 + 4]
            };
            b[0] += q00;
            b[1] += q10;
            b[2] += q01;
            b[3] += q11;
            g0 = q00 + q10;
            g1 = q01 + q11;
            emit_em(&mut em, c, t, g0, g1);
        }
        init[0] += g0;
        init[1] += g1;
        if EMIT {
            // a call per sequence; kept out of the no-output instantiation so the
            // accumulators can stay in registers across sequences
            emit.emit(st, a0, a1);
        }
    }
    if R1 {
        c.trans[0..4].copy_from_slice(&tr);
    }
    if K1 {
        c.em[0..4].copy_from_slice(&[em[1][0], em[1][1], em[2][0], em[2][1]]);
    }
    c.init = init;
    c.ll = ll;
}

pub fn process_any<D: Obs, R: Res>(
    inp: &Input<D, R>,
    m: &Model,
    seqs: Range<usize>,
    c: &mut Counts,
    emit: &mut dyn Emit,
    sc: &mut Scratch,
) {
    macro_rules! go {
        ($e:expr) => {
            match (inp.k == 1, m.a.len() == 1) {
                (true, true) => process::<D, R, true, true, $e>(inp, m, seqs, c, emit, sc),
                (true, false) => process::<D, R, true, false, $e>(inp, m, seqs, c, emit, sc),
                (false, true) => process::<D, R, false, true, $e>(inp, m, seqs, c, emit, sc),
                (false, false) => process::<D, R, false, false, $e>(inp, m, seqs, c, emit, sc),
            }
        };
    }
    if emit.active() {
        go!(true)
    } else {
        go!(false)
    }
}

// ------------------------------------------------------------------ predict

#[inline(always)]
fn predict_seqs<D: Obs, R: Res, const K1: bool, const R1: bool>(
    inp: &Input<D, R>,
    m: &Model,
    seqs: Range<usize>,
    emit: &mut dyn Emit,
    sc: &mut Scratch,
) {
    let [i0, i1] = m.init;
    let lk = m.lik[0];
    for si in seqs {
        let (st, tlen) = inp.span(si);
        let (o0, o1) = sc.get(tlen);
        let o1 = &mut o1[..o0.len()];
        let d = &inp.data[st..st + tlen];
        let rs = if R1 { &inp.res[..0] } else { &inp.res[st..st + tlen] };
        let lik = |t: usize| -> (f64, f64) {
            if K1 {
                let l = lk[d[t].code()];
                (l[0], l[1])
            } else {
                lik_k(inp, m, st + t)
            }
        };
        o0[0] = i0;
        o1[0] = i1;
        let (l0, l1) = lik(0);
        let mut x0 = i0 * l0;
        let mut x1 = i1 * l1;
        let mut norm = x0 + x1;
        x0 /= norm;
        x1 /= norm;
        for t in 0..tlen - 1 {
            let r = if R1 { 0 } else { (rs[t].val() - 1) as usize };
            let a = m.a[r];
            let ap = m.a_pred[r];
            o0[t + 1] = ap[0] * x0 + ap[1] * x1;
            o1[t + 1] = ap[2] * x0 + ap[3] * x1;
            let p0 = a[0] * x0 + a[1] * x1;
            let p1 = a[2] * x0 + a[3] * x1;
            let (l0, l1) = lik(t + 1);
            x0 = p0 * l0;
            x1 = p1 * l1;
            norm = x0 + x1;
            x0 /= norm;
            x1 /= norm;
        }
        emit.emit(st, o0, o1);
    }
}

pub fn predict_any<D: Obs, R: Res>(inp: &Input<D, R>, m: &Model, seqs: Range<usize>, emit: &mut dyn Emit, sc: &mut Scratch) {
    match (inp.k == 1, m.a.len() == 1) {
        (true, true) => predict_seqs::<D, R, true, true>(inp, m, seqs, emit, sc),
        (true, false) => predict_seqs::<D, R, true, false>(inp, m, seqs, emit, sc),
        (false, true) => predict_seqs::<D, R, false, true>(inp, m, seqs, emit, sc),
        (false, false) => predict_seqs::<D, R, false, false>(inp, m, seqs, emit, sc),
    }
}
