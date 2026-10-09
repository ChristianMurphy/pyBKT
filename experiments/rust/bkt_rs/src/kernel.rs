//! Pure-Rust BKT E-step kernels (no Python types, no `unsafe`). The exact path
//! mirrors pyBKT/source-cpp/pyBKT/fit/E_step.cpp operation-for-operation so that
//! the serial path is bit-identical to the C++ serial path.
//!
//! Conventions (same as the C++):
//!   A(i,j) = P(state i at t+1 | state j at t); state 0 = unknown, 1 = known.
//!   A = [[1-learn, forget], [learn, 1-forget]]
//!   B(i, o): o = 0 incorrect -> (1-guess, slip); o = 1 correct -> (guess, 1-slip)
//! Output count layouts are the *raw* C++ buffer layouts as Python sees them:
//!   trans[r][j][i]    (from j, to i)     shape (R,2,2)
//!   emission[n][o][i] (obs o, state i)   shape (K,2,2)
//!   init[i]                              shape (2,1)

use rayon::prelude::*;
use std::ops::Range;

// ------------------------------------------------------------------ inputs

pub trait Obs: Copy + Send + Sync {
    /// 0 = no response, 1 = incorrect (any nonzero value other than 2), 2 = correct
    fn code(self) -> usize;
}
macro_rules! impl_obs {
    ($($t:ty),*) => {$(
        impl Obs for $t {
            #[inline(always)]
            fn code(self) -> usize { (self != 0) as usize + (self == 2) as usize }
        }
    )*};
}
impl_obs!(i8, u8, i16, i32, i64);

pub trait Res: Copy + Send + Sync {
    fn val(self) -> i64;
}
macro_rules! impl_res {
    ($($t:ty),*) => {$(
        impl Res for $t {
            #[inline(always)]
            fn val(self) -> i64 { self as i64 }
        }
    )*};
}
impl_res!(u8, u16, i16, i32, u32, i64);

pub struct Model {
    /// per resource: [a00, a01, a10, a11]
    pub a: Vec<[f64; 4]>,
    /// per resource, used only for predictions (C++ As_model)
    pub a_pred: Vec<[f64; 4]>,
    /// per subpart, per obs code (0,1,2): factor for state 0/1. A factor whose
    /// B entry is 0 is stored as 1.0 (C++ skips the multiply; x*1.0 == x exactly).
    pub lik: Vec<[[f64; 2]; 3]>,
    pub init: [f64; 2],
}

impl Model {
    pub fn new(
        prior: f64,
        learns: &[f64],
        forgets: &[f64],
        guesses: &[f64],
        slips: &[f64],
        pred_learns: &[f64],
        pred_forgets: &[f64],
    ) -> Model {
        let mk = |l: f64, f: f64| [1.0 - l, f, l, 1.0 - f];
        let a = learns.iter().zip(forgets).map(|(&l, &f)| mk(l, f)).collect();
        let a_pred = pred_learns.iter().zip(pred_forgets).map(|(&l, &f)| mk(l, f)).collect();
        let nz = |x: f64| if x != 0.0 { x } else { 1.0 };
        let lik = guesses
            .iter()
            .zip(slips)
            .map(|(&g, &s)| [[1.0, 1.0], [nz(1.0 - g), nz(s)], [nz(g), nz(1.0 - s)]])
            .collect();
        Model { a, a_pred, lik, init: [1.0 - prior, prior] }
    }
}

pub struct Input<'a, D: Obs, R: Res> {
    pub data: &'a [D], // (K, N) row-major
    pub k: usize,
    pub n: usize,
    pub res: &'a [R],
    pub starts: &'a [i64],
    pub lengths: &'a [i64],
}

impl<'a, D: Obs, R: Res> Input<'a, D, R> {
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
            for &r in self.res {
                let v = r.val();
                if v < 1 || v > num_res as i64 {
                    return Err(format!("resource {v} out of range 1..={num_res}"));
                }
            }
        }
        Ok(())
    }

    #[inline(always)]
    pub fn span(&self, si: usize) -> (usize, usize) {
        ((self.starts[si] - 1) as usize, self.lengths[si] as usize)
    }

    /// sequences laid out in index order without overlap (lets chunks own disjoint output spans)
    fn ordered_disjoint(&self) -> bool {
        self.starts.windows(2).zip(self.lengths).all(|(w, &l)| w[0] + l <= w[1])
    }
}

#[derive(Clone)]
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

/// Where one chunk writes its alpha / prediction rows: two disjoint mutable
/// slices plus the offset of each of the chunk's sequences inside them.
pub struct Sink<'a> {
    pub(crate) r0: &'a mut [f64],
    pub(crate) r1: &'a mut [f64],
    pub(crate) offs: Vec<usize>,
    pub(crate) first: usize,
}

impl<'a> Sink<'a> {
    #[inline(always)]
    pub fn rows(&mut self, si: usize, tlen: usize) -> (&mut [f64], &mut [f64]) {
        let o = self.offs[si - self.first];
        (&mut self.r0[o..o + tlen], &mut self.r1[o..o + tlen])
    }
}

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

pub fn pool(threads: usize) -> std::sync::Arc<rayon::ThreadPool> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut g = POOLS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    g.entry(threads)
        .or_insert_with(|| Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()))
        .clone()
}

/// Run `work` over `chunks` (on the calling thread if threads == 0, else on a
/// rayon pool) and return the per-chunk results in chunk order. If `out` is
/// given, each chunk gets a Sink: a disjoint sub-slice of the output when the
/// sequences are ordered and non-overlapping (or there is a single chunk),
/// otherwise a chunk-local buffer that is scattered into `out` afterwards.
pub fn drive<D, R, T, F>(
    inp: &Input<D, R>,
    chunks: &[Range<usize>],
    threads: usize,
    out: Option<(&mut [f64], &mut [f64])>,
    work: F,
) -> Vec<T>
where
    D: Obs,
    R: Res,
    T: Send,
    F: Fn(Range<usize>, Option<&mut Sink>) -> T + Sync + Send,
{
    let run_all = |jobs: Vec<(Range<usize>, Option<Sink>)>| -> Vec<T> {
        let f = |(rg, mut s): (Range<usize>, Option<Sink>)| work(rg, s.as_mut());
        if threads == 0 {
            jobs.into_iter().map(f).collect()
        } else {
            pool(threads).install(|| jobs.into_par_iter().map(f).collect())
        }
    };
    let Some((mut o0, mut o1)) = out else {
        return run_all(chunks.iter().map(|rg| (rg.clone(), None)).collect());
    };
    if chunks.len() <= 1 || inp.ordered_disjoint() {
        let mut jobs = Vec::with_capacity(chunks.len());
        let mut pos = 0usize;
        let single = chunks.len() <= 1;
        for (ci, rg) in chunks.iter().enumerate() {
            let (lo, hi) = if single {
                (0, inp.n)
            } else {
                let lo = inp.span(rg.start).0;
                let hi = if ci + 1 < chunks.len() { inp.span(chunks[ci + 1].start).0 } else { inp.n };
                (lo, hi)
            };
            let (_, r0) = std::mem::take(&mut o0).split_at_mut(lo - pos);
            let (mine0, rest0) = r0.split_at_mut(hi - lo);
            let (_, r1) = std::mem::take(&mut o1).split_at_mut(lo - pos);
            let (mine1, rest1) = r1.split_at_mut(hi - lo);
            o0 = rest0;
            o1 = rest1;
            pos = hi;
            let offs = rg.clone().map(|si| inp.span(si).0 - lo).collect();
            jobs.push((rg.clone(), Some(Sink { r0: mine0, r1: mine1, offs, first: rg.start })));
        }
        run_all(jobs)
    } else {
        // arbitrary layout: chunk-local packed buffers, scattered afterwards
        let f = |rg: &Range<usize>| {
            let mut offs = Vec::with_capacity(rg.len());
            let mut tot = 0usize;
            for si in rg.clone() {
                offs.push(tot);
                tot += inp.span(si).1;
            }
            let (mut b0, mut b1) = (vec![0.0; tot], vec![0.0; tot]);
            let mut s = Sink { r0: &mut b0, r1: &mut b1, offs, first: rg.start };
            let t = work(rg.clone(), Some(&mut s));
            (t, b0, b1)
        };
        let res: Vec<(T, Vec<f64>, Vec<f64>)> = if threads == 0 {
            chunks.iter().map(f).collect()
        } else {
            pool(threads).install(|| chunks.par_iter().map(f).collect())
        };
        let mut outv = Vec::with_capacity(res.len());
        for (rg, (t, b0, b1)) in chunks.iter().zip(res) {
            let mut p = 0;
            for si in rg.clone() {
                let (st, tl) = inp.span(si);
                o0[st..st + tl].copy_from_slice(&b0[p..p + tl]);
                o1[st..st + tl].copy_from_slice(&b1[p..p + tl]);
                p += tl;
            }
            outv.push(t);
        }
        outv
    }
}

// ------------------------------------------------------- exact scalar kernel

#[inline(always)]
pub fn lik_k<D: Obs, R: Res>(inp: &Input<D, R>, m: &Model, pos: usize) -> (f64, f64) {
    let (mut l0, mut l1) = (1.0f64, 1.0f64);
    for n in 0..inp.k {
        let l = &m.lik[n][inp.data[n * inp.n + pos].code()];
        l0 *= l[0];
        l1 *= l[1];
    }
    (l0, l1)
}

/// Process sequences `seqs`, accumulating into `c` in exactly the C++ order.
#[inline(always)]
fn process<D: Obs, R: Res, const K1: bool, const R1: bool>(
    inp: &Input<D, R>,
    m: &Model,
    seqs: Range<usize>,
    c: &mut Counts,
    mut sink: Option<&mut Sink>,
    s0: &mut Vec<f64>,
    s1: &mut Vec<f64>,
) {
    let [i0, i1] = m.init;
    let lk = m.lik[0];
    let a_r1 = m.a[0];
    // local accumulators for the K1/R1 case (same addition order as C++)
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
        let (a0, a1): (&mut [f64], &mut [f64]) = match sink.as_mut() {
            Some(s) => s.rows(si, tlen),
            None => {
                if s0.len() < tlen {
                    s0.resize(tlen, 0.0);
                    s1.resize(tlen, 0.0);
                }
                (&mut s0[..tlen], &mut s1[..tlen])
            }
        };
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
        let emit = |em: &mut [[f64; 2]; 3], c: &mut Counts, t: usize, g0: f64, g1: f64| {
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
        emit(&mut em, c, tlen - 1, g0, g1);
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
            // raw layout [j][i]: idx 0=(j0,i0) 1=(j0,i1) 2=(j1,i0) 3=(j1,i1)
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
            emit(&mut em, c, t, g0, g1);
        }
        init[0] += g0;
        init[1] += g1;
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

fn dispatch_process<D: Obs, R: Res>(
    inp: &Input<D, R>,
    m: &Model,
    seqs: Range<usize>,
    c: &mut Counts,
    sink: Option<&mut Sink>,
    s0: &mut Vec<f64>,
    s1: &mut Vec<f64>,
) {
    match (inp.k == 1, m.a.len() == 1) {
        (true, true) => process::<D, R, true, true>(inp, m, seqs, c, sink, s0, s1),
        (true, false) => process::<D, R, true, false>(inp, m, seqs, c, sink, s0, s1),
        (false, true) => process::<D, R, false, true>(inp, m, seqs, c, sink, s0, s1),
        (false, false) => process::<D, R, false, false>(inp, m, seqs, c, sink, s0, s1),
    }
}

/// threads == 0: one running accumulator over all sequences (C++ serial order).
/// threads >= 1: deterministic chunked reduction (result independent of threads).
pub fn e_step<D: Obs, R: Res>(
    inp: &Input<D, R>,
    m: &Model,
    threads: usize,
    chunk_attempts: usize,
    out: Option<(&mut [f64], &mut [f64])>,
) -> Counts {
    let (nr, k) = (m.a.len(), inp.k);
    let chunks = if threads == 0 { std::iter::once(0..inp.starts.len()).collect() } else { plan_chunks(inp.lengths, chunk_attempts) };
    let parts = drive(inp, &chunks, threads, out, |rg, sink| {
        let mut c = Counts::zero(nr, k);
        let (mut s0, mut s1) = (Vec::new(), Vec::new());
        dispatch_process(inp, m, rg, &mut c, sink, &mut s0, &mut s1);
        c
    });
    reduce(&parts, nr, k)
}

// ------------------------------------------------------------------ predict

#[inline(always)]
fn predict_seqs<D: Obs, R: Res, const K1: bool, const R1: bool>(
    inp: &Input<D, R>,
    m: &Model,
    seqs: Range<usize>,
    sink: &mut Sink,
) {
    let [i0, i1] = m.init;
    let lk = m.lik[0];
    for si in seqs {
        let (st, tlen) = inp.span(si);
        let (o0, o1) = sink.rows(si, tlen);
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
    }
}

pub fn predict<D: Obs, R: Res>(
    inp: &Input<D, R>,
    m: &Model,
    threads: usize,
    chunk_attempts: usize,
    out: (&mut [f64], &mut [f64]),
) {
    let chunks = if threads == 0 { std::iter::once(0..inp.starts.len()).collect() } else { plan_chunks(inp.lengths, chunk_attempts) };
    drive(inp, &chunks, threads, Some(out), |rg, sink| {
        let s = sink.expect("predict always has an output");
        match (inp.k == 1, m.a.len() == 1) {
            (true, true) => predict_seqs::<D, R, true, true>(inp, m, rg, s),
            (true, false) => predict_seqs::<D, R, true, false>(inp, m, rg, s),
            (false, true) => predict_seqs::<D, R, false, true>(inp, m, rg, s),
            (false, false) => predict_seqs::<D, R, false, false>(inp, m, rg, s),
        }
    });
}
