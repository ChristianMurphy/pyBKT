//! SIMD-across-students E-step (K == 1 only).
//!
//! L sequences are processed in lockstep in structure-of-arrays form so that the
//! latency-bound per-sequence recurrences overlap and run in SIMD lanes. Within
//! a chunk, sequences are ordered by length (descending, ties by index) and
//! grouped L at a time; lanes whose sequence is shorter than the group's longest
//! are masked (selects), and a group never has a lane that is idle for longer
//! than the length difference to its neighbour in the sorted order.
//!
//! The same algorithm is written once against the `LaneOps` trait and
//! instantiated with
//!   * `Plain<L>`: plain `[f64; L]` arrays, left to LLVM's autovectorizer, and
//!   * `Fs2/Fs4/Fs8<S>`: explicit `fearless_simd` f64x2/f64x4/f64x8 vectors.
//! Both can be run under `fearless_simd::dispatch!` (runtime ISA selection:
//! SSE2 / SSE4.2 / AVX2 / AVX-512 on x86-64, NEON on aarch64) without any
//! `unsafe` in this crate.
//!
//! Log-likelihood: running product of the normalizers per lane with the binary
//! exponent split off every step (one ln() per sequence instead of per attempt).
//! Accumulation order differs from the C++ (per-lane partial sums), so results
//! are deterministic but not bit-identical to C++ serial.

use crate::kernel::{drive, plan_chunks, reduce, Counts, Input, Model, Obs, Res, Sink};
use fearless_simd::{dispatch, f64x2, f64x4, f64x8, mask64x2, mask64x4, mask64x8, prelude::*, u64x2, u64x4, u64x8, Level};
use std::ops::Range;

const EXP_MASK: u64 = 0x7ff << 52;
const MAXL: usize = 8;

pub trait LaneOps: Copy {
    const L: usize;
    type V: Copy;
    type U: Copy;
    type M: Copy;
    fn splat(self, x: f64) -> Self::V;
    fn load(self, s: &[f64]) -> Self::V;
    fn store(self, v: Self::V, s: &mut [f64]);
    fn add(self, a: Self::V, b: Self::V) -> Self::V;
    fn mul(self, a: Self::V, b: Self::V) -> Self::V;
    fn div(self, a: Self::V, b: Self::V) -> Self::V;
    fn lt(self, a: Self::V, b: Self::V) -> Self::M;
    fn eq(self, a: Self::V, b: Self::V) -> Self::M;
    fn ge(self, a: Self::V, b: Self::V) -> Self::M;
    fn sel(self, m: Self::M, a: Self::V, b: Self::V) -> Self::V;
    fn bits(self, v: Self::V) -> Self::U;
    fn from_bits(self, u: Self::U) -> Self::V;
    fn uand(self, a: Self::U, k: u64) -> Self::U;
    fn uor(self, a: Self::U, k: u64) -> Self::U;
    fn ushr52(self, a: Self::U) -> Self::U;
    fn uadd(self, a: Self::U, b: Self::U) -> Self::U;
    fn usel(self, m: Self::M, a: Self::U, b: Self::U) -> Self::U;
    fn usplat(self, x: u64) -> Self::U;
    fn ustore(self, u: Self::U, s: &mut [u64]);
}

/// Plain arrays; vectorization left to LLVM.
#[derive(Clone, Copy)]
pub struct Plain<const L: usize>;

impl<const L: usize> LaneOps for Plain<L> {
    const L: usize = L;
    type V = [f64; L];
    type U = [u64; L];
    type M = [bool; L];
    #[inline(always)]
    fn splat(self, x: f64) -> [f64; L] {
        [x; L]
    }
    #[inline(always)]
    fn load(self, s: &[f64]) -> [f64; L] {
        std::array::from_fn(|i| s[i])
    }
    #[inline(always)]
    fn store(self, v: [f64; L], s: &mut [f64]) {
        s[..L].copy_from_slice(&v)
    }
    #[inline(always)]
    fn add(self, a: [f64; L], b: [f64; L]) -> [f64; L] {
        std::array::from_fn(|i| a[i] + b[i])
    }
    #[inline(always)]
    fn mul(self, a: [f64; L], b: [f64; L]) -> [f64; L] {
        std::array::from_fn(|i| a[i] * b[i])
    }
    #[inline(always)]
    fn div(self, a: [f64; L], b: [f64; L]) -> [f64; L] {
        std::array::from_fn(|i| a[i] / b[i])
    }
    #[inline(always)]
    fn lt(self, a: [f64; L], b: [f64; L]) -> [bool; L] {
        std::array::from_fn(|i| a[i] < b[i])
    }
    #[inline(always)]
    fn eq(self, a: [f64; L], b: [f64; L]) -> [bool; L] {
        std::array::from_fn(|i| a[i] == b[i])
    }
    #[inline(always)]
    fn ge(self, a: [f64; L], b: [f64; L]) -> [bool; L] {
        std::array::from_fn(|i| a[i] >= b[i])
    }
    #[inline(always)]
    fn sel(self, m: [bool; L], a: [f64; L], b: [f64; L]) -> [f64; L] {
        std::array::from_fn(|i| if m[i] { a[i] } else { b[i] })
    }
    #[inline(always)]
    fn bits(self, v: [f64; L]) -> [u64; L] {
        v.map(f64::to_bits)
    }
    #[inline(always)]
    fn from_bits(self, u: [u64; L]) -> [f64; L] {
        u.map(f64::from_bits)
    }
    #[inline(always)]
    fn uand(self, a: [u64; L], k: u64) -> [u64; L] {
        a.map(|x| x & k)
    }
    #[inline(always)]
    fn uor(self, a: [u64; L], k: u64) -> [u64; L] {
        a.map(|x| x | k)
    }
    #[inline(always)]
    fn ushr52(self, a: [u64; L]) -> [u64; L] {
        a.map(|x| x >> 52)
    }
    #[inline(always)]
    fn uadd(self, a: [u64; L], b: [u64; L]) -> [u64; L] {
        std::array::from_fn(|i| a[i] + b[i])
    }
    #[inline(always)]
    fn usel(self, m: [bool; L], a: [u64; L], b: [u64; L]) -> [u64; L] {
        std::array::from_fn(|i| if m[i] { a[i] } else { b[i] })
    }
    #[inline(always)]
    fn usplat(self, x: u64) -> [u64; L] {
        [x; L]
    }
    #[inline(always)]
    fn ustore(self, u: [u64; L], s: &mut [u64]) {
        s[..L].copy_from_slice(&u)
    }
}

macro_rules! fearless_ops {
    ($name:ident, $l:expr, $fv:ident, $uv:ident, $mv:ident) => {
        /// Explicit fearless_simd vectors.
        #[derive(Clone, Copy)]
        pub struct $name<S: Simd>(pub S);
        impl<S: Simd> LaneOps for $name<S> {
            const L: usize = $l;
            type V = $fv<S>;
            type U = $uv<S>;
            type M = $mv<S>;
            #[inline(always)]
            fn splat(self, x: f64) -> $fv<S> {
                $fv::splat(self.0, x)
            }
            #[inline(always)]
            fn load(self, s: &[f64]) -> $fv<S> {
                $fv::from_slice(self.0, &s[..$l])
            }
            #[inline(always)]
            fn store(self, v: $fv<S>, s: &mut [f64]) {
                v.store_slice(&mut s[..$l])
            }
            #[inline(always)]
            fn add(self, a: $fv<S>, b: $fv<S>) -> $fv<S> {
                a + b
            }
            #[inline(always)]
            fn mul(self, a: $fv<S>, b: $fv<S>) -> $fv<S> {
                a * b
            }
            #[inline(always)]
            fn div(self, a: $fv<S>, b: $fv<S>) -> $fv<S> {
                a / b
            }
            #[inline(always)]
            fn lt(self, a: $fv<S>, b: $fv<S>) -> $mv<S> {
                a.simd_lt(b)
            }
            #[inline(always)]
            fn eq(self, a: $fv<S>, b: $fv<S>) -> $mv<S> {
                a.simd_eq(b)
            }
            #[inline(always)]
            fn ge(self, a: $fv<S>, b: $fv<S>) -> $mv<S> {
                a.simd_ge(b)
            }
            #[inline(always)]
            fn sel(self, m: $mv<S>, a: $fv<S>, b: $fv<S>) -> $fv<S> {
                m.select(a, b)
            }
            #[inline(always)]
            fn bits(self, v: $fv<S>) -> $uv<S> {
                v.bitcast()
            }
            #[inline(always)]
            fn from_bits(self, u: $uv<S>) -> $fv<S> {
                u.bitcast()
            }
            #[inline(always)]
            fn uand(self, a: $uv<S>, k: u64) -> $uv<S> {
                a & k
            }
            #[inline(always)]
            fn uor(self, a: $uv<S>, k: u64) -> $uv<S> {
                a | k
            }
            #[inline(always)]
            fn ushr52(self, a: $uv<S>) -> $uv<S> {
                a >> 52u32
            }
            #[inline(always)]
            fn uadd(self, a: $uv<S>, b: $uv<S>) -> $uv<S> {
                a + b
            }
            #[inline(always)]
            fn usel(self, m: $mv<S>, a: $uv<S>, b: $uv<S>) -> $uv<S> {
                m.select(a, b)
            }
            #[inline(always)]
            fn usplat(self, x: u64) -> $uv<S> {
                $uv::splat(self.0, x)
            }
            #[inline(always)]
            fn ustore(self, u: $uv<S>, s: &mut [u64]) {
                u.store_slice(&mut s[..$l])
            }
        }
    };
}
fearless_ops!(Fs2, 2, f64x2, u64x2, mask64x2);
fearless_ops!(Fs4, 4, f64x4, u64x4, mask64x4);
fearless_ops!(Fs8, 8, f64x8, u64x8, mask64x8);

#[derive(Default)]
pub struct LaneScratch {
    b0: Vec<f64>,
    b1: Vec<f64>,
    order: Vec<usize>,
}

/// exact scalar log-likelihood of one sequence (fallback for degenerate norms)
fn seq_ll_exact<D: Obs, R: Res, const R1: bool>(inp: &Input<D, R>, m: &Model, st: usize, tlen: usize) -> f64 {
    let [i0, i1] = m.init;
    let lk = m.lik[0];
    let lik = |t: usize| lk[inp.data[st + t].code()];
    let amat = |t: usize| if R1 { m.a[0] } else { m.a[(inp.res[st + t].val() - 1) as usize] };
    let l = lik(0);
    let mut x0 = i0 * l[0];
    let mut x1 = i1 * l[1];
    let mut norm = x0 + x1;
    x0 /= norm;
    x1 /= norm;
    let mut ll = norm.ln();
    for t in 0..tlen - 1 {
        let a = amat(t);
        let p0 = a[0] * x0 + a[1] * x1;
        let p1 = a[2] * x0 + a[3] * x1;
        let l = lik(t + 1);
        x0 = p0 * l[0];
        x1 = p1 * l[1];
        norm = x0 + x1;
        x0 /= norm;
        x1 /= norm;
        ll += norm.ln();
    }
    ll
}

#[inline(always)]
fn hsum(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |s, &x| s + x)
}

#[inline(always)]
pub fn lanes_kernel<O: LaneOps, D: Obs, R: Res, const R1: bool>(
    o: O,
    inp: &Input<D, R>,
    m: &Model,
    seqs: Range<usize>,
    c: &mut Counts,
    mut sink: Option<&mut Sink>,
    sc: &mut LaneScratch,
) {
    let ln = O::L;
    let nr = m.a.len();
    let mut order = std::mem::take(&mut sc.order);
    order.clear();
    order.extend(seqs);
    order.sort_by_key(|&s| (std::cmp::Reverse(inp.lengths[s]), s));
    let lk = m.lik[0];
    let [i0, i1] = m.init;
    let z = o.splat(0.0);
    let one = o.splat(1.0);
    let two = o.splat(2.0);
    let a1r = m.a[0];
    let av1 = [o.splat(a1r[0]), o.splat(a1r[1]), o.splat(a1r[2]), o.splat(a1r[3])];
    // accumulators
    let mut tr_v = [z; 4];
    let mut tr_s = vec![[[0.0f64; MAXL]; 4]; if R1 { 0 } else { nr }];
    let mut em = [z; 4]; // (code1,s0),(code1,s1),(code2,s0),(code2,s1)
    let mut ini = [z; 2];
    let mut ll_lane = [0.0f64; MAXL];

    for grp in order.chunks(ln) {
        let nl = grp.len();
        let mut st = [0usize; MAXL];
        let mut tl = [0usize; MAXL];
        let mut lim = [0usize; MAXL];
        let mut tlf = [0.0f64; MAXL];
        let mut off = [0usize; MAXL];
        for l in 0..ln {
            let s = grp[l.min(nl - 1)];
            st[l] = inp.span(s).0;
            tl[l] = if l < nl { inp.span(s).1 } else { 0 }; // dummy lanes: never active
            lim[l] = tl[l].max(1) - 1;
            tlf[l] = tl[l] as f64;
            if let Some(sk) = sink.as_deref() {
                off[l] = sk.offs[s - sk.first];
            }
        }
        let tlv = o.load(&tlf);
        let tmax = tl[0];
        if sc.b0.len() < tmax * ln {
            sc.b0.resize(tmax * ln, 0.0);
            sc.b1.resize(tmax * ln, 0.0);
        }
        // ---- forward
        let mut x0 = z;
        let mut x1 = z;
        let mut lp = one;
        let mut ec = o.usplat(0);
        let mut bad = z;
        let minpos = o.splat(f64::MIN_POSITIVE);
        let inf = o.splat(f64::INFINITY);
        for t in 0..tmax {
            let mut l0a = [0.0f64; MAXL];
            let mut l1a = [0.0f64; MAXL];
            let mut aa = [[0.0f64; MAXL]; 4];
            for l in 0..ln {
                let pos = st[l] + t.min(lim[l]);
                let lv = lk[inp.data[pos].code()];
                l0a[l] = lv[0];
                l1a[l] = lv[1];
                if !R1 && t > 0 {
                    let am = m.a[(inp.res[st[l] + (t - 1).min(lim[l])].val() - 1) as usize];
                    for e in 0..4 {
                        aa[e][l] = am[e];
                    }
                }
            }
            let l0 = o.load(&l0a);
            let l1 = o.load(&l1a);
            let (n0, n1) = if t == 0 {
                (o.mul(o.splat(i0), l0), o.mul(o.splat(i1), l1))
            } else {
                let a = if R1 { av1 } else { [o.load(&aa[0]), o.load(&aa[1]), o.load(&aa[2]), o.load(&aa[3])] };
                (
                    o.mul(o.add(o.mul(a[0], x0), o.mul(a[1], x1)), l0),
                    o.mul(o.add(o.mul(a[2], x0), o.mul(a[3], x1)), l1),
                )
            };
            let norm = o.add(n0, n1);
            x0 = o.div(n0, norm);
            x1 = o.div(n1, norm);
            let act = o.lt(o.splat(t as f64), tlv);
            let g1 = o.ge(norm, minpos);
            let g2 = o.lt(norm, inf);
            bad = o.sel(act, o.sel(g1, o.sel(g2, bad, one), one), bad);
            let bits = o.bits(o.mul(lp, norm));
            let e = o.ushr52(o.uand(bits, EXP_MASK));
            let mant = o.from_bits(o.uor(o.uand(bits, !EXP_MASK), 1023u64 << 52));
            lp = o.sel(act, mant, lp);
            ec = o.usel(act, o.uadd(ec, e), ec);
            let b = t * ln;
            o.store(x0, &mut sc.b0[b..b + ln]);
            o.store(x1, &mut sc.b1[b..b + ln]);
            if let Some(sk) = sink.as_deref_mut() {
                for l in 0..nl {
                    if t < tl[l] {
                        sk.r0[off[l] + t] = sc.b0[b + l];
                        sk.r1[off[l] + t] = sc.b1[b + l];
                    }
                }
            }
        }
        let mut lpa = [0.0f64; MAXL];
        let mut eca = [0u64; MAXL];
        let mut bada = [0.0f64; MAXL];
        o.store(lp, &mut lpa);
        o.ustore(ec, &mut eca);
        o.store(bad, &mut bada);
        for l in 0..nl {
            ll_lane[l] += if bada[l] != 0.0 {
                seq_ll_exact::<D, R, R1>(inp, m, st[l], tl[l])
            } else {
                lpa[l].ln() + (eca[l] as f64 - 1023.0 * tl[l] as f64) * std::f64::consts::LN_2
            };
        }
        // ---- backward
        let mut g0 = z;
        let mut g1 = z;
        for t in (0..tmax).rev() {
            let b = t * ln;
            let y0 = o.load(&sc.b0[b..b + ln]);
            let y1 = o.load(&sc.b1[b..b + ln]);
            let mut aa = [[0.0f64; MAXL]; 4];
            let mut rr = [0usize; MAXL];
            let mut ca = [0.0f64; MAXL];
            for l in 0..ln {
                let pos = st[l] + t.min(lim[l]);
                ca[l] = inp.data[pos].code() as f64;
                if !R1 {
                    rr[l] = (inp.res[pos].val() - 1) as usize;
                    let am = m.a[rr[l]];
                    for e in 0..4 {
                        aa[e][l] = am[e];
                    }
                }
            }
            let a = if R1 { av1 } else { [o.load(&aa[0]), o.load(&aa[1]), o.load(&aa[2]), o.load(&aa[3])] };
            let p0 = o.add(o.mul(a[0], y0), o.mul(a[1], y1));
            let p1 = o.add(o.mul(a[2], y0), o.mul(a[3], y1));
            let r0 = o.div(g0, p0);
            let r1 = o.div(g1, p1);
            let nz = |v: O::V| o.sel(o.eq(v, v), v, z);
            let t1 = o.splat((t + 1) as f64);
            let step = o.lt(t1, tlv);
            let start = o.eq(t1, tlv);
            let q00 = o.sel(step, nz(o.mul(o.mul(a[0], y0), r0)), z);
            let q01 = o.sel(step, nz(o.mul(o.mul(a[1], y1), r0)), z);
            let q10 = o.sel(step, nz(o.mul(o.mul(a[2], y0), r1)), z);
            let q11 = o.sel(step, nz(o.mul(o.mul(a[3], y1), r1)), z);
            g0 = o.sel(step, o.add(q00, q10), o.sel(start, y0, z));
            g1 = o.sel(step, o.add(q01, q11), o.sel(start, y1, z));
            // inactive lanes have g == 0, so they add nothing below
            let cv = o.load(&ca);
            let c1 = o.eq(cv, one);
            let c2 = o.eq(cv, two);
            em[0] = o.add(em[0], o.sel(c1, g0, z));
            em[1] = o.add(em[1], o.sel(c1, g1, z));
            em[2] = o.add(em[2], o.sel(c2, g0, z));
            em[3] = o.add(em[3], o.sel(c2, g1, z));
            // raw layout [j][i]
            if R1 {
                tr_v[0] = o.add(tr_v[0], q00);
                tr_v[1] = o.add(tr_v[1], q10);
                tr_v[2] = o.add(tr_v[2], q01);
                tr_v[3] = o.add(tr_v[3], q11);
            } else {
                let mut qa = [[0.0f64; MAXL]; 4];
                o.store(q00, &mut qa[0]);
                o.store(q10, &mut qa[1]);
                o.store(q01, &mut qa[2]);
                o.store(q11, &mut qa[3]);
                for l in 0..ln {
                    for e in 0..4 {
                        tr_s[rr[l]][e][l] += qa[e][l];
                    }
                }
            }
        }
        ini[0] = o.add(ini[0], g0);
        ini[1] = o.add(ini[1], g1);
    }
    sc.order = order;
    // reduce lanes in fixed order
    let mut tmp = [0.0f64; MAXL];
    let mut vsum = |v: O::V| {
        o.store(v, &mut tmp);
        hsum(&tmp[..ln])
    };
    if R1 {
        for e in 0..4 {
            c.trans[e] += vsum(tr_v[e]);
        }
    } else {
        for r in 0..nr {
            for e in 0..4 {
                c.trans[r * 4 + e] += hsum(&tr_s[r][e][..ln]);
            }
        }
    }
    for e in 0..4 {
        c.em[e] += vsum(em[e]);
    }
    c.init[0] += vsum(ini[0]);
    c.init[1] += vsum(ini[1]);
    c.ll += hsum(&ll_lane[..ln]);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaneImpl {
    /// plain arrays, compiled for the build's baseline target only
    Autovec,
    /// plain arrays, inside fearless_simd::dispatch! (runtime ISA selection)
    AutovecDispatch,
    /// explicit fearless_simd vectors, dispatched at runtime
    Fearless,
}

#[inline(always)]
fn kern<O: LaneOps, D: Obs, R: Res>(
    o: O,
    inp: &Input<D, R>,
    m: &Model,
    seqs: Range<usize>,
    c: &mut Counts,
    sink: Option<&mut Sink>,
    sc: &mut LaneScratch,
) {
    if m.a.len() == 1 {
        lanes_kernel::<O, D, R, true>(o, inp, m, seqs, c, sink, sc)
    } else {
        lanes_kernel::<O, D, R, false>(o, inp, m, seqs, c, sink, sc)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_chunk<D: Obs, R: Res>(
    imp: LaneImpl,
    level: Level,
    lanes: usize,
    inp: &Input<D, R>,
    m: &Model,
    seqs: Range<usize>,
    c: &mut Counts,
    sink: Option<&mut Sink>,
    sc: &mut LaneScratch,
) {
    match imp {
        LaneImpl::Autovec => match lanes {
            2 => kern(Plain::<2>, inp, m, seqs, c, sink, sc),
            4 => kern(Plain::<4>, inp, m, seqs, c, sink, sc),
            _ => kern(Plain::<8>, inp, m, seqs, c, sink, sc),
        },
        LaneImpl::AutovecDispatch => dispatch!(level, _simd => match lanes {
            2 => kern(Plain::<2>, inp, m, seqs, c, sink, sc),
            4 => kern(Plain::<4>, inp, m, seqs, c, sink, sc),
            _ => kern(Plain::<8>, inp, m, seqs, c, sink, sc),
        }),
        LaneImpl::Fearless => dispatch!(level, simd => match lanes {
            2 => kern(Fs2(simd), inp, m, seqs, c, sink, sc),
            4 => kern(Fs4(simd), inp, m, seqs, c, sink, sc),
            _ => kern(Fs8(simd), inp, m, seqs, c, sink, sc),
        }),
    }
}

/// Runtime SIMD level, optionally capped with BKT_RS_ISA=sse2|sse4_2|avx2|avx512.
pub fn level() -> Level {
    let l = Level::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match std::env::var("BKT_RS_ISA").as_deref() {
            Ok("sse2") => return Level::baseline(),
            Ok("sse4_2") => return l.as_sse4_2().map(Level::Sse4_2).unwrap_or(l),
            Ok("avx2") => return l.as_avx2().map(Level::Avx2).unwrap_or(l),
            _ => {}
        }
    }
    l
}

pub fn level_name(l: Level) -> String {
    format!("{l:?}")
}

/// Lockstep variant. Always chunked (threads == 0 runs the chunks on the calling
/// thread), so the result is identical for every thread count. Requires K == 1.
#[allow(clippy::too_many_arguments)]
pub fn e_step_lanes<D: Obs, R: Res>(
    inp: &Input<D, R>,
    m: &Model,
    lanes: usize,
    imp: LaneImpl,
    level: Level,
    threads: usize,
    chunk_attempts: usize,
    out: Option<(&mut [f64], &mut [f64])>,
) -> Counts {
    assert!(inp.k == 1 && matches!(lanes, 2 | 4 | 8));
    let (nr, k) = (m.a.len(), inp.k);
    let chunks = plan_chunks(inp.lengths, chunk_attempts);
    let parts = drive(inp, &chunks, threads, out, |rg, sink| {
        let mut c = Counts::zero(nr, k);
        let mut sc = LaneScratch::default();
        run_chunk(imp, level, lanes, inp, m, rg, &mut c, sink, &mut sc);
        c
    });
    reduce(&parts, nr, k)
}

/// Lane utilization of the length-sorted grouping: sum(lengths) / sum(L * group max).
pub fn utilization(lengths: &[i64], lanes: usize, chunk_attempts: usize, sort: bool) -> f64 {
    let mut used = 0u64;
    let mut slots = 0u64;
    for rg in plan_chunks(lengths, chunk_attempts) {
        let mut v: Vec<i64> = lengths[rg].to_vec();
        if sort {
            v.sort_by(|a, b| b.cmp(a));
        }
        for g in v.chunks(lanes) {
            used += g.iter().sum::<i64>() as u64;
            slots += (*g.iter().max().unwrap() as u64) * lanes as u64;
        }
    }
    used as f64 / slots as f64
}
