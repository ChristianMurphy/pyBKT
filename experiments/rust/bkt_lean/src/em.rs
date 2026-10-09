//! EM loop and M-step in Rust, mirroring pyBKT fit/EM_fit.py and fit/M_step.py
//! (with fixed = {}), so a whole fit marshals its inputs once.

use crate::estep::{Job, Opts, Par, Runner};
use crate::kernel::{Counts, Emit, Input, Model, NoEmit, Obs, Params, Res};

pub struct MOut {
    pub params: Params,
    /// As[r][to][from], R*4
    pub as_: Vec<f64>,
    /// emissions[k][state][obs], K*4
    pub emissions: Vec<f64>,
    pub pi0: [f64; 2],
}

/// pyBKT M_step.run(model, trans.T, emission.T, init, fixed={}) on the raw C++ layouts.
pub fn m_step(c: &Counts) -> MOut {
    let nr = c.trans.len() / 4;
    let k = c.em.len() / 4;
    let mut as_ = vec![0.0; 4 * nr];
    let (mut learns, mut forgets) = (Vec::with_capacity(nr), Vec::with_capacity(nr));
    for r in 0..nr {
        let raw = &c.trans[4 * r..4 * r + 4]; // [from][to]
        // T[to][from] after EM_fit's transpose
        let mut t = [[raw[0], raw[2]], [raw[1], raw[3]]];
        let fix = |t: &mut [[f64; 2]; 2], from: usize| {
            if t[0][from] + t[1][from] == 0.0 {
                t[0][from] = 0.0;
                t[1][from] = 1.0;
            }
        };
        fix(&mut t, 0);
        fix(&mut t, 1);
        for to in 0..2 {
            for from in 0..2 {
                as_[4 * r + 2 * to + from] = t[to][from] / (t[0][from] + t[1][from]);
            }
        }
        learns.push(as_[4 * r + 2]); // As[r,1,0]
        forgets.push(as_[4 * r + 1]); // As[r,0,1]
    }
    let mut emissions = vec![0.0; 4 * k];
    let (mut guesses, mut slips) = (Vec::with_capacity(k), Vec::with_capacity(k));
    for n in 0..k {
        let raw = &c.em[4 * n..4 * n + 4]; // [obs][state]
        let mut e = [[raw[0], raw[2]], [raw[1], raw[3]]]; // [state][obs]
        for row in e.iter_mut() {
            if row[0] + row[1] == 0.0 {
                *row = [1.0, 1.0];
            }
        }
        for s in 0..2 {
            for o in 0..2 {
                emissions[4 * n + 2 * s + o] = e[s][o] / (e[s][0] + e[s][1]);
            }
        }
        guesses.push(emissions[4 * n + 1]); // emissions[:,0,1]
        slips.push(emissions[4 * n + 2]); // emissions[:,1,0]
    }
    let tot = c.init[0] + c.init[1];
    let pi0 = [c.init[0] / tot, c.init[1] / tot];
    MOut { params: Params { prior: pi0[1], learns, forgets, guesses, slips }, as_, emissions, pi0 }
}

pub struct FitOut {
    pub last: Option<MOut>,
    pub params: Params,
    pub lls: Vec<f64>,
}

/// pyBKT EM_fit: E-step, record ll, stop if i > 1 and |ll_i - ll_{i-1}| < tol,
/// else M-step. `compat_int_ll` truncates ll to an integer like the C++
/// E-step's PyLong_FromLong(total_loglike) does.
pub fn fit<D: Obs + Sync, R: Res + Sync>(
    inp: &Input<'_, D, R>,
    p0: Params,
    opts: Opts,
    max_iter: usize,
    tol: f64,
    compat_int_ll: bool,
) -> FitOut {
    let mut params = p0;
    let mut lls = Vec::new();
    let mut last = None;
    for i in 0..max_iter {
        let view = Input { data: inp.data, k: inp.k, n: inp.n, res: inp.res, starts: inp.starts, lengths: inp.lengths };
        let job = Par(Job::new(view, Model::from_params(&params), opts, false));
        let emits: Vec<Box<dyn Emit + Send>> = (0..job.nchunks()).map(|_| Box::new(NoEmit) as Box<dyn Emit + Send>).collect();
        let c = job.run(emits, &mut |_| {});
        let ll = if compat_int_ll { c.ll.trunc() } else { c.ll };
        lls.push(ll);
        if i > 1 && (ll - lls[i - 1]).abs() < tol {
            break;
        }
        let mo = m_step(&c);
        params = mo.params.clone();
        last = Some(mo);
    }
    FitOut { last, params, lls }
}
