//! Ties kernels, chunking, lanes, threads and output emitters together.

use crate::kernel::{
    plan_chunks, predict_any, process_any, reduce, run_inline, run_threads, BoxEmit, Counts, Emit, Input, Model, Obs, Res,
    Scratch,
};
use crate::lanes::{self, LaneImpl, LaneScratch};
use std::ops::Range;

#[derive(Clone, Copy, Debug)]
pub struct Opts {
    pub threads: usize,
    pub chunk: usize,
    pub lanes: usize,
    pub imp: LaneImpl,
}

pub struct Job<'a, D, R> {
    pub inp: Input<'a, D, R>,
    pub m: Model,
    pub opts: Opts,
    pub chunks: Vec<Range<usize>>,
    pub predict: bool,
}

impl<'a, D: Obs, R: Res> Job<'a, D, R> {
    /// threads == 0 and lanes == 0: one chunk = the exact C++ serial order.
    /// Otherwise the fixed chunk plan (independent of the thread count).
    pub fn new(inp: Input<'a, D, R>, m: Model, opts: Opts, predict: bool) -> Self {
        let chunks = if opts.threads == 0 && (opts.lanes == 0 || predict) {
            std::iter::once(0..inp.starts.len()).collect()
        } else {
            plan_chunks(inp.lengths, opts.chunk.max(1))
        };
        Job { inp, m, opts, chunks, predict }
    }

    fn work(&self, i: usize, e: &mut dyn Emit) -> Counts {
        let mut c = Counts::zero(self.m.a.len(), self.inp.k);
        let rg = self.chunks[i].clone();
        if self.predict {
            predict_any(&self.inp, &self.m, rg, e, &mut Scratch::new());
        } else if self.opts.lanes > 0 {
            lanes::run_chunk(self.opts.imp, self.opts.lanes, &self.inp, &self.m, rg, &mut c, e, &mut LaneScratch::default());
        } else {
            process_any(&self.inp, &self.m, rg, &mut c, e, &mut Scratch::new());
        }
        c
    }

    fn finish(&self, parts: Vec<Counts>) -> Counts {
        reduce(&parts, self.m.a.len(), self.inp.k)
    }
}

/// Object-safe view of a job so that output handling is written once for
/// shareable (Sync, copied) inputs and GIL-bound borrowed inputs.
pub trait Runner {
    fn n(&self) -> usize;
    fn nchunks(&self) -> usize;
    fn ordered(&self) -> bool;
    fn spans(&self) -> Vec<(usize, usize)>;
    /// will chunks run on worker threads (emitters must then be Send)?
    fn threaded(&self) -> bool;
    fn run<'b>(&self, emits: Vec<BoxEmit<'b>>, on_done: &mut dyn FnMut(&mut dyn Emit)) -> Counts;
    fn run_local<'b>(&self, emits: Vec<Box<dyn Emit + 'b>>, on_done: &mut dyn FnMut(&mut dyn Emit)) -> Counts;
}

/// inputs that can be shared with worker threads
pub struct Par<'a, D, R>(pub Job<'a, D, R>);
/// inputs bound to the GIL (e.g. &[ReadOnlyCell<T>]): calling thread only
pub struct Seq<'a, D, R>(pub Job<'a, D, R>);

macro_rules! common {
    () => {
        fn n(&self) -> usize {
            self.0.inp.n
        }
        fn nchunks(&self) -> usize {
            self.0.chunks.len()
        }
        fn ordered(&self) -> bool {
            self.0.chunks.len() <= 1 || self.0.inp.ordered_disjoint()
        }
        fn spans(&self) -> Vec<(usize, usize)> {
            self.0.inp.chunk_spans(&self.0.chunks)
        }
        fn run_local<'b>(&self, emits: Vec<Box<dyn Emit + 'b>>, on_done: &mut dyn FnMut(&mut dyn Emit)) -> Counts {
            let parts = run_inline(emits, |i, e| self.0.work(i, e), on_done);
            self.0.finish(parts)
        }
    };
}

impl<D: Obs + Sync, R: Res + Sync> Runner for Par<'_, D, R> {
    common!();
    fn threaded(&self) -> bool {
        self.0.opts.threads >= 2 && self.0.chunks.len() >= 2
    }
    fn run<'b>(&self, emits: Vec<BoxEmit<'b>>, on_done: &mut dyn FnMut(&mut dyn Emit)) -> Counts {
        if self.threaded() {
            let parts = run_threads(self.0.opts.threads, emits, |i, e| self.0.work(i, e), on_done);
            self.0.finish(parts)
        } else {
            self.run_local(emits.into_iter().map(|b| b as Box<dyn Emit>).collect(), on_done)
        }
    }
}

impl<D: Obs, R: Res> Runner for Seq<'_, D, R> {
    common!();
    fn threaded(&self) -> bool {
        false
    }
    fn run<'b>(&self, emits: Vec<BoxEmit<'b>>, on_done: &mut dyn FnMut(&mut dyn Emit)) -> Counts {
        self.run_local(emits.into_iter().map(|b| b as Box<dyn Emit>).collect(), on_done)
    }
}
