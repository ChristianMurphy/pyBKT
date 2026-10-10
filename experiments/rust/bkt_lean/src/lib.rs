#![forbid(unsafe_code)]
//! bkt_lean: pyBKT E-step / predict / EM fit with pyo3 as the only mandatory
//! dependency (no numpy crate, no rayon). Inputs are read through the Python
//! buffer protocol; outputs go to a bytearray, bytes, or a caller-allocated
//! writable buffer (e.g. np.empty). `--features simd` adds fearless_simd lanes.

mod em;
mod estep;
mod kernel;
mod lanes;

use estep::{Job, Opts, Par, Runner, Seq};
use kernel::{
    split_bytes, split_f64, ByteEmit, F64Emit, CellEmit, Counts, Emit, Input, Model, NoEmit, Obs, PackEmit, Params, Res,
};
use lanes::LaneImpl;
use pyo3::buffer::{Element, PyBuffer, ReadOnlyCell};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict};

// GIL-bound zero-copy views of input buffers
impl Obs for ReadOnlyCell<i8> {
    #[inline(always)]
    fn code(&self) -> usize {
        self.get().code()
    }
}
impl Obs for ReadOnlyCell<i32> {
    #[inline(always)]
    fn code(&self) -> usize {
        self.get().code()
    }
}
impl Obs for ReadOnlyCell<i64> {
    #[inline(always)]
    fn code(&self) -> usize {
        self.get().code()
    }
}
impl Res for ReadOnlyCell<i64> {
    #[inline(always)]
    fn val(&self) -> i64 {
        self.get()
    }
}
impl Res for ReadOnlyCell<i32> {
    #[inline(always)]
    fn val(&self) -> i64 {
        self.get() as i64
    }
}
impl Res for ReadOnlyCell<u16> {
    #[inline(always)]
    fn val(&self) -> i64 {
        self.get() as i64
    }
}

/// (trans, emission, init, loglike, alpha) as returned to Python
type EOut = (Vec<f64>, Vec<f64>, Vec<f64>, f64, Py<PyAny>);

fn f64s(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    match PyBuffer::<f64>::get(obj) {
        Ok(b) => b.to_vec(obj.py()),
        Err(_) => obj.extract::<Vec<f64>>(),
    }
}

fn i64s(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    match PyBuffer::<i64>::get(obj) {
        Ok(b) => b.to_vec(obj.py()),
        Err(_) => obj.extract::<Vec<i64>>(),
    }
}

enum DataBuf {
    I8(PyBuffer<i8>),
    I32(PyBuffer<i32>),
    I64(PyBuffer<i64>),
}
enum ResBuf {
    I64(PyBuffer<i64>),
    I32(PyBuffer<i32>),
    U16(PyBuffer<u16>),
}

fn data_buf(obj: &Bound<'_, PyAny>) -> PyResult<(DataBuf, usize, usize)> {
    let shape = |s: &[usize]| -> PyResult<(usize, usize)> {
        match s {
            [k, n] => Ok((*k, *n)),
            [n] => Ok((1, *n)),
            _ => Err(PyValueError::new_err("data must be 1-D or 2-D (K, N)")),
        }
    };
    if let Ok(b) = PyBuffer::<i8>::get(obj) {
        let (k, n) = shape(b.shape())?;
        return Ok((DataBuf::I8(b), k, n));
    }
    if let Ok(b) = PyBuffer::<i32>::get(obj) {
        let (k, n) = shape(b.shape())?;
        return Ok((DataBuf::I32(b), k, n));
    }
    if let Ok(b) = PyBuffer::<i64>::get(obj) {
        let (k, n) = shape(b.shape())?;
        return Ok((DataBuf::I64(b), k, n));
    }
    Err(PyTypeError::new_err("data must be an int8/int32/int64 buffer"))
}

fn res_buf(obj: &Bound<'_, PyAny>) -> PyResult<ResBuf> {
    if let Ok(b) = PyBuffer::<i64>::get(obj) {
        return Ok(ResBuf::I64(b));
    }
    if let Ok(b) = PyBuffer::<i32>::get(obj) {
        return Ok(ResBuf::I32(b));
    }
    if let Ok(b) = PyBuffer::<u16>::get(obj) {
        return Ok(ResBuf::U16(b));
    }
    Err(PyTypeError::new_err("resources must be an int64/int32/uint16 buffer"))
}

#[derive(Clone, Copy, PartialEq)]
enum InMode {
    /// PyBuffer::to_vec (memcpy) -> owned, Sync, threads allowed
    Copy,
    /// &[ReadOnlyCell<T>] straight from the buffer: zero copy, GIL-bound, calling thread only
    Borrow,
    /// convert once into compact u8 codes / u32 resources (what fit() and Dataset do)
    Codes,
}

#[derive(Clone, Copy, PartialEq)]
enum OutMode {
    None,
    ByteArray,
    Bytes,
    Numpy,
}

struct Call<'py, 'o> {
    params: Params,
    pred: Option<(Vec<f64>, Vec<f64>)>,
    opts: Opts,
    predict: bool,
    out_mode: OutMode,
    out: Option<&'o Bound<'py, PyAny>>,
}

/// Run a job with the requested output mode. Always runs with the GIL held
/// (like the C++ module); worker threads never touch Python.
fn execute<'py>(py: Python<'py>, r: &dyn Runner, mode: OutMode, out: Option<&Bound<'py, PyAny>>) -> PyResult<(Counts, Py<PyAny>)> {
    let n = r.n();
    let nch = r.nchunks();
    let none_emits = || (0..nch).map(|_| Box::new(NoEmit) as Box<dyn Emit + Send>).collect::<Vec<_>>();
    // bytes-like output: split spans when ordered, otherwise pack + scatter
    let run_bytes = |buf: &mut [u8]| -> Counts {
        let (b0, b1) = buf.split_at_mut(8 * n);
        if r.ordered() {
            let emits = split_bytes(b0, b1, &r.spans()).into_iter().map(|e| Box::new(e) as Box<dyn Emit + Send>).collect();
            r.run(emits, &mut |_| {})
        } else {
            let emits = (0..nch).map(|_| Box::new(PackEmit::default()) as Box<dyn Emit + Send>).collect();
            let mut sink = ByteEmit { r0: b0, r1: b1, lo: 0 };
            r.run(emits, &mut |e| e.drain(&mut |st, a0, a1| sink.emit(st, a0, a1)))
        }
    };
    match mode {
        OutMode::None => Ok((r.run(none_emits(), &mut |_| {}), py.None())),
        OutMode::ByteArray => {
            let mut counts = None;
            let ba = PyByteArray::new_with(py, 16 * n, |buf| {
                counts = Some(run_bytes(buf));
                Ok(())
            })?;
            Ok((counts.expect("computed"), ba.into_any().unbind()))
        }
        OutMode::Bytes => {
            // baseline: compute into a Rust Vec<f64>, then copy into a new bytes object
            let mut v = vec![0.0f64; 2 * n];
            let (v0, v1) = v.split_at_mut(n);
            let c = if r.ordered() {
                let emits = split_f64(v0, v1, &r.spans()).into_iter().map(|e| Box::new(e) as Box<dyn Emit + Send>).collect();
                r.run(emits, &mut |_| {})
            } else {
                let emits = (0..nch).map(|_| Box::new(PackEmit::default()) as Box<dyn Emit + Send>).collect();
                let mut sink = F64Emit { r0: v0, r1: v1, lo: 0 };
                r.run(emits, &mut |e| e.drain(&mut |st, a0, a1| sink.emit(st, a0, a1)))
            };
            let b = PyBytes::new_with(py, 16 * n, |buf| {
                for (d, x) in buf.chunks_exact_mut(8).zip(&v) {
                    d.copy_from_slice(&x.to_ne_bytes());
                }
                Ok(())
            })?;
            Ok((c, b.into_any().unbind()))
        }
        OutMode::Numpy => {
            let obj = out.ok_or_else(|| PyValueError::new_err("alpha='numpy' needs out=np.empty((2, N))"))?;
            let buf = PyBuffer::<f64>::get(obj)?;
            let cells = buf
                .as_mut_slice(py)
                .ok_or_else(|| PyValueError::new_err("out must be a writable C-contiguous float64 buffer"))?;
            if cells.len() != 2 * n {
                return Err(PyValueError::new_err("out must have 2*N elements"));
            }
            let (c0, c1) = cells.split_at(n);
            let mut sink = CellEmit { r0: c0, r1: c1 };
            let c = if r.threaded() {
                // workers fill chunk-local buffers; this thread writes them into the
                // numpy buffer as chunks finish (pipelined)
                let emits = (0..nch).map(|_| Box::new(PackEmit::default()) as Box<dyn Emit + Send>).collect();
                r.run(emits, &mut |e| e.drain(&mut |st, a0, a1| sink.emit(st, a0, a1)))
            } else {
                // one thread: write straight into the numpy buffer
                let emits: Vec<Box<dyn Emit + '_>> = (0..nch)
                    .map(|_| Box::new(CellEmit { r0: c0, r1: c1 }) as Box<dyn Emit>)
                    .collect();
                r.run_local(emits, &mut |_| {})
            };
            Ok((c, obj.clone().unbind()))
        }
    }
}

fn job_for<'a, D: Obs, R: Res>(inp: Input<'a, D, R>, call: &Call) -> PyResult<Job<'a, D, R>> {
    let nr = call.params.learns.len();
    if call.params.forgets.len() != nr || call.params.guesses.len() != inp.k || call.params.slips.len() != inp.k {
        return Err(PyValueError::new_err("parameter lengths do not match R / K"));
    }
    inp.validate(nr).map_err(PyValueError::new_err)?;
    if call.opts.lanes > 0 && (inp.k != 1 || !matches!(call.opts.lanes, 2 | 4 | 8)) {
        return Err(PyValueError::new_err("lanes must be 0, 2, 4 or 8 and needs K == 1"));
    }
    let m = match &call.pred {
        Some((pl, pf)) => Model::new(&call.params, pl, pf),
        None => Model::from_params(&call.params),
    };
    Ok(Job::new(inp, m, call.opts, call.predict))
}

#[allow(clippy::too_many_arguments)]
fn run_copy<D: Obs + Sync + Element, R: Res + Sync + Element>(
    py: Python<'_>,
    d: &PyBuffer<D>,
    r: &PyBuffer<R>,
    k: usize,
    n: usize,
    starts: &[i64],
    lengths: &[i64],
    call: &Call,
) -> PyResult<(Counts, Py<PyAny>)> {
    let data = d.to_vec(py)?;
    let res = if call.params.learns.len() > 1 { r.to_vec(py)? } else { Vec::new() };
    let inp = Input { data: &data, k, n, res: &res, starts, lengths };
    let job = Par(job_for(inp, call)?);
    execute(py, &job, call.out_mode, call.out)
}

#[allow(clippy::too_many_arguments)]
fn run_borrow<D: Element, R: Element>(
    py: Python<'_>,
    d: &PyBuffer<D>,
    r: &PyBuffer<R>,
    k: usize,
    n: usize,
    starts: &[i64],
    lengths: &[i64],
    call: &Call,
) -> PyResult<(Counts, Py<PyAny>)>
where
    ReadOnlyCell<D>: Obs,
    ReadOnlyCell<R>: Res,
{
    let data = d.as_slice(py).ok_or_else(|| PyValueError::new_err("data must be C-contiguous"))?;
    let res = r.as_slice(py).ok_or_else(|| PyValueError::new_err("resources must be C-contiguous"))?;
    let inp = Input { data, k, n, res, starts, lengths };
    let job = Seq(job_for(inp, call)?);
    execute(py, &job, call.out_mode, call.out)
}

/// compact copy: data -> u8 codes (0/1/2), resources -> u32 (only when R > 1)
struct Compact {
    codes: Vec<u8>,
    res: Vec<u32>,
    k: usize,
    n: usize,
    starts: Vec<i64>,
    lengths: Vec<i64>,
}

fn compact(py: Python<'_>, data: &Bound<'_, PyAny>, resources: &Bound<'_, PyAny>, starts: Vec<i64>, lengths: Vec<i64>, need_res: bool) -> PyResult<Compact> {
    let (db, k, n) = data_buf(data)?;
    let codes: Vec<u8> = match &db {
        DataBuf::I8(b) => b.as_slice(py).ok_or_else(|| PyValueError::new_err("data must be C-contiguous"))?.iter().map(|c| c.code() as u8).collect(),
        DataBuf::I32(b) => b.as_slice(py).ok_or_else(|| PyValueError::new_err("data must be C-contiguous"))?.iter().map(|c| c.code() as u8).collect(),
        DataBuf::I64(b) => b.as_slice(py).ok_or_else(|| PyValueError::new_err("data must be C-contiguous"))?.iter().map(|c| c.code() as u8).collect(),
    };
    let res: Vec<u32> = if need_res {
        let conv = |v: i64| -> PyResult<u32> { u32::try_from(v).map_err(|_| PyValueError::new_err("resource out of range")) };
        match res_buf(resources)? {
            ResBuf::I64(b) => b.as_slice(py).ok_or_else(|| PyValueError::new_err("resources must be C-contiguous"))?.iter().map(|c| conv(c.get())).collect::<PyResult<_>>()?,
            ResBuf::I32(b) => b.as_slice(py).ok_or_else(|| PyValueError::new_err("resources must be C-contiguous"))?.iter().map(|c| conv(c.get() as i64)).collect::<PyResult<_>>()?,
            ResBuf::U16(b) => b.as_slice(py).ok_or_else(|| PyValueError::new_err("resources must be C-contiguous"))?.iter().map(|c| c.get() as u32).collect(),
        }
    } else {
        Vec::new()
    };
    Ok(Compact { codes, res, k, n, starts, lengths })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_call(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    resources: &Bound<'_, PyAny>,
    starts: &Bound<'_, PyAny>,
    lengths: &Bound<'_, PyAny>,
    input: &str,
    call: &Call,
) -> PyResult<(Counts, Py<PyAny>)> {
    let starts = i64s(starts)?;
    let lengths = i64s(lengths)?;
    let mode = match input {
        "copy" => InMode::Copy,
        "borrow" => InMode::Borrow,
        "codes" => InMode::Codes,
        _ => return Err(PyValueError::new_err("input must be copy, borrow or codes")),
    };
    if mode == InMode::Codes {
        let c = compact(py, data, resources, starts, lengths, call.params.learns.len() > 1)?;
        let inp = Input { data: &c.codes, k: c.k, n: c.n, res: &c.res, starts: &c.starts, lengths: &c.lengths };
        let job = Par(job_for(inp, call)?);
        return execute(py, &job, call.out_mode, call.out);
    }
    let (db, k, n) = data_buf(data)?;
    let rb = res_buf(resources)?;
    macro_rules! go {
        ($d:expr, $r:expr) => {
            if mode == InMode::Copy {
                run_copy(py, $d, $r, k, n, &starts, &lengths, call)
            } else {
                run_borrow(py, $d, $r, k, n, &starts, &lengths, call)
            }
        };
    }
    match (&db, &rb) {
        (DataBuf::I8(d), ResBuf::I64(r)) => go!(d, r),
        (DataBuf::I8(d), ResBuf::I32(r)) => go!(d, r),
        (DataBuf::I8(d), ResBuf::U16(r)) => go!(d, r),
        (DataBuf::I32(d), ResBuf::I64(r)) => go!(d, r),
        (DataBuf::I32(d), ResBuf::I32(r)) => go!(d, r),
        (DataBuf::I32(d), ResBuf::U16(r)) => go!(d, r),
        (DataBuf::I64(d), ResBuf::I64(r)) => go!(d, r),
        (DataBuf::I64(d), ResBuf::I32(r)) => go!(d, r),
        (DataBuf::I64(d), ResBuf::U16(r)) => go!(d, r),
    }
}

fn parse_opts(threads: usize, chunk: usize, lanes: usize, lane_impl: &str) -> PyResult<Opts> {
    let imp = match lane_impl {
        "plain" => LaneImpl::Plain,
        "blend" => LaneImpl::Blend,
        #[cfg(feature = "simd")]
        "plain_dispatch" => LaneImpl::PlainDispatch,
        #[cfg(feature = "simd")]
        "blend_dispatch" => LaneImpl::BlendDispatch,
        #[cfg(feature = "simd")]
        "fearless" => LaneImpl::Fearless,
        _ => return Err(PyValueError::new_err(
            "lane_impl must be 'plain' or 'blend' (or 'plain_dispatch', 'blend_dispatch', 'fearless' with the simd feature)")),
    };
    Ok(Opts { threads, chunk, lanes, imp })
}

fn parse_out(s: &str) -> PyResult<OutMode> {
    Ok(match s {
        "none" => OutMode::None,
        "bytearray" => OutMode::ByteArray,
        "bytes" => OutMode::Bytes,
        "numpy" => OutMode::Numpy,
        _ => return Err(PyValueError::new_err("alpha/out_mode must be none, bytearray, bytes or numpy")),
    })
}

#[allow(clippy::too_many_arguments)]
fn params_from(prior: f64, learns: &Bound<'_, PyAny>, forgets: &Bound<'_, PyAny>, guesses: &Bound<'_, PyAny>, slips: &Bound<'_, PyAny>) -> PyResult<Params> {
    Ok(Params { prior, learns: f64s(learns)?, forgets: f64s(forgets)?, guesses: f64s(guesses)?, slips: f64s(slips)? })
}

/// e_step(data, resources, starts, lengths, prior, learns, forgets, guesses, slips, threads=0,
///        alpha="none", out=None, chunk_attempts=65536, lanes=0, lane_impl="plain", input="copy")
/// -> (trans list R*4 [r][from][to], emission list K*4 [k][obs][state], init list 2, loglike float, alpha)
/// alpha: "none" -> None; "bytearray"/"bytes" -> 16*N bytes (np.frombuffer(..).reshape(2, N));
///        "numpy" -> writes into `out` (writable float64 buffer with 2*N elements) and returns it.
/// input: "copy" (memcpy, threads ok), "borrow" (zero copy, calling thread only), "codes" (u8 codes).
#[pyfunction]
#[pyo3(signature = (data, resources, starts, lengths, prior, learns, forgets, guesses, slips, threads=0, alpha="none", out=None, chunk_attempts=65536, lanes=0, lane_impl="plain", input="copy"))]
#[allow(clippy::too_many_arguments)]
fn e_step<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    resources: &Bound<'py, PyAny>,
    starts: &Bound<'py, PyAny>,
    lengths: &Bound<'py, PyAny>,
    prior: f64,
    learns: &Bound<'py, PyAny>,
    forgets: &Bound<'py, PyAny>,
    guesses: &Bound<'py, PyAny>,
    slips: &Bound<'py, PyAny>,
    threads: usize,
    alpha: &str,
    out: Option<&Bound<'py, PyAny>>,
    chunk_attempts: usize,
    lanes: usize,
    lane_impl: &str,
    input: &str,
) -> PyResult<EOut> {
    let call = Call {
        params: params_from(prior, learns, forgets, guesses, slips)?,
        pred: None,
        opts: parse_opts(threads, chunk_attempts, lanes, lane_impl)?,
        predict: false,
        out_mode: parse_out(alpha)?,
        out,
    };
    let (c, a) = dispatch_call(py, data, resources, starts, lengths, input, &call)?;
    Ok((c.trans, c.em, c.init.to_vec(), c.ll, a))
}

/// predict(data, resources, starts, lengths, prior, learns, forgets, guesses, slips, threads=0,
///         out_mode="bytearray", out=None, chunk_attempts=65536, input="copy",
///         pred_learns=None, pred_forgets=None) -> 16*N bytes / bytearray, or `out`
#[pyfunction]
#[pyo3(signature = (data, resources, starts, lengths, prior, learns, forgets, guesses, slips, threads=0, out_mode="bytearray", out=None, chunk_attempts=65536, input="copy", pred_learns=None, pred_forgets=None))]
#[allow(clippy::too_many_arguments)]
fn predict<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    resources: &Bound<'py, PyAny>,
    starts: &Bound<'py, PyAny>,
    lengths: &Bound<'py, PyAny>,
    prior: f64,
    learns: &Bound<'py, PyAny>,
    forgets: &Bound<'py, PyAny>,
    guesses: &Bound<'py, PyAny>,
    slips: &Bound<'py, PyAny>,
    threads: usize,
    out_mode: &str,
    out: Option<&Bound<'py, PyAny>>,
    chunk_attempts: usize,
    input: &str,
    pred_learns: Option<&Bound<'py, PyAny>>,
    pred_forgets: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let params = params_from(prior, learns, forgets, guesses, slips)?;
    let pl = match pred_learns {
        Some(o) => f64s(o)?,
        None => params.learns.clone(),
    };
    let pf = match pred_forgets {
        Some(o) => f64s(o)?,
        None => params.forgets.clone(),
    };
    if pl.len() != params.learns.len() || pf.len() != params.learns.len() {
        return Err(PyValueError::new_err("pred_learns/pred_forgets length mismatch"));
    }
    let mode = parse_out(out_mode)?;
    if mode == OutMode::None {
        return Err(PyValueError::new_err("predict needs an output"));
    }
    let call = Call { params, pred: Some((pl, pf)), opts: parse_opts(threads, chunk_attempts, 0, "plain")?, predict: true, out_mode: mode, out };
    Ok(dispatch_call(py, data, resources, starts, lengths, input, &call)?.1)
}

/// Inputs converted once to compact u8 codes (and u32 resources when R > 1).
#[pyclass(frozen)]
struct Dataset {
    c: Compact,
}

#[pymethods]
impl Dataset {
    #[new]
    #[pyo3(signature = (data, resources, starts, lengths, num_resources=1))]
    fn new(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        resources: &Bound<'_, PyAny>,
        starts: &Bound<'_, PyAny>,
        lengths: &Bound<'_, PyAny>,
        num_resources: usize,
    ) -> PyResult<Self> {
        Ok(Dataset { c: compact(py, data, resources, i64s(starts)?, i64s(lengths)?, num_resources > 1)? })
    }
    #[getter]
    fn n(&self) -> usize {
        self.c.n
    }
}

/// e_step_ds(ds, prior, learns, forgets, guesses, slips, threads=0, alpha="none", out=None,
///           chunk_attempts=65536, lanes=0, lane_impl="plain") -> same as e_step
#[pyfunction]
#[pyo3(signature = (ds, prior, learns, forgets, guesses, slips, threads=0, alpha="none", out=None, chunk_attempts=65536, lanes=0, lane_impl="plain"))]
#[allow(clippy::too_many_arguments)]
fn e_step_ds<'py>(
    py: Python<'py>,
    ds: &Bound<'py, Dataset>,
    prior: f64,
    learns: &Bound<'py, PyAny>,
    forgets: &Bound<'py, PyAny>,
    guesses: &Bound<'py, PyAny>,
    slips: &Bound<'py, PyAny>,
    threads: usize,
    alpha: &str,
    out: Option<&Bound<'py, PyAny>>,
    chunk_attempts: usize,
    lanes: usize,
    lane_impl: &str,
) -> PyResult<EOut> {
    let call = Call {
        params: params_from(prior, learns, forgets, guesses, slips)?,
        pred: None,
        opts: parse_opts(threads, chunk_attempts, lanes, lane_impl)?,
        predict: false,
        out_mode: parse_out(alpha)?,
        out,
    };
    let c = &ds.get().c;
    if call.params.learns.len() > 1 && c.res.is_empty() {
        return Err(PyValueError::new_err("Dataset was built with num_resources=1"));
    }
    let inp = Input { data: &c.codes, k: c.k, n: c.n, res: &c.res, starts: &c.starts, lengths: &c.lengths };
    let job = Par(job_for(inp, &call)?);
    let (cnt, a) = execute(py, &job, call.out_mode, call.out)?;
    Ok((cnt.trans, cnt.em, cnt.init.to_vec(), cnt.ll, a))
}

/// fit(data, resources, starts, lengths, prior, learns, forgets, guesses, slips, max_iter=100,
///     tol=1e-3, threads=0, lanes=0, lane_impl="plain", chunk_attempts=65536, compat_int_loglike=False)
/// Whole pyBKT EM_fit (fixed={}) in one call. Returns a dict with prior, learns, forgets,
/// guesses, slips, As (R*4 [r][to][from]), emissions (K*4 [k][state][obs]), pi_0, log_likelihoods.
#[pyfunction]
#[pyo3(signature = (data, resources, starts, lengths, prior, learns, forgets, guesses, slips, max_iter=100, tol=1e-3, threads=0, lanes=0, lane_impl="plain", chunk_attempts=65536, compat_int_loglike=false))]
#[allow(clippy::too_many_arguments)]
fn fit<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    resources: &Bound<'py, PyAny>,
    starts: &Bound<'py, PyAny>,
    lengths: &Bound<'py, PyAny>,
    prior: f64,
    learns: &Bound<'py, PyAny>,
    forgets: &Bound<'py, PyAny>,
    guesses: &Bound<'py, PyAny>,
    slips: &Bound<'py, PyAny>,
    max_iter: usize,
    tol: f64,
    threads: usize,
    lanes: usize,
    lane_impl: &str,
    chunk_attempts: usize,
    compat_int_loglike: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let p0 = params_from(prior, learns, forgets, guesses, slips)?;
    let opts = parse_opts(threads, chunk_attempts, lanes, lane_impl)?;
    let c = compact(py, data, resources, i64s(starts)?, i64s(lengths)?, p0.learns.len() > 1)?;
    let inp = Input { data: &c.codes, k: c.k, n: c.n, res: &c.res, starts: &c.starts, lengths: &c.lengths };
    // validate once
    let call = Call { params: p0.clone(), pred: None, opts, predict: false, out_mode: OutMode::None, out: None };
    drop(job_for(Input { data: &c.codes, k: c.k, n: c.n, res: &c.res, starts: &c.starts, lengths: &c.lengths }, &call)?);
    let r = em::fit(&inp, p0, opts, max_iter, tol, compat_int_loglike);
    let d = PyDict::new(py);
    d.set_item("prior", r.params.prior)?;
    d.set_item("learns", r.params.learns)?;
    d.set_item("forgets", r.params.forgets)?;
    d.set_item("guesses", r.params.guesses)?;
    d.set_item("slips", r.params.slips)?;
    if let Some(mo) = r.last {
        d.set_item("As", mo.as_)?;
        d.set_item("emissions", mo.emissions)?;
        d.set_item("pi_0", mo.pi0.to_vec())?;
    }
    d.set_item("log_likelihoods", r.lls)?;
    Ok(d)
}

#[pyfunction]
fn simd_level() -> String {
    lanes::level_name()
}

#[pyfunction]
#[pyo3(signature = (lengths, lanes, chunk_attempts=65536, sort=true))]
fn lane_utilization(lengths: &Bound<'_, PyAny>, lanes: usize, chunk_attempts: usize, sort: bool) -> PyResult<f64> {
    Ok(lanes::utilization(&i64s(lengths)?, lanes.max(1), chunk_attempts.max(1), sort))
}

#[pymodule]
fn bkt_lean(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(e_step, m)?)?;
    m.add_function(wrap_pyfunction!(e_step_ds, m)?)?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(simd_level, m)?)?;
    m.add_function(wrap_pyfunction!(lane_utilization, m)?)?;
    m.add_class::<Dataset>()?;
    m.add("HAS_SIMD", cfg!(feature = "simd"))?;
    Ok(())
}
