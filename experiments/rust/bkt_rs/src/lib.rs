#![forbid(unsafe_code)]
//! bkt_rs: Rust prototype of pyBKT's E-step (see REPORT.md). No `unsafe` in this
//! crate; pyo3, numpy, rayon and fearless_simd use `unsafe` internally.

mod kernel;
mod lanes;

use kernel::{Input, Model, Obs, Res};
use lanes::LaneImpl;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

struct Args<'a> {
    starts: &'a [i64],
    lengths: &'a [i64],
    prior: f64,
    learns: &'a [f64],
    forgets: &'a [f64],
    guesses: &'a [f64],
    slips: &'a [f64],
    pred_learns: &'a [f64],
    pred_forgets: &'a [f64],
    threads: usize,
    chunk: usize,
}

fn slice1<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

#[derive(Clone, Copy)]
enum Mode {
    EStep { write_alpha: bool, lanes: usize, imp: LaneImpl },
    Predict,
}

fn run_typed<'py, D: Obs + numpy::Element, R: Res + numpy::Element>(
    py: Python<'py>,
    data: &PyReadonlyArray2<'py, D>,
    res: &PyReadonlyArray1<'py, R>,
    a: &Args,
    mode: Mode,
) -> PyResult<Py<PyAny>> {
    let shape = data.shape();
    let (k, n) = (shape[0], shape[1]);
    let dslice = data.as_slice().map_err(|_| PyValueError::new_err("data must be C-contiguous"))?;
    let rslice = slice1(res, "resources")?;
    let inp = Input { data: dslice, k, n, res: rslice, starts: a.starts, lengths: a.lengths };
    let nr = a.learns.len();
    if a.forgets.len() != nr || a.guesses.len() != k || a.slips.len() != k {
        return Err(PyValueError::new_err("parameter lengths do not match R / K"));
    }
    inp.validate(nr).map_err(PyValueError::new_err)?;
    let model = Model::new(a.prior, a.learns, a.forgets, a.guesses, a.slips, a.pred_learns, a.pred_forgets);
    let (threads, chunk) = (a.threads, a.chunk.max(1));
    // Outputs are allocated by numpy (PyArray_Zeros): calloc + numpy's
    // madvise(MADV_HUGEPAGE) for large buffers, so first-touch page faults are
    // ~512x fewer than for a Rust Vec (measured: ~5 ns/attempt on synth5m).
    // The kernel writes through PyReadwriteArray::as_slice_mut (safe API).
    match mode {
        Mode::Predict => {
            let out = PyArray2::<f64>::zeros(py, [2, n], false);
            {
                let mut rw = out.readwrite();
                let sl = rw.as_slice_mut().map_err(|e| PyValueError::new_err(e.to_string()))?;
                py.detach(|| {
                    let (o0, o1) = sl.split_at_mut(n);
                    kernel::predict(&inp, &model, threads, chunk, (o0, o1))
                });
            }
            Ok(out.into_any().unbind())
        }
        Mode::EStep { write_alpha, lanes, imp } => {
            if lanes > 0 && k != 1 {
                return Err(PyValueError::new_err("lanes>0 requires K == 1"));
            }
            if !matches!(lanes, 0 | 2 | 4 | 8) {
                return Err(PyValueError::new_err("lanes must be 0, 2, 4 or 8"));
            }
            let alpha = if write_alpha { Some(PyArray2::<f64>::zeros(py, [2, n], false)) } else { None };
            let mut rw = alpha.as_ref().map(|a| a.readwrite());
            let sl = match rw.as_mut() {
                Some(r) => Some(r.as_slice_mut().map_err(|e| PyValueError::new_err(e.to_string()))?),
                None => None,
            };
            let c = py.detach(|| {
                let out = sl.map(|s| s.split_at_mut(n));
                if lanes > 0 {
                    lanes::e_step_lanes(&inp, &model, lanes, imp, lanes::level(), threads, chunk, out)
                } else {
                    kernel::e_step(&inp, &model, threads, chunk, out)
                }
            });
            let trans = PyArray1::from_vec(py, c.trans).reshape([nr, 2, 2])?;
            let em = PyArray1::from_vec(py, c.em).reshape([k, 2, 2])?;
            let init = PyArray1::from_vec(py, c.init.to_vec()).reshape([2, 1])?;
            drop(rw);
            let alpha_obj: Py<PyAny> = match alpha {
                Some(a) => a.into_any().unbind(),
                None => py.None(),
            };
            Ok((trans, em, init, c.ll, alpha_obj).into_pyobject(py)?.into_any().unbind())
        }
    }
}

macro_rules! dispatch_res {
    ($py:expr, $data:expr, $res:expr, $args:expr, $mode:expr; $($rt:ty),*) => {{
        $(
            if let Ok(r) = $res.extract::<PyReadonlyArray1<$rt>>() {
                return run_typed($py, &$data, &r, $args, $mode);
            }
        )*
        return Err(PyTypeError::new_err("resources must be a 1-D C-contiguous int64/int32/uint16 array"));
    }};
}

macro_rules! dispatch_types {
    ($py:expr, $data:expr, $res:expr, $args:expr, $mode:expr; $($dt:ty),*) => {{
        $(
            if let Ok(d) = $data.extract::<PyReadonlyArray2<$dt>>() {
                dispatch_res!($py, d, $res, $args, $mode; i64, i32, u16);
            }
        )*
        return Err(PyTypeError::new_err("data must be a 2-D C-contiguous int8/int32/int64 array"));
    }};
}

fn go<'py>(py: Python<'py>, data: &Bound<'py, PyAny>, res: &Bound<'py, PyAny>, args: &Args, mode: Mode) -> PyResult<Py<PyAny>> {
    dispatch_types!(py, data, res, args, mode; i8, i32, i64)
}

/// e_step(data, resources, starts, lengths, prior, learns, forgets, guesses, slips,
///        threads=0, write_alpha=True, chunk_attempts=65536, lanes=0, lane_impl="fearless")
/// -> (trans (R,2,2) [r][from][to], emission (K,2,2) [k][obs][state], init (2,1),
///     loglike (float, untruncated), alpha (2,N) row-major or None)
/// threads=0: serial, same accumulation order as the C++ serial path (bit-identical).
/// threads>=1: deterministic chunked reduction on a rayon pool of that size.
/// lanes in {2,4,8}: SIMD-across-students variant (K == 1 only), always chunked and
/// deterministic; lane_impl = "fearless" | "autovec" | "autovec_dispatch".
#[pyfunction]
#[pyo3(signature = (data, resources, starts, lengths, prior, learns, forgets, guesses, slips, threads=0, write_alpha=true, chunk_attempts=65536, lanes=0, lane_impl="fearless"))]
#[allow(clippy::too_many_arguments)]
fn e_step<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    resources: &Bound<'py, PyAny>,
    starts: PyReadonlyArray1<'py, i64>,
    lengths: PyReadonlyArray1<'py, i64>,
    prior: f64,
    learns: PyReadonlyArray1<'py, f64>,
    forgets: PyReadonlyArray1<'py, f64>,
    guesses: PyReadonlyArray1<'py, f64>,
    slips: PyReadonlyArray1<'py, f64>,
    threads: usize,
    write_alpha: bool,
    chunk_attempts: usize,
    lanes: usize,
    lane_impl: &str,
) -> PyResult<Py<PyAny>> {
    let imp = match lane_impl {
        "fearless" => LaneImpl::Fearless,
        "autovec" => LaneImpl::Autovec,
        "autovec_dispatch" => LaneImpl::AutovecDispatch,
        _ => return Err(PyValueError::new_err("lane_impl must be fearless, autovec or autovec_dispatch")),
    };
    let args = Args {
        starts: slice1(&starts, "starts")?,
        lengths: slice1(&lengths, "lengths")?,
        prior,
        learns: slice1(&learns, "learns")?,
        forgets: slice1(&forgets, "forgets")?,
        guesses: slice1(&guesses, "guesses")?,
        slips: slice1(&slips, "slips")?,
        pred_learns: slice1(&learns, "learns")?,
        pred_forgets: slice1(&forgets, "forgets")?,
        threads,
        chunk: chunk_attempts,
    };
    go(py, data, resources, &args, Mode::EStep { write_alpha, lanes, imp })
}

/// predict(data, resources, starts, lengths, prior, learns, forgets, guesses, slips,
///         threads=0, pred_learns=None, pred_forgets=None, chunk_attempts=65536) -> (2,N)
/// pred[:, start] = initial distribution; pred[:, t+1] = A_pred[res[t]] @ alpha[:, t].
/// pred_learns/pred_forgets default to learns/forgets (C++ uses the model's own
/// learn/forget for the prediction step even when 'fixed' overrides them in the filter).
#[pyfunction]
#[pyo3(signature = (data, resources, starts, lengths, prior, learns, forgets, guesses, slips, threads=0, pred_learns=None, pred_forgets=None, chunk_attempts=65536))]
#[allow(clippy::too_many_arguments)]
fn predict<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    resources: &Bound<'py, PyAny>,
    starts: PyReadonlyArray1<'py, i64>,
    lengths: PyReadonlyArray1<'py, i64>,
    prior: f64,
    learns: PyReadonlyArray1<'py, f64>,
    forgets: PyReadonlyArray1<'py, f64>,
    guesses: PyReadonlyArray1<'py, f64>,
    slips: PyReadonlyArray1<'py, f64>,
    threads: usize,
    pred_learns: Option<PyReadonlyArray1<'py, f64>>,
    pred_forgets: Option<PyReadonlyArray1<'py, f64>>,
    chunk_attempts: usize,
) -> PyResult<Py<PyAny>> {
    let pl = pred_learns.as_ref().unwrap_or(&learns);
    let pf = pred_forgets.as_ref().unwrap_or(&forgets);
    let args = Args {
        starts: slice1(&starts, "starts")?,
        lengths: slice1(&lengths, "lengths")?,
        prior,
        learns: slice1(&learns, "learns")?,
        forgets: slice1(&forgets, "forgets")?,
        guesses: slice1(&guesses, "guesses")?,
        slips: slice1(&slips, "slips")?,
        pred_learns: slice1(pl, "pred_learns")?,
        pred_forgets: slice1(pf, "pred_forgets")?,
        threads,
        chunk: chunk_attempts,
    };
    if args.pred_learns.len() != args.learns.len() || args.pred_forgets.len() != args.learns.len() {
        return Err(PyValueError::new_err("pred_learns/pred_forgets length mismatch"));
    }
    go(py, data, resources, &args, Mode::Predict)
}

/// Name of the SIMD level the lane kernels dispatch to (honours BKT_RS_ISA).
#[pyfunction]
fn simd_level() -> String {
    lanes::level_name(lanes::level())
}

/// Fraction of lane-steps doing useful work for the lane kernels' grouping.
#[pyfunction]
#[pyo3(signature = (lengths, lanes, chunk_attempts=65536, sort=true))]
fn lane_utilization(lengths: PyReadonlyArray1<'_, i64>, lanes: usize, chunk_attempts: usize, sort: bool) -> PyResult<f64> {
    Ok(lanes::utilization(slice1(&lengths, "lengths")?, lanes.max(1), chunk_attempts.max(1), sort))
}

#[pymodule]
fn bkt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(e_step, m)?)?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    m.add_function(wrap_pyfunction!(simd_level, m)?)?;
    m.add_function(wrap_pyfunction!(lane_utilization, m)?)?;
    Ok(())
}
