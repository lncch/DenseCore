//! Python handle for PyTorch's CPU generator (`torch.Generator` semantics).
use crate::torch_random::{DEFAULT_SEED, Generator};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Mutex;

#[pyclass(name = "_Generator", frozen)]
pub struct NativeGenerator {
    inner: Mutex<Generator>,
}

fn bytes<'py, T: Copy, const W: usize>(
    py: Python<'py>,
    values: &[T],
    le: fn(T) -> [u8; W],
) -> Bound<'py, PyBytes> {
    let out: Vec<u8> = values.iter().flat_map(|&v| le(v)).collect();
    PyBytes::new(py, &out)
}

impl NativeGenerator {
    fn with<R>(&self, f: impl FnOnce(&mut Generator) -> R) -> PyResult<R> {
        Ok(f(&mut *self.inner.lock().map_err(|_| {
            PyRuntimeError::new_err("generator lock poisoned")
        })?))
    }
}

#[pymethods]
impl NativeGenerator {
    #[new]
    #[pyo3(signature = (seed = DEFAULT_SEED))]
    fn new(seed: u64) -> Self {
        NativeGenerator {
            inner: Mutex::new(Generator::new(seed)),
        }
    }
    fn manual_seed(&self, seed: u64) -> PyResult<()> {
        self.with(|g| g.manual_seed(seed))
    }
    fn initial_seed(&self) -> PyResult<u64> {
        self.with(|g| g.initial_seed())
    }
    fn random(&self) -> PyResult<u32> {
        self.with(|g| g.random())
    }
    fn random64(&self) -> PyResult<u64> {
        self.with(|g| g.random64())
    }
    /// float32 (or float64 with double=True) little-endian bytes.
    #[pyo3(signature = (n, low, high, double = false))]
    fn uniform<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        low: f64,
        high: f64,
        double: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if low.partial_cmp(&high).is_none_or(|o| o.is_gt()) {
            return Err(PyValueError::new_err("uniform_ expects from <= to"));
        }
        Ok(if double {
            bytes(
                py,
                &self.with(|g| g.uniform_double(n, low, high))?,
                f64::to_le_bytes,
            )
        } else {
            bytes(
                py,
                &self.with(|g| g.uniform(n, low, high))?,
                f32::to_le_bytes,
            )
        })
    }
    #[pyo3(signature = (n, mean, std, double = false))]
    fn normal<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        mean: f64,
        std: f64,
        double: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if std.partial_cmp(&0.0).is_none_or(|o| o.is_lt()) {
            return Err(PyValueError::new_err("normal expects std >= 0.0"));
        }
        Ok(if double {
            bytes(
                py,
                &self.with(|g| g.normal_double(n, mean, std))?,
                f64::to_le_bytes,
            )
        } else {
            bytes(
                py,
                &self.with(|g| g.normal(n, mean, std))?,
                f32::to_le_bytes,
            )
        })
    }
    /// int64 little-endian bytes in [low, high).
    fn randint<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        low: i64,
        high: i64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if low >= high {
            return Err(PyValueError::new_err(
                "random_ expects 'from' to be less than 'to'",
            ));
        }
        Ok(bytes(
            py,
            &self.with(|g| g.randint(n, low, high))?,
            i64::to_le_bytes,
        ))
    }
    fn randperm<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyBytes>> {
        Ok(bytes(py, &self.with(|g| g.randperm(n))?, i64::to_le_bytes))
    }
    /// float32 0/1 little-endian bytes.
    fn bernoulli<'py>(&self, py: Python<'py>, n: usize, p: f64) -> PyResult<Bound<'py, PyBytes>> {
        if !(0.0..=1.0).contains(&p) {
            return Err(PyValueError::new_err("bernoulli expects 0 <= p <= 1"));
        }
        Ok(bytes(
            py,
            &self.with(|g| g.bernoulli(n, p))?,
            f32::to_le_bytes,
        ))
    }
    /// float64 bytes from one of the double-precision distributions.
    fn sample<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        n: usize,
        a: f64,
        b: f64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let values = self
            .with(|g| g.sample(kind, n, a, b))?
            .map_err(PyValueError::new_err)?;
        Ok(bytes(py, &values, f64::to_le_bytes))
    }
    fn uniform_doubles<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyBytes>> {
        Ok(bytes(
            py,
            &self.with(|g| g.uniform_doubles(n))?,
            f64::to_le_bytes,
        ))
    }
    fn get_state(&self) -> PyResult<Vec<u64>> {
        self.with(|g| g.to_words())
    }
    fn set_state(&self, words: Vec<u64>) -> PyResult<()> {
        let fresh = Generator::from_words(&words).map_err(PyValueError::new_err)?;
        self.with(|g| *g = fresh)
    }
}
