//! Nondifferentiable boolean handles, isolated from float32 autograd.
use crate::tensor::{BoolTensorData, SliceSpec};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass(name = "_BoolTensor", frozen)]
pub(crate) struct NativeBoolTensor {
    pub(crate) inner: BoolTensorData,
}
#[pymethods]
impl NativeBoolTensor {
    #[new]
    fn new(data: Vec<bool>, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: BoolTensorData::new(data, shape).map_err(PyValueError::new_err)?,
        })
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().name()
    }
    #[getter]
    fn device(&self) -> &'static str {
        self.inner.device().name()
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }
    #[getter]
    fn strides(&self) -> Vec<usize> {
        self.inner.strides().to_vec()
    }
    #[getter]
    fn storage_offset(&self) -> usize {
        self.inner.storage_offset()
    }
    #[getter]
    fn values(&self) -> Vec<bool> {
        self.inner.values().to_vec()
    }
    #[getter]
    fn requires_grad(&self) -> bool {
        false
    }
    #[getter]
    fn is_leaf(&self) -> bool {
        true
    }
    #[getter]
    fn grad(&self) -> Option<Self> {
        None
    }
    fn numel(&self) -> usize {
        self.inner.numel()
    }
    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }
    fn shares_storage(&self, other: &Self) -> bool {
        self.inner.shares_storage(&other.inner)
    }
    fn detach(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.reshape(shape).map_err(PyValueError::new_err)?,
        })
    }
    fn permute(&self, axes: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.permute(&axes).map_err(PyValueError::new_err)?,
        })
    }
    fn transpose(&self) -> PyResult<Self> {
        if self.inner.shape().len() != 2 {
            return Err(PyValueError::new_err("transpose requires rank 2"));
        }
        self.permute(vec![1, 0])
    }
    fn slice(&self, specs: Vec<(usize, usize, usize, bool)>) -> PyResult<Self> {
        let specs: Vec<_> = specs
            .into_iter()
            .map(|(start, step, len, squeeze)| SliceSpec {
                start,
                step,
                len,
                squeeze,
            })
            .collect();
        Ok(Self {
            inner: self.inner.slice(&specs).map_err(PyValueError::new_err)?,
        })
    }
    fn compare(&self, other: &Self, operation: &str) -> PyResult<Self> {
        Ok(Self {
            inner: crate::dispatch::compare_bool(&self.inner, &other.inner, operation)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn logical(&self, other: &Self, operation: &str) -> PyResult<Self> {
        Ok(Self {
            inner: crate::dispatch::logical_binary(&self.inner, &other.inner, operation)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn logical_not(&self) -> PyResult<Self> {
        let mut data = Vec::new();
        data.try_reserve_exact(self.inner.numel())
            .map_err(|_| PyValueError::new_err("boolean allocation failed"))?;
        data.extend(self.inner.values().iter().map(|v| !v));
        Ok(Self {
            inner: BoolTensorData::new(data, self.inner.shape().to_vec())
                .map_err(PyValueError::new_err)?,
        })
    }
}
