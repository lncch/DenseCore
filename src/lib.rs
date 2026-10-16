mod autograd;
mod binary_math;
mod boolean_api;
mod cpu_kernels;
mod custom_api;
mod dispatch;
#[allow(dead_code)]
mod graph;
mod graph_api;
mod loss;
mod normalization;
mod ops;
mod optim;
mod random_api;
mod sleef;
mod tensor;
mod torch_attention;
mod torch_blas;
mod torch_random;
mod torch_softmax;
mod torch_sum;
mod typed_api;
use autograd::TensorHandle;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Mutex;
use tensor::TensorData;
type StateSnapshot = (u64, Vec<f32>, Vec<f32>);
#[pyclass(name = "_Tensor", frozen)]
struct NativeTensor {
    inner: TensorHandle,
}
#[pymethods]
impl NativeTensor {
    /// Build from little-endian float32 bytes (NumPy's layout), with no per-element objects.
    #[staticmethod]
    #[pyo3(signature = (data, shape, requires_grad=false))]
    fn from_bytes(data: &[u8], shape: Vec<usize>, requires_grad: bool) -> PyResult<Self> {
        if !data.len().is_multiple_of(4) {
            return Err(PyValueError::new_err(
                "float32 byte length must be a multiple of 4",
            ));
        }
        let values = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        Self::new(values, shape, requires_grad)
    }
    /// Logical row-major values as little-endian float32 bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let values = self.inner.value().values();
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        PyBytes::new(py, &out)
    }
    #[new]
    #[pyo3(signature = (data, shape, requires_grad=false))]
    fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> PyResult<Self> {
        Ok(Self {
            inner: TensorHandle::new(
                TensorData::new(data, shape).map_err(PyValueError::new_err)?,
                requires_grad,
            ),
        })
    }
    fn compare(&self, other: &Self, operation: &str) -> PyResult<boolean_api::NativeBoolTensor> {
        Ok(boolean_api::NativeBoolTensor {
            inner: dispatch::compare(self.inner.value(), other.inner.value(), operation)
                .map_err(PyValueError::new_err)?,
        })
    }
    #[getter]
    fn has_custom(&self) -> bool {
        self.inner.has_custom()
    }
    fn _check_current(&self) -> PyResult<()> {
        self.inner.check_current().map_err(PyRuntimeError::new_err)
    }
    fn _identity(&self) -> custom_api::NativeTensorIdentity {
        custom_api::identity(&self.inner)
    }
    fn _saved_snapshot(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .saved_snapshot()
                .map_err(PyRuntimeError::new_err)?,
        })
    }
    fn add(&self, b: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .binary(&b.inner, ops::BinaryOp::Add)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn binary_math(&self, other: &Self, name: &str) -> PyResult<Self> {
        let op = binary_math::BinaryMath::by_name(name).map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: self
                .inner
                .binary_math(&other.inner, op)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn mul(&self, b: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .binary(&b.inner, ops::BinaryOp::Mul)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn sub(&self, b: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .binary(&b.inner, ops::BinaryOp::Sub)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn div(&self, b: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .binary(&b.inner, ops::BinaryOp::Div)
                .map_err(PyValueError::new_err)?,
        })
    }
    #[staticmethod]
    fn cat(inputs: Vec<PyRef<'_, NativeTensor>>, axis: usize) -> PyResult<Self> {
        let handles = inputs.iter().map(|a| a.inner.clone()).collect();
        Ok(Self {
            inner: TensorHandle::cat(handles, axis).map_err(PyValueError::new_err)?,
        })
    }
    fn mv(&self, v: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.mv(&v.inner).map_err(PyValueError::new_err)?,
        })
    }
    fn dot(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .dot(&other.inner)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn bmm(&self, b: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.bmm(&b.inner).map_err(PyValueError::new_err)?,
        })
    }
    fn matmul(&self, b: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.matmul(&b.inner).map_err(PyValueError::new_err)?,
        })
    }
    fn mse_loss(&self, target: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .mse_loss(&target.inner)
                .map_err(PyValueError::new_err)?,
        })
    }

    /// `positions` are little-endian int64 flat indices into self, one per element of src.
    fn scatter_add(&self, src: &Self, positions: &[u8]) -> PyResult<Self> {
        let positions = positions
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| usize::try_from(i64::from_le_bytes(b)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| PyValueError::new_err("scatter_add positions must be nonnegative"))?;
        Ok(Self {
            inner: self
                .inner
                .scatter_add(&src.inner, positions)
                .map_err(PyValueError::new_err)?,
        })
    }
    /// `mask` holds one byte per element (nonzero = fill), sent as bytes, not Python bools.
    fn masked_fill(&self, mask: &[u8], value: f32) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .masked_fill(mask.iter().map(|&m| m != 0).collect(), value)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn embedding(&self, indices: Vec<usize>, index_shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .embedding(indices, &index_shape)
                .map_err(PyValueError::new_err)?,
        })
    }

    fn cross_entropy(&self, labels: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .cross_entropy(labels)
                .map_err(PyValueError::new_err)?,
        })
    }
    #[pyo3(signature=(weight,bias=None))]
    fn linear(&self, weight: &Self, bias: Option<&Self>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .linear(&weight.inner, bias.map(|b| &b.inner))
                .map_err(PyValueError::new_err)?,
        })
    }
    #[pyo3(signature=(weight=None,bias=None,eps=1e-5))]
    fn layer_norm(&self, weight: Option<&Self>, bias: Option<&Self>, eps: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .layer_norm(weight.map(|w| &w.inner), bias.map(|b| &b.inner), eps as f32)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn slice(&self, specs: Vec<(usize, usize, usize, bool)>) -> PyResult<Self> {
        let specs = specs
            .into_iter()
            .map(|(start, step, len, squeeze)| tensor::SliceSpec {
                start,
                step,
                len,
                squeeze,
            })
            .collect();
        Ok(Self {
            inner: self.inner.slice(specs).map_err(PyValueError::new_err)?,
        })
    }
    fn permute(&self, axes: Vec<usize>) -> PyResult<Self> {
        self.apply(ops::UnaryOp::Permute(axes))
    }
    fn transpose(&self) -> PyResult<Self> {
        self.apply(ops::UnaryOp::Transpose)
    }
    fn unary_math(&self, name: &str) -> PyResult<Self> {
        self.apply(ops::UnaryOp::Math(
            dispatch::unary_by_name(name).map_err(PyValueError::new_err)?,
        ))
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.value().dtype().name()
    }
    #[getter]
    fn device(&self) -> &'static str {
        self.inner.value().device().name()
    }
    fn softmax(&self, axis: usize) -> PyResult<Self> {
        self.apply(ops::UnaryOp::Softmax(axis))
    }
    fn log_softmax(&self, axis: usize) -> PyResult<Self> {
        self.apply(ops::UnaryOp::LogSoftmax(axis))
    }
    fn relu(&self) -> PyResult<Self> {
        self.apply(ops::UnaryOp::Relu)
    }
    fn sum(&self) -> PyResult<Self> {
        self.apply(ops::UnaryOp::Sum)
    }
    fn mean(&self) -> PyResult<Self> {
        self.apply(ops::UnaryOp::Mean)
    }
    fn extreme(&self, axes: Vec<usize>, keepdims: bool, max: bool) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .extreme(axes, keepdims, max)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn reduce_axes(&self, axes: Vec<usize>, keepdims: bool, mean: bool) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .reduce_axes(axes, keepdims, mean)
                .map_err(PyValueError::new_err)?,
        })
    }
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.apply(ops::UnaryOp::Reshape(shape))
    }
    /// A fresh contiguous copy that keeps the gradient link (clone / contiguous).
    fn clone_contiguous(&self) -> PyResult<Self> {
        self.apply(ops::UnaryOp::Clone)
    }
    fn detach(&self) -> Self {
        Self {
            inner: TensorHandle::new(self.inner.value().clone(), false),
        }
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.value().shape().to_vec()
    }
    #[getter]
    fn strides(&self) -> Vec<usize> {
        self.inner.value().strides().to_vec()
    }
    #[getter]
    fn storage_offset(&self) -> usize {
        self.inner.value().storage_offset()
    }
    fn numel(&self) -> usize {
        self.inner.value().numel()
    }
    fn is_contiguous(&self) -> bool {
        self.inner.value().is_contiguous()
    }
    fn shares_storage(&self, other: &Self) -> bool {
        self.inner.value().shares_storage(other.inner.value())
    }
    #[getter]
    fn values(&self) -> Vec<f32> {
        self.inner.value().values().to_vec()
    }
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }
    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }
    #[getter]
    fn grad(&self) -> PyResult<Option<Self>> {
        Ok(self
            .inner
            .grad()
            .map_err(PyRuntimeError::new_err)?
            .map(|v| Self {
                inner: TensorHandle::new(v, false),
            }))
    }
    fn zero_grad(&self) -> PyResult<()> {
        self.inner.zero_grad().map_err(PyRuntimeError::new_err)
    }
    #[pyo3(signature = (grad=None))]
    fn set_grad(&self, grad: Option<&Self>) -> PyResult<()> {
        self.inner
            .set_grad(grad.map(|g| g.inner.value().clone()))
            .map_err(PyRuntimeError::new_err)
    }
    #[pyo3(signature = (gradient=None, *, retain_graph=false))]
    fn backward(&self, gradient: Option<&Self>, retain_graph: bool) -> PyResult<()> {
        if gradient.is_some_and(|g| g.inner.requires_grad()) {
            return Err(PyRuntimeError::new_err("gradient seed must be detached"));
        }
        self.inner
            .backward(gradient.map(|g| g.inner.value()), retain_graph)
            .map_err(PyRuntimeError::new_err)
    }
}
impl NativeTensor {
    fn apply(&self, op: ops::UnaryOp) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.unary(op).map_err(PyValueError::new_err)?,
        })
    }
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeTensor>()?;
    m.add_class::<custom_api::NativeBackwardSession>()?;
    m.add_class::<custom_api::NativeTensorIdentity>()?;
    m.add_function(wrap_pyfunction!(custom_api::custom_result, m)?)?;
    m.add_class::<boolean_api::NativeBoolTensor>()?;
    m.add_class::<typed_api::NativeTypedTensor>()?;
    m.add_class::<graph_api::NativeGraphExecutor>()?;
    m.add_class::<NativeOptimizer>()?;
    m.add_class::<random_api::NativeGenerator>()?;
    m.add_function(wrap_pyfunction!(_replace_leaves, m)?)?;
    m.add_function(wrap_pyfunction!(_pointwise, m)?)?;
    m.add_function(wrap_pyfunction!(_flash_attention, m)?)?;
    m.add_function(wrap_pyfunction!(_norm2, m)?)?;
    m.add_function(wrap_pyfunction!(_flash_attention_backward, m)?)?;
    m.add_function(wrap_pyfunction!(_is_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(_set_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_backend, m)?)
}

#[pyclass(name = "_Optimizer", frozen)]
struct NativeOptimizer {
    inner: Mutex<optim::Optimizer>,
}
fn handles(py: Python<'_>, params: &[Py<NativeTensor>]) -> Vec<TensorHandle> {
    params.iter().map(|p| p.borrow(py).inner.clone()).collect()
}
#[pymethods]
impl NativeOptimizer {
    #[new]
    #[pyo3(signature=(kind,params,lr,momentum=0.0,beta1=0.9,beta2=0.999,eps=1e-8,weight_decay=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        kind: &str,
        params: Vec<Py<NativeTensor>>,
        lr: f64,
        momentum: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> PyResult<Self> {
        let kind = match kind {
            "sgd" => optim::Kind::Sgd,
            "adam" => optim::Kind::Adam,
            "adamw" => optim::Kind::AdamW,
            _ => return Err(PyValueError::new_err("unsupported optimizer")),
        };
        let config = optim::Config {
            kind,
            lr,
            momentum,
            beta1,
            beta2,
            eps,
            weight_decay,
        };
        Ok(Self {
            inner: Mutex::new(
                optim::Optimizer::new(config, &handles(py, &params))
                    .map_err(PyValueError::new_err)?,
            ),
        })
    }
    fn step(&self, py: Python<'_>, params: Vec<Py<NativeTensor>>) -> PyResult<Vec<NativeTensor>> {
        let mut engine = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("optimizer lock poisoned"))?;
        Ok(engine
            .step(&handles(py, &params))
            .map_err(PyRuntimeError::new_err)?
            .into_iter()
            .map(|inner| NativeTensor { inner })
            .collect())
    }
    fn snapshot(&self) -> PyResult<Vec<StateSnapshot>> {
        let engine = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("optimizer lock poisoned"))?;
        Ok(engine
            .snapshot()
            .into_iter()
            .map(|s| (s.step, s.first, s.second))
            .collect())
    }
    fn set_hyper(&self, name: &str, value: f64) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("optimizer lock poisoned"))?
            .set_hyper(name, value)
            .map_err(PyValueError::new_err)
    }
    fn set_lr(&self, lr: f64) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("optimizer lock poisoned"))?
            .set_lr(lr)
            .map_err(PyValueError::new_err)
    }
    fn load(&self, states: Vec<StateSnapshot>) -> PyResult<()> {
        let mut engine = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("optimizer lock poisoned"))?;
        engine
            .load(
                states
                    .into_iter()
                    .map(|(step, first, second)| optim::State {
                        step,
                        first,
                        second,
                    })
                    .collect(),
            )
            .map_err(PyValueError::new_err)
    }
}
#[pyfunction]
fn _replace_leaves(
    py: Python<'_>,
    params: Vec<Py<NativeTensor>>,
    values: Vec<Py<NativeTensor>>,
) -> PyResult<Vec<NativeTensor>> {
    let values = values
        .iter()
        .map(|v| v.borrow(py).inner.value().clone())
        .collect();
    Ok(autograd::replace_batch(&handles(py, &params), values)
        .map_err(PyRuntimeError::new_err)?
        .into_iter()
        .map(|inner| NativeTensor { inner })
        .collect())
}

/// Elementwise values (float32 bytes) with one slope array (float32 bytes) per input.
#[pyfunction]
fn _pointwise(
    py: Python<'_>,
    inputs: Vec<Py<NativeTensor>>,
    value: &[u8],
    shape: Vec<usize>,
    slopes: Vec<Vec<u8>>,
) -> PyResult<NativeTensor> {
    let floats = |bytes: &[u8]| -> PyResult<Vec<f32>> {
        if !bytes.len().is_multiple_of(4) {
            return Err(PyValueError::new_err(
                "float32 byte length must be a multiple of 4",
            ));
        }
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect())
    };
    let value = TensorData::new(floats(value)?, shape).map_err(PyValueError::new_err)?;
    let slopes = slopes
        .iter()
        .map(|s| floats(s))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(NativeTensor {
        inner: TensorHandle::pointwise(&handles(py, &inputs), value, slopes)
            .map_err(PyValueError::new_err)?,
    })
}

fn attention_dims(
    q: &TensorData,
    k: &TensorData,
    v: &TensorData,
) -> PyResult<torch_attention::Dims> {
    let (qs, ks, vs) = (q.shape(), k.shape(), v.shape());
    if qs.len() != 4 || ks.len() != 4 || vs.len() != 4 {
        return Err(PyValueError::new_err(
            "flash attention takes (B, H, L, E) tensors",
        ));
    }
    let ok = ks == vs && ks[0] == qs[0] && ks[3] == qs[3] && ks[1] > 0 && qs[1] % ks[1] == 0;
    if !ok {
        return Err(PyValueError::new_err(
            "flash attention q/k/v shapes do not match",
        ));
    }
    Ok(torch_attention::Dims {
        batch: qs[0],
        heads: qs[1],
        kv_heads: ks[1],
        q_len: qs[2],
        kv_len: ks[2],
        head_dim: qs[3],
    })
}

fn plain(values: Vec<f32>, shape: Vec<usize>) -> PyResult<NativeTensor> {
    Ok(NativeTensor {
        inner: TensorHandle::new(
            TensorData::new(values, shape).map_err(PyValueError::new_err)?,
            false,
        ),
    })
}

/// PyTorch's CPU flash attention forward: (output, logsumexp). No autograd here; the Python
/// side wraps it in a Function.
/// An additive float mask broadcast to (batch, heads, q_len, kv_len), if given.
fn attention_mask(
    py: Python<'_>,
    mask: &Option<Py<NativeTensor>>,
    d: torch_attention::Dims,
) -> PyResult<Option<TensorData>> {
    let Some(mask) = mask else { return Ok(None) };
    let mask = mask.borrow(py).inner.value().clone();
    if mask.shape() != [d.batch, d.heads, d.q_len, d.kv_len] {
        return Err(PyValueError::new_err(
            "flash attention mask must be (B, H, L, S)",
        ));
    }
    Ok(Some(mask))
}

#[pyfunction]
#[pyo3(signature=(q, k, v, causal, scale=None, mask=None))]
fn _flash_attention(
    py: Python<'_>,
    q: Py<NativeTensor>,
    k: Py<NativeTensor>,
    v: Py<NativeTensor>,
    causal: bool,
    scale: Option<f64>,
    mask: Option<Py<NativeTensor>>,
) -> PyResult<(NativeTensor, NativeTensor)> {
    let (q, k, v) = (q.borrow(py), k.borrow(py), v.borrow(py));
    let (q, k, v) = (q.inner.value(), k.inner.value(), v.inner.value());
    let d = attention_dims(q, k, v)?;
    let mask = attention_mask(py, &mask, d)?;
    let (out, lse) = torch_attention::forward(
        q.values(),
        k.values(),
        v.values(),
        mask.as_ref().map(|m| m.values()),
        d,
        causal,
        scale,
    )
    .map_err(PyRuntimeError::new_err)?;
    Ok((
        plain(out, q.shape().to_vec())?,
        plain(lse, vec![d.batch, d.heads, d.q_len])?,
    ))
}

#[pyfunction]
#[pyo3(signature=(grad_out, q, k, v, out, lse, causal, scale=None, mask=None))]
#[allow(clippy::too_many_arguments)]
fn _flash_attention_backward(
    py: Python<'_>,
    grad_out: Py<NativeTensor>,
    q: Py<NativeTensor>,
    k: Py<NativeTensor>,
    v: Py<NativeTensor>,
    out: Py<NativeTensor>,
    lse: Py<NativeTensor>,
    causal: bool,
    scale: Option<f64>,
    mask: Option<Py<NativeTensor>>,
) -> PyResult<(NativeTensor, NativeTensor, NativeTensor)> {
    let b = |t: &Py<NativeTensor>| t.borrow(py).inner.value().clone();
    let (g, q, k, v, o, l) = (b(&grad_out), b(&q), b(&k), b(&v), b(&out), b(&lse));
    let d = attention_dims(&q, &k, &v)?;
    if g.shape() != q.shape() || o.shape() != q.shape() || l.shape() != [d.batch, d.heads, d.q_len]
    {
        return Err(PyValueError::new_err(
            "flash attention backward shapes do not match",
        ));
    }
    let mask = attention_mask(py, &mask, d)?;
    let (gq, gk, gv) = torch_attention::backward(
        g.values(),
        q.values(),
        k.values(),
        v.values(),
        o.values(),
        l.values(),
        mask.as_ref().map(|m| m.values()),
        d,
        causal,
        scale,
    )
    .map_err(PyRuntimeError::new_err)?;
    Ok((
        plain(gq, q.shape().to_vec())?,
        plain(gk, k.shape().to_vec())?,
        plain(gv, v.shape().to_vec())?,
    ))
}

/// PyTorch's float32 `linalg.vector_norm(x, 2)` over all elements.
#[pyfunction]
fn _norm2(py: Python<'_>, t: Py<NativeTensor>) -> f32 {
    torch_sum::norm2(t.borrow(py).inner.value().values())
}

/// Intra-op thread count; like PyTorch's, it decides how reductions are split (and rounded).
#[pyfunction]
fn get_num_threads() -> usize {
    torch_sum::num_threads()
}
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    torch_sum::set_num_threads(n).map_err(PyValueError::new_err)
}
#[pyfunction]
fn _is_grad_enabled() -> bool {
    autograd::is_grad_enabled()
}
#[pyfunction]
fn _set_grad_enabled(enabled: bool) -> bool {
    autograd::set_grad_enabled(enabled)
}

#[pyfunction]
fn cpu_backend() -> &'static str {
    cpu_kernels::backend()
}
