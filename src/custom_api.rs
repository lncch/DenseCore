//! Callback-free native boundary for Python-driven custom backward.
use crate::{
    NativeTensor,
    autograd::{BackwardSession, BackwardStep, TensorHandle},
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};
#[pyclass(name = "_TensorIdentity", frozen)]
pub(crate) struct NativeTensorIdentity {
    inner: crate::autograd::TensorIdentity,
}
#[pymethods]
impl NativeTensorIdentity {
    #[getter]
    fn id(&self) -> usize {
        self.inner.id()
    }
    #[getter]
    fn alive(&self) -> bool {
        self.inner.alive()
    }
    fn set_hooks(&self, enabled: bool) {
        self.inner.set_hooks(enabled);
    }
}
pub(crate) fn identity(value: &TensorHandle) -> NativeTensorIdentity {
    NativeTensorIdentity {
        inner: value.identity(),
    }
}
type CallbackEvent = (String, u64, NativeTensor, Vec<NativeTensor>);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[pyclass(name = "_BackwardSession", frozen, unsendable)]
pub(crate) struct NativeBackwardSession {
    inner: Mutex<BackwardSession>,
}
#[pymethods]
impl NativeBackwardSession {
    #[new]
    #[pyo3(signature=(output, gradient=None, retain_graph=false, hook_ids=Vec::new()))]
    fn new(
        output: &NativeTensor,
        gradient: Option<&NativeTensor>,
        retain_graph: bool,
        hook_ids: Vec<usize>,
    ) -> PyResult<Self> {
        if gradient.is_some_and(|g| g.inner.requires_grad()) {
            return Err(PyRuntimeError::new_err("gradient seed must be detached"));
        }
        Ok(Self {
            inner: Mutex::new(
                BackwardSession::with_hooks(
                    &output.inner,
                    gradient.map(|g| g.inner.value()),
                    retain_graph,
                    hook_ids.into_iter().collect(),
                )
                .map_err(PyRuntimeError::new_err)?,
            ),
        })
    }
    fn advance(&self) -> PyResult<Option<CallbackEvent>> {
        let mut run = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("backward session lock poisoned"))?;
        match run.advance().map_err(PyRuntimeError::new_err)? {
            BackwardStep::Done => Ok(None),
            BackwardStep::Hook { node_id, gradient } => Ok(Some((
                "hook".into(),
                node_id as u64,
                NativeTensor {
                    inner: TensorHandle::new(gradient, false),
                },
                Vec::new(),
            ))),
            BackwardStep::Custom {
                callback_id,
                gradient,
            } => {
                let saved = run
                    .saved()
                    .map_err(PyRuntimeError::new_err)?
                    .into_iter()
                    .map(|inner| NativeTensor { inner })
                    .collect();
                Ok(Some((
                    "custom".into(),
                    callback_id,
                    NativeTensor {
                        inner: TensorHandle::new(gradient, false),
                    },
                    saved,
                )))
            }
        }
    }
    fn resume(&self, py: Python<'_>, gradients: Vec<Option<Py<NativeTensor>>>) -> PyResult<()> {
        let values = gradients.iter().map(|g| {
            g.as_ref().map(|g| {
                let g = g.borrow(py);
                if g.inner.requires_grad() { return Err(PyRuntimeError::new_err("custom backward gradients must be detached; higher-order AD is unsupported")); }
                Ok(g.inner.value().clone())
            }).transpose()
        }).collect::<PyResult<Vec<_>>>()?;
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("backward session lock poisoned"))?
            .resume(values)
            .map_err(PyRuntimeError::new_err)
    }
    fn check_hook(&self, gradient: &NativeTensor) -> PyResult<()> {
        if gradient.inner.requires_grad() {
            return Err(PyRuntimeError::new_err("hook gradient must be detached"));
        }
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("backward session lock poisoned"))?
            .check_hook(gradient.inner.value())
            .map_err(PyRuntimeError::new_err)
    }
    fn resume_hook(&self, gradient: &NativeTensor) -> PyResult<()> {
        if gradient.inner.requires_grad() {
            return Err(PyRuntimeError::new_err("hook gradient must be detached"));
        }
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("backward session lock poisoned"))?
            .resume_hook(gradient.inner.value().clone())
            .map_err(PyRuntimeError::new_err)
    }
    fn abort(&self) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("backward session lock poisoned"))?
            .abort()
            .map_err(PyRuntimeError::new_err)
    }
}

#[pyfunction(name = "_custom_result")]
pub(crate) fn custom_result(
    py: Python<'_>,
    value: &NativeTensor,
    inputs: Vec<Py<NativeTensor>>,
    saved: Vec<Py<NativeTensor>>,
) -> PyResult<(NativeTensor, u64)> {
    let id = loop {
        let current = NEXT_ID.load(Ordering::Relaxed);
        let next = current
            .checked_add(1)
            .ok_or_else(|| PyRuntimeError::new_err("custom autograd identifier overflow"))?;
        if NEXT_ID
            .compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            break current;
        }
    };
    let inputs = crate::handles(py, &inputs);
    let saved = crate::handles(py, &saved);
    let inner = TensorHandle::custom(value.inner.value().clone(), inputs, &saved, id)
        .map_err(PyRuntimeError::new_err)?;
    Ok((NativeTensor { inner }, id))
}
