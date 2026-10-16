//! Private Python binding for supported graph capture and shape specialization.
use crate::{
    NativeTensor, autograd,
    graph::{BackwardPlan, ExecutionWorkspace, Graph, TrainingWorkspace},
    handles,
    tensor::TensorSpec,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyList, PyTuple},
};
use std::sync::{Arc, Mutex};
// GC allocation can synchronously invoke Python callbacks on CPython 3.11/3.12.
// Prevent callbacks while optimizer/engine/executor mutexes are held. Declare
// this guard before lock guards so GC is restored only after every lock drops.
struct GcPause {
    was_enabled: bool,
}
impl GcPause {
    fn new(_py: Python<'_>) -> Self {
        // SAFETY: caller holds the GIL; package supports ordinary GIL Python.
        Self {
            was_enabled: unsafe { pyo3::ffi::PyGC_Disable() != 0 },
        }
    }
}
impl Drop for GcPause {
    fn drop(&mut self) {
        if self.was_enabled {
            // SAFETY: lifetime is confined to a binding call with the GIL held.
            unsafe {
                pyo3::ffi::PyGC_Enable();
            }
        }
    }
}
type OpSpec = (String, usize, Option<usize>, Option<usize>);
struct Specialization {
    specs: Vec<TensorSpec>,
    flags: Vec<bool>,
    plan: Arc<BackwardPlan>,
    inference: Option<ExecutionWorkspace>,
    training: Option<TrainingWorkspace>,
}
struct ExecutorState {
    graph: Graph,
    cache: Vec<Specialization>,
    cache_training_workspace: bool,
}
#[pyclass(name = "_GraphExecutor", frozen)]
pub(crate) struct NativeGraphExecutor {
    state: Mutex<ExecutorState>,
}
#[pymethods]
impl NativeGraphExecutor {
    #[new]
    #[pyo3(signature=(input_count,operations,normalization_eps=None,normalized_features=None,cache_training_workspace=true))]
    fn new(
        input_count: usize,
        operations: Vec<OpSpec>,
        normalization_eps: Option<Vec<Option<f64>>>,
        normalized_features: Option<Vec<Option<usize>>>,
        cache_training_workspace: bool,
    ) -> PyResult<Self> {
        let normalized_features =
            normalized_features.unwrap_or_else(|| vec![None; operations.len()]);
        if normalized_features.len() != operations.len() {
            return Err(PyValueError::new_err("normalized feature count mismatch"));
        }
        let normalization_eps = normalization_eps.unwrap_or_else(|| vec![None; operations.len()]);
        if normalization_eps.len() != operations.len() {
            return Err(PyValueError::new_err("normalization eps count mismatch"));
        }
        if input_count == 0 {
            return Err(PyValueError::new_err("compiled graph requires inputs"));
        }
        let mut graph = Graph::new(input_count);
        for (((kind, input, weight, bias), eps), features) in operations
            .into_iter()
            .zip(normalization_eps)
            .zip(normalized_features)
        {
            if kind != "layer_norm" && (eps.is_some() || features.is_some()) {
                return Err(PyValueError::new_err("eps only supported for layer_norm"));
            }
            match kind.as_str() {
                "layer_norm" => {
                    graph
                        .layer_norm(input, weight, bias, eps.unwrap_or(1e-5) as f32, features)
                        .map_err(PyValueError::new_err)?;
                }
                "linear" => {
                    graph
                        .linear(
                            input,
                            weight.ok_or_else(|| {
                                PyValueError::new_err("linear requires weight slot")
                            })?,
                            bias,
                        )
                        .map_err(PyValueError::new_err)?;
                }
                "add" | "mul" if bias.is_none() => {
                    let right = weight.ok_or_else(|| {
                        PyValueError::new_err("binary operation requires right operand slot")
                    })?;
                    let op = if kind == "add" {
                        crate::ops::BinaryOp::Add
                    } else {
                        crate::ops::BinaryOp::Mul
                    };
                    graph
                        .binary(input, right, op)
                        .map_err(PyValueError::new_err)?;
                }
                "matmul" if bias.is_none() => {
                    graph
                        .matmul(
                            input,
                            weight.ok_or_else(|| {
                                PyValueError::new_err("matmul requires right slot")
                            })?,
                        )
                        .map_err(PyValueError::new_err)?;
                }
                "sum" | "mean" if weight.is_none() && bias.is_none() => {
                    graph
                        .reduce(input, kind == "mean")
                        .map_err(PyValueError::new_err)?;
                }
                "relu" if weight.is_none() && bias.is_none() => {
                    graph.relu(input).map_err(PyValueError::new_err)?;
                }
                "mse" if bias.is_none() => {
                    let target =
                        weight.ok_or_else(|| PyValueError::new_err("mse requires target slot"))?;
                    graph.mse(input, target).map_err(PyValueError::new_err)?;
                }
                "cross_entropy" if bias.is_none() => {
                    let target = weight.ok_or_else(|| {
                        PyValueError::new_err("cross_entropy requires target slot")
                    })?;
                    graph
                        .cross_entropy(input, target)
                        .map_err(PyValueError::new_err)?;
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "unsupported compiled graph operation",
                    ));
                }
            }
        }
        Ok(Self {
            state: Mutex::new(ExecutorState {
                graph,
                cache: Vec::new(),
                cache_training_workspace,
            }),
        })
    }
    #[pyo3(signature = (inputs, parameter_slots, optimizer, publication=None))]
    fn train_step(
        &self,
        py: Python<'_>,
        inputs: Vec<Py<NativeTensor>>,
        parameter_slots: Vec<usize>,
        optimizer: PyRef<'_, crate::NativeOptimizer>,
        publication: Option<(Vec<Py<PyAny>>, Py<PyAny>)>,
    ) -> PyResult<Py<PyTuple>> {
        // The public wrapper passes the original Tensor slot descriptor and exact
        // Parameter objects. Publication runs in Rust while the GIL is held,
        // without Python bytecode between parameter assignments.
        if let Some((parameters, descriptor)) = &publication {
            if parameters.len() != parameter_slots.len()
                || descriptor.bind(py).get_type().name()? != "member_descriptor"
                || descriptor
                    .bind(py)
                    .getattr("__name__")?
                    .extract::<String>()?
                    != "_native"
            {
                return Err(PyValueError::new_err(
                    "invalid training publication descriptor",
                ));
            }
            for (parameter, &slot) in parameters.iter().zip(&parameter_slots) {
                let input = inputs
                    .get(slot)
                    .ok_or_else(|| PyValueError::new_err("invalid parameter slot"))?;
                if !parameter.bind(py).getattr("_native")?.is(input.bind(py)) {
                    return Err(PyValueError::new_err(
                        "publication parameter handle mismatch",
                    ));
                }
                // Validate descriptor ownership and writability before optimizer
                // mutation. Reassigning the identical input is an inert slot write.
                descriptor
                    .bind(py)
                    .call_method1("__set__", (parameter.bind(py), input.bind(py)))?;
            }
        }
        let _gc_pause = GcPause::new(py);
        // Existing optimizer.step() acquires optimizer -> engine.
        // Maintain that order, then acquire the executor lock.
        let mut engine = optimizer
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("optimizer lock poisoned"))?;

        let _gate = autograd::engine_gate().map_err(PyRuntimeError::new_err)?;

        if !autograd::is_grad_enabled() {
            return Err(PyValueError::new_err(
                "compiled training requires gradients enabled",
            ));
        }

        let values = handles(py, &inputs);
        if values.iter().any(|value| value.has_hooks()) {
            return Err(PyValueError::new_err(
                "compiled training does not support tensor hooks",
            ));
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("compiled executor lock poisoned"))?;

        let index = if let Some(index) = state.cache.iter().position(|c| {
            c.specs.len() == values.len()
                && c.flags.len() == values.len()
                && values.iter().enumerate().all(|(j, value)| {
                    c.specs[j].matches(value.value()) && c.flags[j] == value.requires_grad()
                })
        }) {
            index
        } else {
            let specs: Vec<_> = values
                .iter()
                .map(|v| TensorSpec::from_tensor(v.value()))
                .collect();

            let flags: Vec<_> = values.iter().map(|v| v.requires_grad()).collect();

            let plan = Arc::new(
                state
                    .graph
                    .compile_backward_specs(&specs, &flags)
                    .map_err(PyValueError::new_err)?,
            );

            if state.cache.len() == 4 {
                state.cache.remove(0);
            }

            state.cache.push(Specialization {
                specs,
                flags,
                plan,
                inference: None,
                training: None,
            });

            state.cache.len() - 1
        };

        let reuse = state.cache_training_workspace;
        let cached = &mut state.cache[index];
        let mut fresh_workspace;
        let workspace = if reuse {
            if cached.training.is_none() {
                cached.training = Some(
                    cached
                        .plan
                        .create_planned_workspace()
                        .map_err(PyRuntimeError::new_err)?,
                );
            }
            cached.training.as_mut().unwrap()
        } else {
            fresh_workspace = cached
                .plan
                .create_planned_workspace()
                .map_err(PyRuntimeError::new_err)?;
            &mut fresh_workspace
        };

        let (loss, prepared) = cached
            .plan
            .prepare_train_step_unlocked(&values, &parameter_slots, &engine, workspace)
            .map_err(PyRuntimeError::new_err)?;

        // All fallible Python allocations occur before parameter versions and
        // optimizer state commit. Native handles are immutable staged snapshots.
        let updated = prepared
            .parameters()
            .iter()
            .map(|value| {
                Py::new(
                    py,
                    NativeTensor {
                        inner: value.clone(),
                    },
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let loss = Py::new(
            py,
            NativeTensor {
                inner: autograd::TensorHandle::new(loss, false),
            },
        )?;
        let updated_list = PyList::new(py, &updated)?;
        let result =
            PyTuple::new(py, [loss.into_any(), updated_list.into_any().unbind()])?.unbind();
        let publication = if let Some((parameters, descriptor)) = publication {
            let setter = descriptor.bind(py).getattr("__set__")?.unbind();
            let arguments = parameters
                .into_iter()
                .zip(&updated)
                .map(|(parameter, native)| {
                    PyTuple::new(py, [parameter, native.clone_ref(py).into_any()])
                        .map(Bound::unbind)
                })
                .collect::<PyResult<Vec<_>>>()?;
            Some((setter, arguments))
        } else {
            None
        };
        engine.commit_prepared_unlocked(prepared);
        if let Some((setter, arguments)) = publication {
            // Validated built-in member descriptor, preallocated argument tuples,
            // no user callbacks or Python bytecode between writes; GIL held.
            for arguments in arguments {
                setter.bind(py).call1(arguments.bind(py))?;
            }
        }
        Ok(result)
    }

    fn execute(&self, py: Python<'_>, inputs: Vec<Py<NativeTensor>>) -> PyResult<NativeTensor> {
        let inputs = handles(py, &inputs);
        // Global engine lock always precedes executor/workspace locks.
        let _gate = autograd::engine_gate().map_err(PyRuntimeError::new_err)?;
        let enabled = autograd::is_grad_enabled();
        let mut state = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("compiled executor lock poisoned"))?;

        // Compare borrowed tensor metadata on cache hits.
        // Allocate owned metadata only when compiling a new specialization.
        let index = if let Some(i) = state.cache.iter().position(|c| {
            c.specs.len() == inputs.len()
                && c.flags.len() == inputs.len()
                && inputs.iter().enumerate().all(|(j, p)| {
                    c.specs[j].matches(p.value()) && c.flags[j] == (enabled && p.requires_grad())
                })
        }) {
            i
        } else {
            let specs: Vec<_> = inputs
                .iter()
                .map(|p| TensorSpec::from_tensor(p.value()))
                .collect();
            let flags: Vec<_> = inputs
                .iter()
                .map(|p| enabled && p.requires_grad())
                .collect();

            let plan = Arc::new(
                state
                    .graph
                    .compile_backward_specs(&specs, &flags)
                    .map_err(PyValueError::new_err)?,
            );
            if state.cache.len() == 4 {
                state.cache.remove(0);
            }
            state.cache.push(Specialization {
                specs,
                flags,
                plan,
                inference: None,
                training: None,
            });
            state.cache.len() - 1
        };
        let cached = &mut state.cache[index];
        if !enabled {
            if cached.inference.is_none() {
                cached.inference = Some(
                    cached
                        .plan
                        .create_inference_workspace()
                        .map_err(PyValueError::new_err)?,
                );
            }
            let output = cached
                .plan
                .inference_unlocked(
                    &inputs,
                    cached.inference.as_mut().expect("initialized workspace"),
                )
                .map_err(PyValueError::new_err)?;
            return Ok(NativeTensor {
                inner: autograd::TensorHandle::new(output, false),
            });
        }
        // Each live autograd result owns saved activations. Cache eviction cannot
        // invalidate it, and another forward cannot overwrite retained values.
        let mut workspace = cached
            .plan
            .create_planned_workspace()
            .map_err(PyValueError::new_err)?;
        let output = cached
            .plan
            .forward_unlocked(&inputs, &mut workspace)
            .map_err(PyValueError::new_err)?;
        Ok(NativeTensor {
            inner: autograd::TensorHandle::compiled(output, cached.plan.clone(), workspace, inputs),
        })
    }
}
