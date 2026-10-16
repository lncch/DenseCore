//! Registration for existing CPU ReLU and softmax-family kernels.
use crate::tensor::{DType, Device, TensorData, TensorLayout};
pub struct SoftmaxSpec {
    pub width: usize,
}
impl SoftmaxSpec {
    pub fn new(shape: &[usize], axis: usize) -> Result<Self, String> {
        TensorLayout::count(shape)?;
        if axis >= shape.len() {
            return Err("softmax axis out of range".into());
        }
        if shape[axis] == 0 {
            return Err("softmax requires a nonempty reduction axis".into());
        }
        Ok(Self { width: shape[axis] })
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ActivationKind {
    Relu,
    Softmax,
    LogSoftmax,
}
type ActivationKernel = fn(&TensorData, usize) -> Result<TensorData, String>;
struct ActivationRegistration {
    kind: ActivationKind,
    device: Device,
    dtype: DType,
    forward: ActivationKernel,
}
static ACTIVATIONS: &[ActivationRegistration] = &[
    ActivationRegistration {
        kind: ActivationKind::Relu,
        device: Device::Cpu,
        dtype: DType::Float32,
        forward: |a, _| crate::ops::relu_cpu(a),
    },
    ActivationRegistration {
        kind: ActivationKind::Softmax,
        device: Device::Cpu,
        dtype: DType::Float32,
        forward: |a, axis| crate::ops::softmax_cpu(a, axis, false),
    },
    ActivationRegistration {
        kind: ActivationKind::LogSoftmax,
        device: Device::Cpu,
        dtype: DType::Float32,
        forward: |a, axis| crate::ops::softmax_cpu(a, axis, true),
    },
];
fn execute(kind: ActivationKind, input: &TensorData, axis: usize) -> Result<TensorData, String> {
    let registration = ACTIVATIONS
        .iter()
        .find(|r| r.kind == kind && r.device == input.device() && r.dtype == input.dtype())
        .ok_or("no registered activation kernel for device/dtype")?;
    (registration.forward)(input, axis)
}
pub fn relu(input: &TensorData) -> Result<TensorData, String> {
    execute(ActivationKind::Relu, input, 0)
}
pub fn softmax(input: &TensorData, axis: usize, logarithmic: bool) -> Result<TensorData, String> {
    execute(
        if logarithmic {
            ActivationKind::LogSoftmax
        } else {
            ActivationKind::Softmax
        },
        input,
        axis,
    )
}
pub fn infer_relu_shape(shape: &[usize]) -> Result<Vec<usize>, String> {
    TensorLayout::count(shape)?;
    Ok(shape.to_vec())
}
