//! Shared dense Linear metadata and CPU float32 registration.
use crate::tensor::{DType, Device, TensorData, TensorLayout};
pub struct LinearSpec {
    pub rows: usize,
    pub inputs: usize,
    pub outputs: usize,
}
impl LinearSpec {
    pub fn new(input: &[usize], weight: &[usize], bias: Option<&[usize]>) -> Result<Self, String> {
        if input.len() != 2 || weight.len() != 2 || input[1] != weight[1] {
            return Err("linear requires compatible rank2 input and weight".into());
        }
        if bias.is_some_and(|b| b != [weight[0]]) {
            return Err("linear bias must have shape (out_features,)".into());
        }
        for shape in [input, weight, &[input[0], weight[0]]] {
            TensorLayout::count(shape)?;
        }
        Ok(Self {
            rows: input[0],
            inputs: input[1],
            outputs: weight[0],
        })
    }
}
type LinearKernel = fn(&TensorData, &TensorData, Option<&TensorData>) -> Result<TensorData, String>;
struct LinearRegistration {
    device: Device,
    dtype: DType,
    forward: LinearKernel,
}
static LINEAR_KERNELS: &[LinearRegistration] = &[LinearRegistration {
    device: Device::Cpu,
    dtype: DType::Float32,
    forward: crate::ops::linear_cpu,
}];
pub fn infer_linear_shape(
    input: &[usize],
    weight: &[usize],
    bias: Option<&[usize]>,
) -> Result<Vec<usize>, String> {
    let spec = LinearSpec::new(input, weight, bias)?;
    Ok(vec![spec.rows, spec.outputs])
}
pub fn linear(
    input: &TensorData,
    weight: &TensorData,
    bias: Option<&TensorData>,
) -> Result<TensorData, String> {
    let registration = LINEAR_KERNELS
        .iter()
        .find(|r| {
            [input, weight]
                .into_iter()
                .chain(bias)
                .all(|t| t.device() == r.device && t.dtype() == r.dtype)
        })
        .ok_or("no registered Linear kernel for operand device/dtype")?;
    (registration.forward)(input, weight, bias)
}
