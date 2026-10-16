//! Mean-reduction loss schemas and CPU float32 forward registrations.
use crate::tensor::{DType, Device, TensorData, TensorLayout};

pub struct MseSpec {
    pub elements: usize,
}
impl MseSpec {
    pub fn new(input: &[usize], target: &[usize]) -> Result<Self, String> {
        if input != target {
            return Err("MSE input and target shapes must match".into());
        }
        let elements = TensorLayout::count(input)?;
        if elements == 0 {
            return Err("MSE requires at least one element".into());
        }
        Ok(Self { elements })
    }
}
#[derive(Clone, Copy)]
pub enum CrossEntropyMode {
    EagerIndices,
    CompiledTensor,
}
pub struct CrossEntropySpec {
    pub batch: usize,
    pub classes: usize,
}
impl CrossEntropySpec {
    pub fn new(logits: &[usize], labels: &[usize], mode: CrossEntropyMode) -> Result<Self, String> {
        if logits.len() != 2 || logits[1] == 0 {
            return Err("cross_entropy requires (batch, classes) logits with classes > 0".into());
        }
        TensorLayout::count(logits)?;
        if labels != [logits[0]] {
            return Err("cross_entropy labels must match batch with rank1 shape".into());
        }
        if matches!(mode, CrossEntropyMode::CompiledTensor) && logits[0] == 0 {
            return Err("compiled cross_entropy requires a nonempty batch".into());
        }
        Ok(Self {
            batch: logits[0],
            classes: logits[1],
        })
    }
    pub fn validate_indices(&self, labels: &[usize]) -> Result<(), String> {
        // usize::MAX marks a row PyTorch's ignore_index skips.
        if labels.len() != self.batch
            || labels.iter().any(|&v| v >= self.classes && v != usize::MAX)
        {
            return Err("cross_entropy labels must match batch and be valid class indices".into());
        }
        Ok(())
    }
}
pub fn decode_class_index(value: f32, classes: usize) -> Result<usize, String> {
    if !value.is_finite() || value < 0. || value.fract() != 0. || (value as f64) >= classes as f64 {
        return Err("cross_entropy labels must be finite integer class indices".into());
    }
    Ok(value as usize)
}
pub fn infer_mse_shape(input: &[usize], target: &[usize]) -> Result<Vec<usize>, String> {
    MseSpec::new(input, target)?;
    Ok(vec![])
}
pub fn infer_cross_entropy_shape(input: &[usize], target: &[usize]) -> Result<Vec<usize>, String> {
    CrossEntropySpec::new(input, target, CrossEntropyMode::CompiledTensor)?;
    Ok(vec![])
}

type MseKernel = fn(&TensorData, &TensorData) -> Result<TensorData, String>;
type CrossEntropyKernel = fn(&TensorData, &[usize]) -> Result<TensorData, String>;
struct LossRegistration<K> {
    device: Device,
    dtype: DType,
    forward: K,
}
static MSE_KERNELS: &[LossRegistration<MseKernel>] = &[LossRegistration {
    device: Device::Cpu,
    dtype: DType::Float32,
    forward: crate::loss::mse_loss_cpu,
}];
static CROSS_ENTROPY_KERNELS: &[LossRegistration<CrossEntropyKernel>] = &[LossRegistration {
    device: Device::Cpu,
    dtype: DType::Float32,
    forward: crate::loss::cross_entropy_cpu,
}];
pub fn mse_loss(input: &TensorData, target: &TensorData) -> Result<TensorData, String> {
    let kernel = MSE_KERNELS
        .iter()
        .find(|r| {
            r.device == input.device()
                && r.dtype == input.dtype()
                && r.device == target.device()
                && r.dtype == target.dtype()
        })
        .ok_or("no registered MSE kernel for operand device/dtype")?;
    (kernel.forward)(input, target)
}
pub fn cross_entropy(input: &TensorData, labels: &[usize]) -> Result<TensorData, String> {
    let kernel = CROSS_ENTROPY_KERNELS
        .iter()
        .find(|r| r.device == input.device() && r.dtype == input.dtype())
        .ok_or("no registered cross_entropy kernel for device/dtype")?;
    (kernel.forward)(input, labels)
}
