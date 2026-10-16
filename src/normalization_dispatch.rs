//! Shared LayerNorm shape semantics and CPU kernel registration.
use crate::tensor::{DType, Device, TensorData, TensorLayout};

pub struct LayerNormSpec {
    pub rows: usize,
    pub features: usize,
}
impl LayerNormSpec {
    pub fn new(
        input: &[usize],
        weight: Option<&[usize]>,
        bias: Option<&[usize]>,
        eps: f32,
        expected_features: Option<usize>,
    ) -> Result<Self, String> {
        if input.len() != 2 || input[1] == 0 {
            return Err("layer_norm requires rank2 input with positive features".into());
        }
        TensorLayout::count(input)?;
        let features = input[1];
        if expected_features.is_some_and(|n| n != features) {
            return Err("layer_norm normalized feature count mismatch".into());
        }
        if weight.is_some_and(|shape| shape != [features])
            || bias.is_some_and(|shape| shape != [features])
        {
            return Err("layer_norm affine tensors must match feature count".into());
        }
        crate::normalization::validate_epsilon(eps)?;
        Ok(Self {
            rows: input[0],
            features,
        })
    }
}

type LayerNormKernel =
    fn(&TensorData, Option<&TensorData>, Option<&TensorData>, f32) -> Result<TensorData, String>;
struct LayerNormRegistration {
    device: Device,
    dtype: DType,
    forward: LayerNormKernel,
}
static LAYER_NORM_KERNELS: &[LayerNormRegistration] = &[LayerNormRegistration {
    device: Device::Cpu,
    dtype: DType::Float32,
    forward: crate::normalization::forward_cpu,
}];

pub fn infer_layer_norm_shape(
    input: &[usize],
    weight: Option<&[usize]>,
    bias: Option<&[usize]>,
    eps: f32,
    expected_features: Option<usize>,
) -> Result<Vec<usize>, String> {
    LayerNormSpec::new(input, weight, bias, eps, expected_features)?;
    Ok(input.to_vec())
}
pub fn layer_norm(
    input: &TensorData,
    weight: Option<&TensorData>,
    bias: Option<&TensorData>,
    eps: f32,
) -> Result<TensorData, String> {
    let kernel = LAYER_NORM_KERNELS
        .iter()
        .find(|r| {
            r.device == input.device()
                && r.dtype == input.dtype()
                && weight
                    .into_iter()
                    .chain(bias)
                    .all(|t| t.device() == r.device && t.dtype() == r.dtype)
        })
        .ok_or("no registered LayerNorm kernel for operand device/dtype")?;
    (kernel.forward)(input, weight, bias, eps)
}
