//! Checked reduction semantics and registered CPU float32 kernels.
use crate::tensor::{DType, Device, TensorData, TensorLayout};

pub struct ReductionSpec {
    pub output_shape: Vec<usize>,
    selected: Vec<bool>,
    input_shape: Vec<usize>,
}
impl ReductionSpec {
    pub fn new(shape: &[usize], axes: Option<&[usize]>, keepdims: bool) -> Result<Self, String> {
        TensorLayout::count(shape)?;
        let mut selected = vec![axes.is_none(); shape.len()];
        if let Some(axes) = axes {
            for &axis in axes {
                if axis >= shape.len() || selected[axis] {
                    return Err("invalid or duplicate reduction axis".into());
                }
                selected[axis] = true;
            }
        }
        let mut output_shape = Vec::new();
        for (i, &dim) in shape.iter().enumerate() {
            if !selected[i] {
                output_shape.push(dim);
            } else if keepdims {
                output_shape.push(1);
            }
        }
        TensorLayout::count(&output_shape)?;
        Ok(Self {
            output_shape,
            selected,
            input_shape: shape.to_vec(),
        })
    }
    pub fn divisor(&self) -> Result<usize, String> {
        if self
            .input_shape
            .iter()
            .zip(&self.selected)
            .any(|(&d, &s)| s && d == 0)
        {
            return Ok(0);
        }
        self.input_shape
            .iter()
            .zip(&self.selected)
            .try_fold(
                1usize,
                |n, (&d, &s)| {
                    if s { n.checked_mul(d) } else { Some(n) }
                },
            )
            .ok_or_else(|| "reduction count overflow".into())
    }
}

type AxisResult = Result<(TensorData, Vec<usize>), String>;
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReductionKind {
    Sum,
    Mean,
}
struct ReductionRegistration {
    kind: ReductionKind,
    device: Device,
    dtype: DType,
    whole: fn(&[f32]) -> f32,
    axes: fn(&TensorData, &ReductionSpec) -> AxisResult,
}
static REDUCTIONS: &[ReductionRegistration] = &[
    ReductionRegistration {
        kind: ReductionKind::Sum,
        device: Device::Cpu,
        dtype: DType::Float32,
        whole: sum_cpu,
        axes: |a, spec| axes_cpu(a, spec, false),
    },
    ReductionRegistration {
        kind: ReductionKind::Mean,
        device: Device::Cpu,
        dtype: DType::Float32,
        whole: mean_cpu,
        axes: |a, spec| axes_cpu(a, spec, true),
    },
];
fn registration(a: &TensorData, mean: bool) -> Result<&'static ReductionRegistration, String> {
    let kind = if mean {
        ReductionKind::Mean
    } else {
        ReductionKind::Sum
    };
    REDUCTIONS
        .iter()
        .find(|r| r.kind == kind && r.device == a.device() && r.dtype == a.dtype())
        .ok_or_else(|| "no registered reduction kernel for device/dtype".into())
}
pub fn infer_reduction_shape(
    shape: &[usize],
    axes: Option<&[usize]>,
    keepdims: bool,
) -> Result<Vec<usize>, String> {
    Ok(ReductionSpec::new(shape, axes, keepdims)?.output_shape)
}
pub fn reduce_whole(a: &TensorData, mean: bool) -> Result<TensorData, String> {
    let kernel = registration(a, mean)?;
    let shape = infer_reduction_shape(a.shape(), None, false)?;
    TensorData::new(vec![(kernel.whole)(a.values())], shape)
}
pub fn reduce_axes(a: &TensorData, axes: &[usize], keepdims: bool, mean: bool) -> AxisResult {
    let kernel = registration(a, mean)?;
    let spec = ReductionSpec::new(a.shape(), Some(axes), keepdims)?;
    (kernel.axes)(a, &spec)
}
fn sum_cpu(values: &[f32]) -> f32 {
    crate::torch_sum::sum(values, &[values.len()], &[true])[0]
}
fn mean_cpu(values: &[f32]) -> f32 {
    sum_cpu(values) / values.len() as f32
}
/// Specialized compiled CPU instructions reuse the registered kernel arithmetic.
pub fn reduce_cpu_values(values: &[f32], mean: bool) -> f32 {
    if mean {
        mean_cpu(values)
    } else {
        sum_cpu(values)
    }
}
fn axes_cpu(a: &TensorData, spec: &ReductionSpec, mean: bool) -> AxisResult {
    let shape = a.shape();
    let rank = shape.len();
    let out_len = TensorLayout::count(&spec.output_shape)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(out_len)
        .map_err(|_| "reduction output allocation failed")?;
    values.resize(out_len, 0.0f32);
    let mut mapping = Vec::new();
    mapping
        .try_reserve_exact(a.numel())
        .map_err(|_| "reduction index allocation failed")?;
    let mut output_strides = vec![0usize; rank];
    let mut stride = 1usize;
    for axis in (0..rank).rev() {
        if !spec.selected[axis] {
            output_strides[axis] = stride;
            stride = if out_len == 0 {
                stride.saturating_mul(shape[axis])
            } else {
                stride
                    .checked_mul(shape[axis])
                    .ok_or("shape product overflow")?
            };
        }
    }
    for flat in 0..a.numel() {
        let mut remainder = flat;
        let mut out_index = 0usize;
        for axis in (0..rank).rev() {
            let coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            out_index += coordinate * output_strides[axis];
        }
        mapping.push(out_index);
    }
    if out_len > 0 {
        values = crate::torch_sum::sum(a.values(), shape, &spec.selected);
    }
    if mean && !values.is_empty() {
        let denominator = spec.divisor()?;
        for v in &mut values {
            *v /= denominator as f32;
        }
    }
    Ok((TensorData::new(values, spec.output_shape.clone())?, mapping))
}
