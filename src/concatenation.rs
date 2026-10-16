//! Checked concatenation of logical row-major CPU values.
use crate::tensor::{DType, Device, TensorData, TensorLayout};

struct CatSpec {
    shape: Vec<usize>,
    count: usize,
}
impl CatSpec {
    fn new(shapes: &[&[usize]], axis: usize) -> Result<Self, String> {
        let first = shapes.first().ok_or("cat requires at least one tensor")?;
        if axis >= first.len() {
            return Err("cat requires a valid axis and non-scalar inputs".into());
        }
        let mut shape = first.to_vec();
        shape[axis] = 0;
        for input in shapes {
            TensorLayout::count(input)?;
            if input.len() != first.len()
                || input
                    .iter()
                    .enumerate()
                    .any(|(i, d)| i != axis && *d != first[i])
            {
                return Err("cat non-axis dimensions must match".into());
            }
            shape[axis] = shape[axis]
                .checked_add(input[axis])
                .ok_or("cat axis size overflow")?;
        }
        let count = TensorLayout::count(&shape)?;
        Ok(Self { shape, count })
    }
}
struct CatRegistration {
    device: Device,
    dtype: DType,
    forward: fn(&[&TensorData], usize) -> Result<TensorData, String>,
}
static CAT: &[CatRegistration] = &[CatRegistration {
    device: Device::Cpu,
    dtype: DType::Float32,
    forward: cpu,
}];
pub fn cat(inputs: &[&TensorData], axis: usize) -> Result<TensorData, String> {
    if inputs.is_empty() {
        return Err("cat requires at least one tensor".into());
    }
    let registration = CAT
        .iter()
        .find(|r| {
            inputs
                .iter()
                .all(|a| a.device() == r.device && a.dtype() == r.dtype)
        })
        .ok_or("no registered cat kernel for device/dtype")?;
    (registration.forward)(inputs, axis)
}
fn cpu(inputs: &[&TensorData], axis: usize) -> Result<TensorData, String> {
    let shapes: Vec<_> = inputs.iter().map(|a| a.shape()).collect();
    let spec = CatSpec::new(&shapes, axis)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(spec.count)
        .map_err(|_| "cat allocation failed")?;
    if spec.count == 0 {
        return TensorData::new(values, spec.shape);
    }
    let inner: usize = spec.shape[axis + 1..].iter().product();
    let outer: usize = spec.shape[..axis].iter().product();
    let sources: Vec<_> = inputs
        .iter()
        .map(|a| (a.values(), a.shape()[axis] * inner))
        .collect();
    for row in 0..outer {
        for (source, width) in &sources {
            values.extend_from_slice(&source[row * width..(row + 1) * width]);
        }
    }
    TensorData::new(values, spec.shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_empty_shapes_and_axis_overflow() {
        let a = TensorData::new(vec![], vec![0, usize::MAX, 2]).unwrap();
        assert_eq!(cat(&[&a], 0).unwrap().shape(), a.shape());
        assert!(CatSpec::new(&[&[0, usize::MAX], &[0, 1]], 1).is_err());
        assert!(CatSpec::new(&[], 0).is_err());
    }
}
