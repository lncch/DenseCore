//! Typed CPU predicates. Boolean results never acquire an autograd node.
use crate::tensor::{
    BoolTensorData, DType, Device, Element, TensorData, TensorLayout, TypedTensorData,
};
struct PredicateRegistration<T> {
    name: &'static str,
    device: Device,
    input_dtype: DType,
    scalar: fn(T, T) -> bool,
}
static COMPARISONS: &[PredicateRegistration<f32>] = &[
    PredicateRegistration {
        name: "eq",
        device: Device::Cpu,
        input_dtype: DType::Float32,
        scalar: |a, b| a == b,
    },
    PredicateRegistration {
        name: "ne",
        device: Device::Cpu,
        input_dtype: DType::Float32,
        scalar: |a, b| a != b,
    },
    PredicateRegistration {
        name: "lt",
        device: Device::Cpu,
        input_dtype: DType::Float32,
        scalar: |a, b| a < b,
    },
    PredicateRegistration {
        name: "le",
        device: Device::Cpu,
        input_dtype: DType::Float32,
        scalar: |a, b| a <= b,
    },
    PredicateRegistration {
        name: "gt",
        device: Device::Cpu,
        input_dtype: DType::Float32,
        scalar: |a, b| a > b,
    },
    PredicateRegistration {
        name: "ge",
        device: Device::Cpu,
        input_dtype: DType::Float32,
        scalar: |a, b| a >= b,
    },
];
static BOOL_COMPARISONS: &[PredicateRegistration<bool>] = &[
    PredicateRegistration {
        name: "eq",
        device: Device::Cpu,
        input_dtype: DType::Bool,
        scalar: |a, b| a == b,
    },
    PredicateRegistration {
        name: "ne",
        device: Device::Cpu,
        input_dtype: DType::Bool,
        scalar: |a, b| a != b,
    },
    PredicateRegistration {
        name: "lt",
        device: Device::Cpu,
        input_dtype: DType::Bool,
        scalar: |a, b| !a && b,
    },
    PredicateRegistration {
        name: "le",
        device: Device::Cpu,
        input_dtype: DType::Bool,
        scalar: |a, b| !a || b,
    },
    PredicateRegistration {
        name: "gt",
        device: Device::Cpu,
        input_dtype: DType::Bool,
        scalar: |a, b| a && !b,
    },
    PredicateRegistration {
        name: "ge",
        device: Device::Cpu,
        input_dtype: DType::Bool,
        scalar: |a, b| a || !b,
    },
];
static LOGICAL: &[PredicateRegistration<bool>] = &[
    PredicateRegistration {
        name: "logical_and",
        device: Device::Cpu,
        input_dtype: DType::Bool,
        scalar: |a, b| a && b,
    },
    PredicateRegistration {
        name: "logical_or",
        device: Device::Cpu,
        input_dtype: DType::Bool,
        scalar: |a, b| a || b,
    },
    PredicateRegistration {
        name: "logical_xor",
        device: Device::Cpu,
        input_dtype: DType::Bool,
        scalar: |a, b| a ^ b,
    },
];
fn execute<T: Element>(
    a: &TypedTensorData<T>,
    b: &TypedTensorData<T>,
    name: &str,
    registrations: &[PredicateRegistration<T>],
) -> Result<BoolTensorData, String> {
    let registration = registrations
        .iter()
        .find(|r| {
            r.name == name
                && r.device == a.device()
                && r.device == b.device()
                && r.input_dtype == a.dtype()
                && r.input_dtype == b.dtype()
        })
        .ok_or("no registered predicate kernel for operation/device/dtype")?;
    let shape = crate::ops::broadcast_shape(a.shape(), b.shape())?;
    let count = TensorLayout::count(&shape)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| "predicate allocation failed")?;
    if count != 0 {
        let plan = crate::ops::broadcast_plan(a.shape(), b.shape())?;
        let (left, right) = (a.values(), b.values());
        output.extend(
            plan.left
                .iter()
                .zip(&plan.right)
                .map(|(&i, &j)| (registration.scalar)(left[i], right[j])),
        );
    }
    BoolTensorData::new(output, shape)
}
pub fn compare(a: &TensorData, b: &TensorData, name: &str) -> Result<BoolTensorData, String> {
    execute(a, b, name, COMPARISONS)
}
pub fn compare_bool(
    a: &BoolTensorData,
    b: &BoolTensorData,
    name: &str,
) -> Result<BoolTensorData, String> {
    execute(a, b, name, BOOL_COMPARISONS)
}
pub fn logical_binary(
    a: &BoolTensorData,
    b: &BoolTensorData,
    name: &str,
) -> Result<BoolTensorData, String> {
    execute(a, b, name, LOGICAL)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn predicates_check_shapes_names_and_real_output_dtype() {
        let a = TensorData::new(vec![0., f32::NAN], vec![2]).unwrap();
        let b = TensorData::new(vec![0.], vec![]).unwrap();
        let out = compare(&a, &b, "ne").unwrap();
        assert_eq!(out.dtype(), DType::Bool);
        assert_eq!(out.values(), &[false, true]);
        assert!(compare(&a, &b, "invalid").is_err());
        let x = BoolTensorData::new(vec![true], vec![]).unwrap();
        assert!(logical_binary(&x, &x, "invalid").is_err());
        assert!(crate::ops::broadcast_shape(&[usize::MAX, 1], &[1, 2]).is_err());
    }
}
