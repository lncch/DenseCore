//! Static registration for CPU operator families.
//! CPU kernels consume logical values; autograd formulas remain in autograd.rs.
#[path = "reduction.rs"]
mod reduction;
pub use reduction::{
    ReductionSpec, infer_reduction_shape, reduce_axes, reduce_cpu_values, reduce_whole,
};

#[path = "normalization_dispatch.rs"]
mod normalization_dispatch;
pub use normalization_dispatch::{LayerNormSpec, infer_layer_norm_shape, layer_norm};

#[path = "loss_dispatch.rs"]
mod loss_dispatch;
pub use loss_dispatch::{
    CrossEntropyMode, CrossEntropySpec, MseSpec, cross_entropy, decode_class_index,
    infer_cross_entropy_shape, infer_mse_shape, mse_loss,
};

#[path = "linear_dispatch.rs"]
mod linear_dispatch;
pub use linear_dispatch::{LinearSpec, infer_linear_shape, linear};

#[path = "activation_dispatch.rs"]
mod activation_dispatch;
pub use activation_dispatch::{SoftmaxSpec, infer_relu_shape, relu, softmax};

#[path = "batched_matmul.rs"]
mod batched_matmul;
pub use batched_matmul::bmm;

#[path = "concatenation.rs"]
mod concatenation;
pub use concatenation::cat;

#[path = "predicate_dispatch.rs"]
mod predicate_dispatch;
pub use predicate_dispatch::{compare, compare_bool, logical_binary};

use crate::tensor::{DType, Device, TensorData};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryMath {
    Sigmoid,
    Tanh,
    Neg,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Gelu,
    GeluTanh,
    Abs,
    Sign,
    Floor,
    Ceil,
    Trunc,
    Round,
    Frac,
    Reciprocal,
    Rsqrt,
    Square,
    Exp2,
    Expm1,
    Log1p,
    Log2,
    Log10,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Deg2Rad,
    Rad2Deg,
    Erf,
    Erfc,
    Logit,
    Sinc,
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Exact GELU as PyTorch computes it (see `crate::sleef::gelu`).
fn gelu_array(xs: &[f32]) -> Vec<f32> {
    crate::sleef::gelu(xs)
}

/// The GELU slope PyTorch's backward multiplies the gradient by (see `crate::sleef::gelu_slope`).
pub(crate) fn gelu_derivative_array(xs: &[f32]) -> Vec<f32> {
    crate::sleef::gelu_slope(xs)
}

/// GELU values and slopes together, for a node that records slopes in the forward pass.
pub(crate) fn gelu_value_and_slope(xs: &[f32]) -> (Vec<f32>, Vec<f32>) {
    (gelu_array(xs), gelu_derivative_array(xs))
}

const GELU_TANH_SCALE: f64 = 0.797_884_560_802_865_4; // sqrt(2 / pi)
const GELU_TANH_CUBIC: f64 = 0.044_715;

/// One element of PyTorch's GELU (a lone element takes the scalar tail path, as in PyTorch).
pub(crate) fn gelu(x: f32) -> f32 {
    gelu_array(&[x])[0]
}

/// One element of PyTorch's GELU slope.
pub(crate) fn gelu_derivative(x: f32) -> f32 {
    gelu_derivative_array(&[x])[0]
}

/// Logistic function in f64; each branch avoids overflow and cancellation.
fn sigmoid64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Tanh-approximated GELU. Uses 0.5 * (1 + tanh(u)) = sigmoid(2u), which keeps
/// relative accuracy in the negative tail where 1 + tanh(u) cancels.
pub(crate) fn gelu_tanh(x: f32) -> f32 {
    let x = f64::from(x);
    let u = GELU_TANH_SCALE * (x + GELU_TANH_CUBIC * x * x * x);
    (x * sigmoid64(2.0 * u)) as f32
}

pub(crate) fn gelu_tanh_derivative(x: f32) -> f32 {
    let x = f64::from(x);
    let u = GELU_TANH_SCALE * (x + GELU_TANH_CUBIC * x * x * x);
    let inner = GELU_TANH_SCALE * (1.0 + 3.0 * GELU_TANH_CUBIC * x * x);
    let (s, rest) = (sigmoid64(2.0 * u), sigmoid64(-2.0 * u));
    (s + 2.0 * x * s * rest * inner) as f32
}

/// PyTorch sign: -1 or 1, and +0 for zeros and NaN.
pub(crate) fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// d/dx erf(x) = 2/sqrt(pi) * exp(-x^2).
pub(crate) fn erf_slope(x: f32) -> f32 {
    let x = f64::from(x);
    (std::f64::consts::FRAC_2_SQRT_PI * (-x * x).exp()) as f32
}

/// Normalized sinc, sin(pi x) / (pi x), with sinc(0) = 1.
fn sinc64(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let p = std::f64::consts::PI * x;
        p.sin() / p
    }
}

pub(crate) fn sinc_slope(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        let p = std::f64::consts::PI * x;
        (p.cos() * p - p.sin()) / (p * x)
    }
}

type ArrayKernel = fn(&[f32]) -> Vec<f32>;

macro_rules! unary_kernel {
    ($op:ident, $name:literal, |$x:ident| $kernel:expr) => {
        UnaryRegistration {
            op: UnaryMath::$op,
            name: $name,
            device: Device::Cpu,
            dtype: DType::Float32,
            scalar_kernel: |$x: f32| $kernel,
            array_kernel: None,
        }
    };
}

/// A kernel ported from SLEEF (crate::sleef): the array form runs 4-lane groups as PyTorch's
/// vectorized loop does; single elements go through one zero-padded group.
macro_rules! sleef_kernel {
    ($op:ident, $name:literal, $array:ident) => {
        UnaryRegistration {
            op: UnaryMath::$op,
            name: $name,
            device: Device::Cpu,
            dtype: DType::Float32,
            scalar_kernel: |x: f32| crate::sleef::scalar(x, crate::sleef::$array),
            array_kernel: Some(crate::sleef::$array),
        }
    };
}

struct UnaryRegistration {
    op: UnaryMath,
    name: &'static str,
    device: Device,
    dtype: DType,
    scalar_kernel: fn(f32) -> f32,
    /// Optional whole-array kernel with the same results; used instead of the scalar map.
    array_kernel: Option<ArrayKernel>,
}
// Schema: one input, one output, unchanged shape/device/dtype, no mutation.
// Additional backends must register real kernels, never implicit CPU fallback.
static UNARY_KERNELS: &[UnaryRegistration] = &[
    UnaryRegistration {
        op: UnaryMath::Sigmoid,
        name: "sigmoid",
        device: Device::Cpu,
        dtype: DType::Float32,
        scalar_kernel: |x: f32| crate::sleef::sigmoid_array(&[x])[0],
        array_kernel: Some(crate::sleef::sigmoid_array),
    },
    sleef_kernel!(Tanh, "tanh", tanh_array),
    UnaryRegistration {
        op: UnaryMath::Neg,
        name: "neg",
        device: Device::Cpu,
        dtype: DType::Float32,
        scalar_kernel: |x| -x,
        array_kernel: None,
    },
    sleef_kernel!(Exp, "exp", exp_array),
    sleef_kernel!(Log, "log", log_array),
    UnaryRegistration {
        op: UnaryMath::Sqrt,
        name: "sqrt",
        device: Device::Cpu,
        dtype: DType::Float32,
        scalar_kernel: f32::sqrt,
        array_kernel: None,
    },
    sleef_kernel!(Sin, "sin", sin_array),
    sleef_kernel!(Cos, "cos", cos_array),
    UnaryRegistration {
        op: UnaryMath::Gelu,
        name: "gelu",
        device: Device::Cpu,
        dtype: DType::Float32,
        scalar_kernel: gelu,
        array_kernel: Some(gelu_array),
    },
    UnaryRegistration {
        op: UnaryMath::GeluTanh,
        name: "gelu_tanh",
        device: Device::Cpu,
        dtype: DType::Float32,
        scalar_kernel: gelu_tanh,
        array_kernel: None,
    },
    unary_kernel!(Abs, "abs", |x| x.abs()),
    unary_kernel!(Sign, "sign", |x| sign(x)),
    unary_kernel!(Floor, "floor", |x| x.floor()),
    unary_kernel!(Ceil, "ceil", |x| x.ceil()),
    unary_kernel!(Trunc, "trunc", |x| x.trunc()),
    unary_kernel!(Round, "round", |x| x.round_ties_even()),
    unary_kernel!(Frac, "frac", |x| x - x.trunc()),
    unary_kernel!(Reciprocal, "reciprocal", |x| 1.0 / x),
    unary_kernel!(Rsqrt, "rsqrt", |x| 1.0 / x.sqrt()),
    unary_kernel!(Square, "square", |x| x * x),
    sleef_kernel!(Exp2, "exp2", exp2_array),
    sleef_kernel!(Expm1, "expm1", expm1_array),
    sleef_kernel!(Log1p, "log1p", log1p_array),
    sleef_kernel!(Log2, "log2", log2_array),
    sleef_kernel!(Log10, "log10", log10_array),
    sleef_kernel!(Tan, "tan", tan_array),
    sleef_kernel!(Asin, "asin", asin_array),
    sleef_kernel!(Acos, "acos", acos_array),
    sleef_kernel!(Atan, "atan", atan_array),
    unary_kernel!(Sinh, "sinh", |x| crate::sleef::libm_sinh(x)),
    unary_kernel!(Cosh, "cosh", |x| crate::sleef::libm_cosh(x)),
    unary_kernel!(Asinh, "asinh", |x| crate::sleef::libm_asinh(x)),
    unary_kernel!(Acosh, "acosh", |x| crate::sleef::libm_acosh(x)),
    sleef_kernel!(Atanh, "atanh", atanh_array),
    unary_kernel!(Deg2Rad, "deg2rad", |x| x * (std::f32::consts::PI / 180.0)),
    unary_kernel!(Rad2Deg, "rad2deg", |x| x * (180.0 / std::f32::consts::PI)),
    sleef_kernel!(Erf, "erf", erf_array),
    sleef_kernel!(Erfc, "erfc", erfc_array),
    unary_kernel!(Logit, "logit", |x| (x / (1.0 - x)).ln()),
    unary_kernel!(Sinc, "sinc", |x| sinc64(f64::from(x)) as f32),
];

pub fn unary_by_name(name: &str) -> Result<UnaryMath, String> {
    UNARY_KERNELS
        .iter()
        .find(|r| r.name == name)
        .map(|r| r.op)
        .ok_or_else(|| format!("unregistered unary operator: {name}"))
}

pub fn unary(op: UnaryMath, input: &TensorData) -> Result<TensorData, String> {
    let kernel = UNARY_KERNELS
        .iter()
        .find(|r| r.op == op && r.device == input.device() && r.dtype == input.dtype())
        .ok_or_else(|| {
            format!(
                "no kernel for {op:?}, {:?}, {:?}",
                input.device(),
                input.dtype()
            )
        })?;
    let values = match kernel.array_kernel {
        Some(array) => array(input.values()),
        None => input
            .values()
            .iter()
            .map(|&x| (kernel.scalar_kernel)(x))
            .collect(),
    };
    TensorData::new(values, input.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gelu_value_and_slope_match_the_separate_kernels() {
        // Repointed 2026-10-16 (roadmap: PyTorch's bits are the spec; approved by the owner):
        // this used to require the array kernels within 1 ulp of a float64 GELU. The
        // PyTorch bit-for-bit checks live in tests/test_gelu.py and tests/test_unary_exact.py.
        let mut xs: Vec<f32> = (-4000..=4000).map(|i| i as f32 * 0.004).collect();
        xs.extend([
            -40.,
            -12.,
            -9.5,
            9.5,
            40.,
            0.,
            -0.,
            1e-30,
            -1e-30,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ]);
        let (values, slopes) = gelu_value_and_slope(&xs);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&values), bits(&gelu_array(&xs)));
        assert_eq!(bits(&slopes), bits(&gelu_derivative_array(&xs)));
        assert!(gelu_array(&[]).is_empty());
    }
    #[test]
    fn registry_is_unique_and_preserves_tensor_metadata() {
        let input = TensorData::new(vec![0.2, 0.4, 0.6, 0.8], vec![2, 2])
            .unwrap()
            .permute(&[1, 0])
            .unwrap();
        for (i, registration) in UNARY_KERNELS.iter().enumerate() {
            assert!(
                UNARY_KERNELS[..i]
                    .iter()
                    .all(|r| r.op != registration.op && r.name != registration.name)
            );
            assert_eq!(unary_by_name(registration.name).unwrap(), registration.op);
            let out = unary(registration.op, &input).unwrap();
            assert_eq!(out.shape(), input.shape());
            assert_eq!(out.dtype(), input.dtype());
            assert_eq!(out.device(), input.device());
            assert!(!out.shares_storage(&input));
            assert!(out.is_contiguous());
        }
        assert!(unary_by_name("missing").is_err());
    }
}

#[cfg(test)]
mod binary_tests {
    use super::*;
    use crate::ops::BinaryOp;
    #[test]
    fn binary_registration_handles_layouts_and_shared_inference() {
        let a = TensorData::new((0..6).map(|x| x as f32).collect(), vec![2, 3])
            .unwrap()
            .permute(&[1, 0])
            .unwrap();
        let b = TensorData::new(vec![10., 20.], vec![2]).unwrap();
        assert_eq!(
            infer_binary_shape(BinaryOp::Add, a.shape(), b.shape()).unwrap(),
            vec![3, 2]
        );
        assert_eq!(
            binary(BinaryOp::Add, &a, &b).unwrap().values(),
            &[10., 23., 11., 24., 12., 25.]
        );
        assert_eq!(
            binary(BinaryOp::Mul, &a, &b).unwrap().values(),
            &[0., 60., 10., 80., 20., 100.]
        );
        let w = TensorData::new(vec![2., 3.], vec![2, 1]).unwrap();
        assert_eq!(matmul(&a, &w).unwrap().values(), &[9., 14., 19.]);
        assert_eq!(
            infer_matmul_shape(a.shape(), w.shape()).unwrap(),
            vec![3, 1]
        );
    }
    #[test]
    fn shape_inference_rejects_incompatible_and_overflow_shapes() {
        assert!(infer_matmul_shape(&[3], &[3, 2]).is_err());
        assert!(infer_matmul_shape(&[2, 3], &[4, 2]).is_err());
        assert!(infer_matmul_shape(&[usize::MAX, 0], &[0, 2]).is_err());
        assert_eq!(infer_matmul_shape(&[0, 3], &[3, 2]).unwrap(), vec![0, 2]);
        assert!(infer_binary_shape(BinaryOp::Mul, &[2, 3], &[2]).is_err());
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BinaryKind {
    Sub,
    Div,
    Add,
    Mul,
    Matmul,
}
type BinaryKernel = fn(&TensorData, &TensorData) -> Result<TensorData, String>;
type BinaryShape = fn(&[usize], &[usize]) -> Result<Vec<usize>, String>;
struct BinaryRegistration {
    kind: BinaryKind,
    device: Device,
    dtype: DType,
    infer: BinaryShape,
    kernel: BinaryKernel,
}
static BINARY_KERNELS: &[BinaryRegistration] = &[
    BinaryRegistration {
        kind: BinaryKind::Sub,
        device: Device::Cpu,
        dtype: DType::Float32,
        infer: crate::ops::broadcast_shape,
        kernel: |a, b| crate::ops::binary_cpu(a, b, crate::ops::BinaryOp::Sub),
    },
    BinaryRegistration {
        kind: BinaryKind::Div,
        device: Device::Cpu,
        dtype: DType::Float32,
        infer: crate::ops::broadcast_shape,
        kernel: |a, b| crate::ops::binary_cpu(a, b, crate::ops::BinaryOp::Div),
    },
    BinaryRegistration {
        kind: BinaryKind::Add,
        device: Device::Cpu,
        dtype: DType::Float32,
        infer: crate::ops::broadcast_shape,
        kernel: |a, b| crate::ops::binary_cpu(a, b, crate::ops::BinaryOp::Add),
    },
    BinaryRegistration {
        kind: BinaryKind::Mul,
        device: Device::Cpu,
        dtype: DType::Float32,
        infer: crate::ops::broadcast_shape,
        kernel: |a, b| crate::ops::binary_cpu(a, b, crate::ops::BinaryOp::Mul),
    },
    BinaryRegistration {
        kind: BinaryKind::Matmul,
        device: Device::Cpu,
        dtype: DType::Float32,
        infer: matmul_shape,
        kernel: crate::ops::matmul_cpu,
    },
];
fn binary_kind(op: crate::ops::BinaryOp) -> BinaryKind {
    match op {
        crate::ops::BinaryOp::Sub => BinaryKind::Sub,
        crate::ops::BinaryOp::Div => BinaryKind::Div,
        crate::ops::BinaryOp::Add => BinaryKind::Add,
        crate::ops::BinaryOp::Mul => BinaryKind::Mul,
    }
}
fn run_binary(kind: BinaryKind, a: &TensorData, b: &TensorData) -> Result<TensorData, String> {
    let registration = BINARY_KERNELS
        .iter()
        .find(|r| {
            r.kind == kind
                && r.device == a.device()
                && r.device == b.device()
                && r.dtype == a.dtype()
                && r.dtype == b.dtype()
        })
        .ok_or("no registered kernel for operand device/dtype pair")?;
    (registration.kernel)(a, b)
}
pub fn binary(
    op: crate::ops::BinaryOp,
    a: &TensorData,
    b: &TensorData,
) -> Result<TensorData, String> {
    run_binary(binary_kind(op), a, b)
}
pub fn matmul(a: &TensorData, b: &TensorData) -> Result<TensorData, String> {
    run_binary(BinaryKind::Matmul, a, b)
}
fn infer(kind: BinaryKind, a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    // Shape schemas are independent of data layout and kernel execution.
    let registration = BINARY_KERNELS
        .iter()
        .find(|r| r.kind == kind)
        .ok_or("unregistered shape schema")?;
    (registration.infer)(a, b)
}
pub fn infer_binary_shape(
    op: crate::ops::BinaryOp,
    a: &[usize],
    b: &[usize],
) -> Result<Vec<usize>, String> {
    infer(binary_kind(op), a, b)
}
pub fn infer_matmul_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    infer(BinaryKind::Matmul, a, b)
}
fn matmul_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    if a.len() != 2 || b.len() != 2 {
        return Err("matmul requires rank 2".into());
    }
    if a[1] != b[0] {
        return Err("matmul inner dimensions differ".into());
    }
    for shape in [a, b] {
        crate::tensor::TensorLayout::count(shape)?;
    }
    let output = vec![a[0], b[1]];
    crate::tensor::TensorLayout::count(&output)?;
    Ok(output)
}

#[cfg(test)]
mod reduction_tests {
    use super::*;
    #[test]
    fn reduction_schema_handles_axes_scalars_and_empty_dimensions() {
        let p = ReductionSpec::new(&[2, 3, 4], Some(&[2, 0]), true).unwrap();
        assert_eq!(p.output_shape, vec![1, 3, 1]);
        assert_eq!(p.divisor().unwrap(), 8);
        assert_eq!(
            infer_reduction_shape(&[], None, false).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            ReductionSpec::new(&[], None, false)
                .unwrap()
                .divisor()
                .unwrap(),
            1
        );
        assert_eq!(
            ReductionSpec::new(&[0, 3], Some(&[0]), false)
                .unwrap()
                .divisor()
                .unwrap(),
            0
        );
        assert_eq!(
            infer_reduction_shape(&[2, 3], Some(&[]), false).unwrap(),
            vec![2, 3]
        );
        assert_eq!(
            ReductionSpec::new(&[0, 3], Some(&[]), false)
                .unwrap()
                .divisor()
                .unwrap(),
            1
        );
        assert!(ReductionSpec::new(&[2, 3], Some(&[1, 1]), false).is_err());
        assert!(ReductionSpec::new(&[2, 3], Some(&[2]), false).is_err());
        assert!(ReductionSpec::new(&[usize::MAX, 2], None, false).is_err());
        assert!(ReductionSpec::new(&[0, usize::MAX, 2], Some(&[0]), false).is_err());
        // A selected zero dimension makes the divisor zero before multiplying huge dimensions.
        assert_eq!(
            ReductionSpec::new(&[usize::MAX, 2, 0], None, false)
                .unwrap()
                .divisor()
                .unwrap(),
            0
        );
    }
    #[test]
    fn reduction_output_capacity_errors_and_empty_axes_are_checked() {
        let empty = TensorData::new(vec![], vec![0, usize::MAX]).unwrap();
        assert!(reduce_axes(&empty, &[0], false, false).is_err());
        let huge = TensorData::new(vec![], vec![0, usize::MAX, 2]).unwrap();
        assert!(
            reduce_axes(&huge, &[1, 2], false, false)
                .unwrap()
                .0
                .values()
                .is_empty()
        );
        assert!(
            reduce_axes(&huge, &[1, 2], false, true)
                .unwrap()
                .0
                .values()
                .is_empty()
        );
    }
    #[test]
    fn registered_reductions_preserve_strided_logical_order() {
        let a = TensorData::new((0..6).map(|x| x as f32).collect(), vec![2, 3])
            .unwrap()
            .permute(&[1, 0])
            .unwrap();
        assert_eq!(reduce_whole(&a, false).unwrap().values(), &[15.]);
        assert_eq!(reduce_whole(&a, true).unwrap().values(), &[2.5]);
        let (out, mapping) = reduce_axes(&a, &[1], true, false).unwrap();
        assert_eq!(out.shape(), &[3, 1]);
        assert_eq!(out.values(), &[3., 5., 7.]);
        assert_eq!(mapping, vec![0, 0, 1, 1, 2, 2]);
    }
}

#[cfg(test)]
mod normalization_tests {
    use super::*;
    #[test]
    fn layer_norm_schema_checks_shapes_features_epsilon_and_overflow() {
        let spec = LayerNormSpec::new(&[2, 3], Some(&[3]), Some(&[3]), 1e-5, Some(3)).unwrap();
        assert_eq!((spec.rows, spec.features), (2, 3));
        assert_eq!(
            infer_layer_norm_shape(&[0, 3], None, None, 1e-5, Some(3)).unwrap(),
            vec![0, 3]
        );
        assert!(LayerNormSpec::new(&[], None, None, 1e-5, None).is_err());
        assert!(LayerNormSpec::new(&[2, 0], None, None, 1e-5, None).is_err());
        assert!(LayerNormSpec::new(&[2, 3], Some(&[1, 3]), None, 1e-5, None).is_err());
        assert!(LayerNormSpec::new(&[2, 3], None, Some(&[2]), 1e-5, None).is_err());
        assert!(LayerNormSpec::new(&[2, 3], None, None, 1e-5, Some(4)).is_err());
        assert!(LayerNormSpec::new(&[usize::MAX, 2], None, None, 1e-5, None).is_err());
        for eps in [0., -1., f32::NAN, f32::INFINITY] {
            assert!(LayerNormSpec::new(&[2, 3], None, None, eps, None).is_err());
        }
    }
    #[test]
    fn registered_layer_norm_matches_existing_cpu_kernel_on_views() {
        let input = TensorData::new(vec![0., 1., 2., 3., 4., 5.], vec![3, 2])
            .unwrap()
            .permute(&[1, 0])
            .unwrap();
        let w = TensorData::new(vec![1., 2., 3.], vec![3]).unwrap();
        let b = TensorData::new(vec![0.2, -0.1, 0.3], vec![3]).unwrap();
        for (weight, bias) in [
            (None, None),
            (Some(&w), None),
            (None, Some(&b)),
            (Some(&w), Some(&b)),
        ] {
            let got = layer_norm(&input, weight, bias, 1e-5).unwrap();
            let mut expected = Vec::new();
            crate::normalization::forward_into(
                input.values(),
                weight.map(TensorData::values),
                bias.map(TensorData::values),
                2,
                3,
                1e-5,
                &mut expected,
            )
            .unwrap();
            assert_eq!(got.values(), expected);
            assert_eq!(got.shape(), input.shape());
            assert_eq!(got.dtype(), input.dtype());
            assert_eq!(got.device(), input.device());
            assert!(!got.shares_storage(&input));
        }
    }
}

#[cfg(test)]
mod loss_tests {
    use super::*;
    #[test]
    fn loss_schemas_check_shapes_overflow_and_explicit_empty_policy() {
        assert_eq!(MseSpec::new(&[], &[]).unwrap().elements, 1);
        assert!(MseSpec::new(&[0, 3], &[0, 3]).is_err());
        assert!(MseSpec::new(&[2, 3], &[3]).is_err());
        assert!(MseSpec::new(&[usize::MAX, 2], &[usize::MAX, 2]).is_err());
        assert!(CrossEntropySpec::new(&[2, 0], &[2], CrossEntropyMode::EagerIndices).is_err());
        assert!(CrossEntropySpec::new(&[2, 3], &[1], CrossEntropyMode::CompiledTensor).is_err());
        assert!(
            CrossEntropySpec::new(
                &[usize::MAX, 2],
                &[usize::MAX],
                CrossEntropyMode::EagerIndices
            )
            .is_err()
        );
        assert!(CrossEntropySpec::new(&[0, 3], &[0], CrossEntropyMode::EagerIndices).is_ok());
        assert!(CrossEntropySpec::new(&[0, 3], &[0], CrossEntropyMode::CompiledTensor).is_err());
        assert_eq!(
            infer_mse_shape(&[2, 3], &[2, 3]).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            infer_cross_entropy_shape(&[2, 3], &[2]).unwrap(),
            Vec::<usize>::new()
        );
    }
}

#[cfg(test)]
mod linear_tests {
    use super::*;
    #[test]
    fn linear_schema_preserves_degenerate_dimensions_and_checks_overflow() {
        assert_eq!(
            infer_linear_shape(&[2, 3], &[4, 3], Some(&[4])).unwrap(),
            vec![2, 4]
        );
        assert_eq!(
            infer_linear_shape(&[0, 3], &[4, 3], None).unwrap(),
            vec![0, 4]
        );
        assert_eq!(
            infer_linear_shape(&[2, 0], &[4, 0], Some(&[4])).unwrap(),
            vec![2, 4]
        );
        assert_eq!(
            infer_linear_shape(&[2, 3], &[0, 3], Some(&[0])).unwrap(),
            vec![2, 0]
        );
        assert!(LinearSpec::new(&[2, 3], &[4, 2], None).is_err());
        assert!(LinearSpec::new(&[2, 3], &[4, 3], Some(&[1, 4])).is_err());
        assert!(LinearSpec::new(&[usize::MAX, 0], &[2, 0], None).is_err());
        assert!(LinearSpec::new(&[usize::MAX, 2], &[0, 2], None).is_err());
    }
}

#[cfg(test)]
mod activation_tests {
    use super::*;
    #[test]
    fn activation_shape_schemas_validate_axes_and_zero_widths() {
        assert_eq!(infer_relu_shape(&[]).unwrap(), Vec::<usize>::new());
        assert_eq!(SoftmaxSpec::new(&[0, 3], 1).unwrap().width, 3);
        assert!(SoftmaxSpec::new(&[2, 0], 1).is_err());
        assert!(SoftmaxSpec::new(&[], 0).is_err());
        assert!(SoftmaxSpec::new(&[2, 3], 2).is_err());
        assert!(infer_relu_shape(&[usize::MAX, 2]).is_err());
        let x = TensorData::new(vec![-2., 1., 3., -4.], vec![2, 2])
            .unwrap()
            .permute(&[1, 0])
            .unwrap();
        assert_eq!(relu(&x).unwrap().values(), &[0., 3., 1., 0.]);
    }
}
