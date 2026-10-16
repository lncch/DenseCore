use crate::tensor::TensorData;
#[derive(Clone, Copy, Debug)]
pub enum BinaryOp {
    Sub,
    Div,
    Add,
    Mul,
}
#[derive(Clone)]
pub enum UnaryOp {
    Math(crate::dispatch::UnaryMath),
    Relu,
    Softmax(usize),
    LogSoftmax(usize),
    Sum,
    Mean,
    Transpose,
    Reshape(Vec<usize>),
    /// A fresh contiguous copy (clone / contiguous).
    Clone,
    Permute(Vec<usize>),
}
#[derive(Clone)]
pub struct BroadcastPlan {
    pub shape: Vec<usize>,
    pub left: Vec<usize>,
    pub right: Vec<usize>,
}
fn count(shape: &[usize]) -> Result<usize, String> {
    crate::tensor::TensorLayout::count(shape)
}
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let rank = a.len().max(b.len());
    let mut result = vec![1; rank];
    for (axis, dimension) in result.iter_mut().enumerate() {
        let ai = a.len().checked_sub(rank - axis).map(|i| a[i]).unwrap_or(1);
        let bi = b.len().checked_sub(rank - axis).map(|i| b[i]).unwrap_or(1);
        if ai != bi && ai != 1 && bi != 1 {
            return Err(format!("unsupported broadcast: {a:?} with {b:?}"));
        }
        // A zero-length axis broadcast with a singleton remains empty.
        *dimension = if ai == 1 { bi } else { ai };
    }
    count(&result)?;
    count(a)?;
    count(b)?;
    Ok(result)
}

pub fn broadcast_strides(input: &[usize], output: &[usize]) -> Result<Vec<usize>, String> {
    if input.len() > output.len() {
        return Err("input rank exceeds output rank".into());
    }
    let mut strides = vec![0; output.len()];
    let mut stride = 1usize;
    for i in (0..input.len()).rev() {
        let axis = output.len() - input.len() + i;
        if input[i] != output[axis] && input[i] != 1 {
            return Err("incompatible broadcast dimensions".into());
        }
        if input[i] != 1 {
            strides[axis] = stride;
        }
        stride = stride
            .checked_mul(input[i])
            .ok_or("shape product overflow")?;
    }
    Ok(strides)
}

pub fn broadcast_plan(a: &[usize], b: &[usize]) -> Result<BroadcastPlan, String> {
    let shape = broadcast_shape(a, b)?;
    let n = count(&shape)?;
    if n == 0 {
        return Ok(BroadcastPlan {
            shape,
            left: Vec::new(),
            right: Vec::new(),
        });
    }
    let a_strides = broadcast_strides(a, &shape)?;
    let b_strides = broadcast_strides(b, &shape)?;
    let mut left = Vec::new();
    let mut right = Vec::new();
    left.try_reserve_exact(n)
        .map_err(|_| "broadcast index allocation failed")?;
    right
        .try_reserve_exact(n)
        .map_err(|_| "broadcast index allocation failed")?;
    for flat in 0..n {
        let mut remaining = flat;
        let mut ai = 0;
        let mut bi = 0;
        for axis in (0..shape.len()).rev() {
            let coordinate = remaining % shape[axis];
            remaining /= shape[axis];
            ai += coordinate * a_strides[axis];
            bi += coordinate * b_strides[axis];
        }
        left.push(ai);
        right.push(bi);
    }
    Ok(BroadcastPlan { shape, left, right })
}
/// Some(n) when `small` broadcasts into `big` by repeating over leading axes (a scalar, a
/// bias row, a position table): result element i then reads small[i % n].
pub(crate) fn leading_repeat(big: &[usize], small: &[usize]) -> Option<usize> {
    let first = small.iter().position(|&d| d != 1).unwrap_or(small.len());
    let core = &small[first..];
    let n: usize = core.iter().product();
    let total: usize = big.iter().product();
    (small.len() <= big.len() && big.ends_with(core) && n > 0 && total > 0).then_some(n)
}
pub fn binary(a: &TensorData, b: &TensorData, op: BinaryOp) -> Result<TensorData, String> {
    crate::dispatch::binary(op, a, b)
}
pub(crate) fn binary_cpu(
    a: &TensorData,
    b: &TensorData,
    op: BinaryOp,
) -> Result<TensorData, String> {
    // One monomorphized loop per op: a match inside the loop compiled to all four ops
    // (including a divide) per element plus a select.
    match op {
        BinaryOp::Sub => broadcast_map(a, b, |l, r| l - r),
        BinaryOp::Div => broadcast_map(a, b, |l, r| l / r),
        BinaryOp::Add => broadcast_map(a, b, |l, r| l + r),
        BinaryOp::Mul => broadcast_map(a, b, |l, r| l * r),
    }
}
/// Elementwise map over two broadcast operands, producing the row-major result.
pub(crate) fn broadcast_map(
    a: &TensorData,
    b: &TensorData,
    f: impl Fn(f32, f32) -> f32,
) -> Result<TensorData, String> {
    if a.shape() == b.shape() {
        // Same shape: skip the per-element index plan; arithmetic is unchanged.
        let data = a
            .values()
            .iter()
            .zip(b.values())
            .map(|(&l, &r)| f(l, r))
            .collect();
        return TensorData::new(data, a.shape().to_vec());
    }
    // One side repeats over the other's leading axes: walk it in chunks, no index plan.
    if let Some(n) = leading_repeat(a.shape(), b.shape()) {
        let small = b.values();
        if n == 1 {
            // A scalar operand: a flat map vectorizes, a chunk-of-one flat_map does not.
            let r = small[0];
            let data = a.values().iter().map(|&l| f(l, r)).collect();
            return TensorData::new(data, a.shape().to_vec());
        }
        let data = a
            .values()
            .chunks_exact(n)
            .flat_map(|chunk| chunk.iter().zip(small).map(|(&l, &r)| f(l, r)))
            .collect();
        return TensorData::new(data, a.shape().to_vec());
    }
    if let Some(n) = leading_repeat(b.shape(), a.shape()) {
        let small = a.values();
        if n == 1 {
            let l = small[0];
            let data = b.values().iter().map(|&r| f(l, r)).collect();
            return TensorData::new(data, b.shape().to_vec());
        }
        let data = b
            .values()
            .chunks_exact(n)
            .flat_map(|chunk| small.iter().zip(chunk).map(|(&l, &r)| f(l, r)))
            .collect();
        return TensorData::new(data, b.shape().to_vec());
    }
    let p = broadcast_plan(a.shape(), b.shape())?;
    let data = p
        .left
        .iter()
        .zip(&p.right)
        .map(|(&i, &j)| f(a.values()[i], b.values()[j]))
        .collect();
    TensorData::new(data, p.shape)
}
pub fn reduce_axes(
    a: &TensorData,
    axes: &[usize],
    keepdims: bool,
    mean: bool,
) -> Result<(TensorData, Vec<usize>), String> {
    crate::dispatch::reduce_axes(a, axes, keepdims, mean)
}

pub fn permute(a: &TensorData, axes: &[usize]) -> Result<TensorData, String> {
    a.permute(axes)
}

pub fn softmax(a: &TensorData, axis: usize, logarithmic: bool) -> Result<TensorData, String> {
    crate::dispatch::softmax(a, axis, logarithmic)
}
pub(crate) fn softmax_cpu(
    a: &TensorData,
    axis: usize,
    logarithmic: bool,
) -> Result<TensorData, String> {
    let shape = a.shape();
    let width = crate::dispatch::SoftmaxSpec::new(shape, axis)?.width;
    if a.numel() == 0 {
        return TensorData::new(vec![], shape.to_vec());
    }
    let inner = shape[axis + 1..].iter().product::<usize>();
    let outer = shape[..axis].iter().product::<usize>();
    if inner == 1 {
        // Last axis: PyTorch's kernels, bit for bit (crate::torch_softmax).
        let values = a.values();
        let output = if logarithmic {
            crate::torch_softmax::log_softmax_lastdim(values, width)
        } else {
            crate::torch_softmax::softmax_lastdim(values, width)
        };
        return TensorData::new(output, shape.to_vec());
    }
    let mut output = vec![0.0f32; a.values().len()];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * width * inner + i;
            let mut maximum = f32::NEG_INFINITY;
            for k in 0..width {
                maximum = maximum.max(a.values()[base + k * inner]);
            }
            // Each exponential is computed once and kept in the output slot.
            let mut normalizer = 0.0f32;
            for k in 0..width {
                let index = base + k * inner;
                let e = (a.values()[index] - maximum).exp();
                output[index] = e;
                normalizer += e;
            }
            let log_normalizer = normalizer.ln();
            for k in 0..width {
                let index = base + k * inner;
                output[index] = if logarithmic {
                    (a.values()[index] - maximum) - log_normalizer
                } else {
                    output[index] / normalizer
                };
            }
        }
    }
    TensorData::new(output, shape.to_vec())
}

pub fn softmax_grad(
    y: &TensorData,
    gradient: &TensorData,
    axis: usize,
    logarithmic: bool,
) -> Result<TensorData, String> {
    if y.shape() != gradient.shape() || axis >= y.shape().len() {
        return Err("invalid softmax gradient".into());
    }
    let shape = y.shape();
    let width = shape[axis];
    if width == 0 {
        return Err("softmax requires a nonempty reduction axis".into());
    }
    if y.numel() == 0 {
        return TensorData::new(vec![], shape.to_vec());
    }
    let inner = shape[axis + 1..].iter().product::<usize>();
    let outer = shape[..axis].iter().product::<usize>();
    if inner == 1 {
        let (yv, gv) = (y.values(), gradient.values());
        let output = if logarithmic {
            crate::torch_softmax::log_softmax_backward_lastdim(gv, yv, width)
        } else {
            crate::torch_softmax::softmax_backward_lastdim(gv, yv, width)
        };
        return TensorData::new(output, shape.to_vec());
    }
    let mut output = vec![0.0f32; y.values().len()];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * width * inner + i;
            let mut total = 0.0f32;
            for k in 0..width {
                let index = base + k * inner;
                total += if logarithmic {
                    gradient.values()[index]
                } else {
                    gradient.values()[index] * y.values()[index]
                };
            }
            for k in 0..width {
                let index = base + k * inner;
                output[index] = if logarithmic {
                    gradient.values()[index] - y.values()[index].exp() * total
                } else {
                    y.values()[index] * (gradient.values()[index] - total)
                };
            }
        }
    }
    TensorData::new(output, shape.to_vec())
}

pub(crate) fn relu_cpu(a: &TensorData) -> Result<TensorData, String> {
    TensorData::new(
        a.values()
            .iter()
            .map(|&x| if x < 0.0 { 0.0 } else { x })
            .collect(),
        a.shape().to_vec(),
    )
}
pub fn unary(a: &TensorData, op: &UnaryOp) -> Result<TensorData, String> {
    match op {
        UnaryOp::Math(kind) => crate::dispatch::unary(*kind, a),
        UnaryOp::Reshape(s) => a.reshape(s.clone()),
        UnaryOp::Clone => TensorData::new(a.values().to_vec(), a.shape().to_vec()),
        UnaryOp::Permute(axes) => permute(a, axes),
        UnaryOp::Softmax(axis) => softmax(a, *axis, false),
        UnaryOp::LogSoftmax(axis) => softmax(a, *axis, true),
        UnaryOp::Relu => crate::dispatch::relu(a),
        UnaryOp::Sum | UnaryOp::Mean => {
            crate::dispatch::reduce_whole(a, matches!(op, UnaryOp::Mean))
        }
        UnaryOp::Transpose => {
            if a.shape().len() != 2 {
                return Err("transpose requires rank 2".into());
            }
            a.permute(&[1, 0])
        }
    }
}
pub fn matmul(a: &TensorData, b: &TensorData) -> Result<TensorData, String> {
    crate::dispatch::matmul(a, b)
}
pub(crate) fn matmul_cpu(a: &TensorData, b: &TensorData) -> Result<TensorData, String> {
    let shape = crate::dispatch::infer_matmul_shape(a.shape(), b.shape())?;
    let (m, k, n) = (shape[0], a.shape()[1], shape[1]);
    let v = crate::cpu_kernels::gemm(a.values(), b.values(), m, k, n)?;
    TensorData::new(v, shape)
}

pub fn linear(
    x: &TensorData,
    w: &TensorData,
    bias: Option<&TensorData>,
) -> Result<TensorData, String> {
    crate::dispatch::linear(x, w, bias)
}
pub(crate) fn linear_cpu(
    x: &TensorData,
    w: &TensorData,
    bias: Option<&TensorData>,
) -> Result<TensorData, String> {
    let spec = crate::dispatch::LinearSpec::new(x.shape(), w.shape(), bias.map(TensorData::shape))?;
    let (m, k, n) = (spec.rows, spec.inputs, spec.outputs);
    let mut out = Vec::new();
    crate::cpu_kernels::gemm_into(
        x.values(),
        w.values(),
        m,
        k,
        n,
        false,
        true,
        bias.map(|b| b.values()),
        &mut out,
    )?;
    TensorData::new(out, vec![m, n])
}
/// Replace masked logical elements with value; mask is row-major and matches x.
pub(crate) fn masked_fill(x: &TensorData, mask: &[bool], value: f32) -> Result<TensorData, String> {
    if mask.len() != x.numel() {
        return Err("masked_fill mask must have one entry per element".into());
    }
    let values = x
        .values()
        .iter()
        .zip(mask)
        .map(|(&v, &m)| if m { value } else { v })
        .collect();
    TensorData::new(values, x.shape().to_vec())
}
/// Gradient of masked_fill: masked elements receive exactly zero.
pub(crate) fn masked_fill_grad(g: &TensorData, mask: &[bool]) -> Result<TensorData, String> {
    let values = g
        .values()
        .iter()
        .zip(mask)
        .map(|(&v, &m)| if m { 0.0 } else { v })
        .collect();
    TensorData::new(values, g.shape().to_vec())
}
/// Row gather for embeddings: output shape is index_shape + [embedding_dim].
pub(crate) fn embedding(
    weight: &TensorData,
    indices: &[usize],
    index_shape: &[usize],
) -> Result<TensorData, String> {
    let [rows, dim] = weight.shape() else {
        return Err("embedding weight must be rank2 (num_embeddings, embedding_dim)".into());
    };
    let (rows, dim) = (*rows, *dim);
    if crate::tensor::TensorLayout::count(index_shape)? != indices.len() {
        return Err("embedding index shape does not match index count".into());
    }
    if let Some(&bad) = indices.iter().find(|&&i| i >= rows) {
        return Err(format!(
            "embedding index {bad} out of range for {rows} rows"
        ));
    }
    let total = indices
        .len()
        .checked_mul(dim)
        .ok_or("embedding output size overflow")?;
    let table = weight.values();
    let mut out = Vec::with_capacity(total);
    for &i in indices {
        out.extend_from_slice(&table[i * dim..(i + 1) * dim]);
    }
    let mut shape = index_shape.to_vec();
    shape.push(dim);
    TensorData::new(out, shape)
}
/// Scatter-add of output gradients back into a zeroed weight-shaped gradient.
pub(crate) fn embedding_grad(
    weight_shape: &[usize],
    indices: &[usize],
    g: &TensorData,
) -> Result<TensorData, String> {
    let dim = weight_shape[1];
    let count = crate::tensor::TensorLayout::count(weight_shape)?;
    let mut values = vec![0.; count];
    for (&i, row) in indices.iter().zip(g.values().chunks(dim.max(1))) {
        for (dst, &part) in values[i * dim..(i + 1) * dim].iter_mut().zip(row) {
            *dst += part;
        }
    }
    TensorData::new(values, weight_shape.to_vec())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn leading_repeat_detects_only_row_major_repeats() {
        assert_eq!(leading_repeat(&[4, 3, 5], &[3, 5]), Some(15));
        assert_eq!(leading_repeat(&[4, 3, 5], &[1, 1, 5]), Some(5));
        assert_eq!(leading_repeat(&[4, 3, 5], &[]), Some(1));
        assert_eq!(leading_repeat(&[4, 3, 5], &[1]), Some(1));
        assert_eq!(leading_repeat(&[4, 3, 5], &[3, 1]), None);
        assert_eq!(leading_repeat(&[3, 5], &[1, 1, 5]), None);
        assert_eq!(leading_repeat(&[0, 5], &[5]), None);
        assert_eq!(leading_repeat(&[], &[1]), None);
    }
    #[test]
    fn masked_fill_replaces_exactly_and_zeroes_gradients() {
        let x = TensorData::new(vec![1., f32::INFINITY, 3., f32::NAN], vec![2, 2]).unwrap();
        let mask = [false, true, false, true];
        let out = masked_fill(&x, &mask, f32::NEG_INFINITY).unwrap();
        assert_eq!(&out.values()[..3], &[1., f32::NEG_INFINITY, 3.]);
        assert_eq!(out.values()[3], f32::NEG_INFINITY);
        let t = x.permute(&[1, 0]).unwrap();
        let out = masked_fill(&t, &[true, false, false, false], 0.).unwrap();
        assert_eq!(&out.values()[1..3], &[3., f32::INFINITY]);
        let g = TensorData::new(vec![5., 6., 7., 8.], vec![2, 2]).unwrap();
        assert_eq!(
            masked_fill_grad(&g, &mask).unwrap().values(),
            &[5., 0., 7., 0.]
        );
        assert!(masked_fill(&x, &[true], 0.).is_err());
    }
    #[test]
    fn embedding_gathers_rows_and_scatters_repeated_gradients() {
        let w = TensorData::new((0..6).map(|v| v as f32).collect(), vec![3, 2]).unwrap();
        let out = embedding(&w, &[2, 0, 2], &[3]).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.values(), &[4., 5., 0., 1., 4., 5.]);
        let strided = w.permute(&[1, 0]).unwrap().permute(&[1, 0]).unwrap();
        assert_eq!(embedding(&strided, &[1], &[]).unwrap().shape(), &[2]);
        let g = TensorData::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]).unwrap();
        let dw = embedding_grad(&[3, 2], &[2, 0, 2], &g).unwrap();
        assert_eq!(dw.values(), &[3., 4., 0., 0., 6., 8.]);
        assert!(embedding(&w, &[3], &[1]).is_err());
        assert!(embedding(&w, &[0, 1], &[3]).is_err());
        let empty = embedding(&w, &[], &[0, 4]).unwrap();
        assert_eq!(empty.shape(), &[0, 4, 2]);
        let flat = TensorData::new(vec![1.; 3], vec![3]).unwrap();
        assert!(embedding(&flat, &[0], &[1]).is_err());
    }
    #[test]
    fn row_maps_and_overflow() {
        let p = broadcast_plan(&[2, 3], &[3]).unwrap();
        assert_eq!(p.right, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(
            broadcast_plan(&[2, 3], &[2, 1]).unwrap().right,
            vec![0, 0, 0, 1, 1, 1]
        );
        assert!(broadcast_plan(&[usize::MAX, 2], &[]).is_err());
    }
    #[test]
    fn matmul_validation() {
        let a = TensorData::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        assert!(matmul(&a, &a).is_err());
    }
}

#[cfg(test)]
mod empty_layout_tests {
    use super::*;
    #[test]
    fn softmax_skips_unreachable_empty_dimensions() {
        let x = TensorData::new(vec![], vec![usize::MAX, usize::MAX, 0, 2]).unwrap();
        for logarithmic in [false, true] {
            let y = softmax(&x, 3, logarithmic).unwrap();
            assert!(y.values().is_empty());
            let g = softmax_grad(&y, &y, 3, logarithmic).unwrap();
            assert!(g.values().is_empty());
        }
    }
}
