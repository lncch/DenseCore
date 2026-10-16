//! Rank2 float32 feature LayerNorm and first-order derivatives.
use crate::tensor::TensorData;
pub fn dimensions(
    x: &TensorData,
    weight: Option<&TensorData>,
    bias: Option<&TensorData>,
    eps: f32,
) -> Result<(usize, usize), String> {
    let spec = crate::dispatch::LayerNormSpec::new(
        x.shape(),
        weight.map(TensorData::shape),
        bias.map(TensorData::shape),
        eps,
        None,
    )?;
    Ok((spec.rows, spec.features))
}
pub fn validate_epsilon(eps: f32) -> Result<(), String> {
    if !eps.is_finite() || eps <= 0. {
        return Err("layer_norm eps must be positive finite float32".into());
    }
    Ok(())
}
// PyTorch's float32 layer norm (aten/src/ATen/native/cpu/layer_norm_kernel.cpp and
// moments_utils.h, PyTorch 2.14.1): Welford moments over 4-lane vectors in cascaded chunks
// of 16, then (x - mean) * rstd * gamma + beta. Ported so the bits match; PyTorch is BSD-3
// (THIRD_PARTY_NOTICES.md).

const VS: usize = 4; // Vectorized<float>::size() on aarch64
const CHUNK: usize = 16; // kChunkSize

// Scalar C++ statements clang may contract into a fused multiply-add (fp-contract=on).
const F_TAIL: bool = true; // m2 += delta * (x - m1)
const F_M1: bool = true; // m1 += c * delta
const F_M2: bool = true; // m2 += m2_add + delta * delta * c * m0
const F_SECOND: bool = true; // (x + bias) * scale * gamma + beta, scalar loop

type L = [f32; VS];
fn lanes(x: &[f32]) -> L {
    std::array::from_fn(|i| x[i])
}
fn fmadd(a: L, b: L, c: L) -> L {
    std::array::from_fn(|i| a[i].mul_add(b[i], c[i]))
}
fn ceil_log2(x: usize) -> usize {
    if x <= 2 {
        1
    } else {
        (usize::BITS - (x - 1).leading_zeros()) as usize
    }
}
/// `AddMomentsVec`.
fn add_moments_vec(m0_add: usize, m1_add: L, m2_add: L, m0: &mut usize, m1: &mut L, m2: &mut L) {
    let n = *m0 + m0_add;
    let c = if n == 0 {
        0.0
    } else {
        m0_add as f32 / n as f32
    };
    let delta: L = std::array::from_fn(|i| m1_add[i] - m1[i]);
    let m2_tmp: L = std::array::from_fn(|i| m2[i] + m2_add[i]);
    let c_delta: L = std::array::from_fn(|i| c * delta[i]);
    let m0_delta: L = std::array::from_fn(|i| delta[i] * *m0 as f32);
    *m1 = std::array::from_fn(|i| m1[i] + c_delta[i]);
    *m2 = fmadd(m0_delta, c_delta, m2_tmp);
    *m0 = n;
}
/// `AddMoments` (scalar).
fn add_moments(
    m0_add: usize,
    m1_add: f32,
    m2_add: f32,
    m0: &mut usize,
    m1: &mut f32,
    m2: &mut f32,
) {
    let n = *m0 + m0_add;
    let c = if n == 0 {
        0.0
    } else {
        m0_add as f32 / n as f32
    };
    let delta = m1_add - *m1;
    *m1 = if F_M1 {
        c.mul_add(delta, *m1)
    } else {
        *m1 + c * delta
    };
    let t = if F_M2 {
        (delta * delta * c).mul_add(*m0 as f32, m2_add)
    } else {
        m2_add + delta * delta * c * *m0 as f32
    };
    *m2 += t;
    *m0 = n;
}
/// `RowwiseMoments(X, N)`: (mean, biased variance).
fn rowwise_moments(x: &[f32]) -> (f32, f32) {
    let total = x.len();
    let n = total / VS;
    let m = n.div_ceil(CHUNK);
    let depth = ceil_log2(m);
    let mut m0_stk = vec![0usize; depth];
    let mut m1_stk = vec![[0.0f32; VS]; depth];
    let mut m2_stk = vec![[0.0f32; VS]; depth];
    for i in 0..m {
        let base = i * CHUNK * VS;
        let m0 = CHUNK.min(n - i * CHUNK);
        // UpdateMomentsVec
        let (mut m1v, mut m2v) = ([0.0f32; VS], [0.0f32; VS]);
        for j in 0..m0 {
            let xv = lanes(&x[base + j * VS..]);
            let c = [1.0 / (j + 1) as f32; VS];
            let delta: L = std::array::from_fn(|l| xv[l] - m1v[l]);
            m1v = fmadd(c, delta, m1v);
            let delta2: L = std::array::from_fn(|l| xv[l] - m1v[l]);
            m2v = fmadd(delta, delta2, m2v);
        }
        let (a, b, cc) = (&mut m0_stk[0], &mut m1_stk[0], &mut m2_stk[0]);
        add_moments_vec(m0, m1v, m2v, a, b, cc);
        let mut mask = i + 1;
        let mut j = 1;
        while j < depth && mask & 1 == 0 {
            let (m0a, m1a, m2a) = (m0_stk[j - 1], m1_stk[j - 1], m2_stk[j - 1]);
            let (mut m0b, mut m1b, mut m2b) = (m0_stk[j], m1_stk[j], m2_stk[j]);
            add_moments_vec(m0a, m1a, m2a, &mut m0b, &mut m1b, &mut m2b);
            (m0_stk[j], m1_stk[j], m2_stk[j]) = (m0b, m1b, m2b);
            (m0_stk[j - 1], m1_stk[j - 1], m2_stk[j - 1]) = (0, [0.0; VS], [0.0; VS]);
            mask >>= 1;
            j += 1;
        }
    }
    for i in 1..depth {
        let (m0a, m1a, m2a) = (m0_stk[i], m1_stk[i], m2_stk[i]);
        let (mut m0b, mut m1b, mut m2b) = (m0_stk[0], m1_stk[0], m2_stk[0]);
        add_moments_vec(m0a, m1a, m2a, &mut m0b, &mut m1b, &mut m2b);
        (m0_stk[0], m1_stk[0], m2_stk[0]) = (m0b, m1b, m2b);
    }
    let (m1_arr, m2_arr) = (m1_stk[0], m2_stk[0]);
    let (mut m0, mut m1, mut m2) = (0usize, 0.0f32, 0.0f32);
    for &v in &x[n * VS..] {
        let delta = v - m1;
        m0 += 1;
        m1 += delta / m0 as f32;
        m2 = if F_TAIL {
            delta.mul_add(v - m1, m2)
        } else {
            m2 + delta * (v - m1)
        };
    }
    for l in 0..VS {
        add_moments(n, m1_arr[l], m2_arr[l], &mut m0, &mut m1, &mut m2);
    }
    (m1, m2 / total as f32)
}
/// Row statistics as PyTorch keeps them: (mean, rstd).
fn statistics(row: &[f32], eps: f32) -> (f32, f32) {
    let (mean, var) = rowwise_moments(row);
    (mean, 1.0 / (var + eps).sqrt())
}
#[allow(clippy::too_many_arguments)]
pub fn forward_into(
    x: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    eps: f32,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    validate_epsilon(eps)?;
    if n == 0
        || x.len() != m.checked_mul(n).ok_or("layer_norm size overflow")?
        || weight.is_some_and(|w| w.len() != n)
        || bias.is_some_and(|b| b.len() != n)
    {
        return Err("layer_norm input length mismatch".into());
    }
    out.resize(x.len(), 0.);
    for (row, result) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let (mean, rstd) = statistics(row, eps);
        let (scale, shift) = (rstd, -mean);
        match (weight, bias) {
            // vec::map3: vector ops, nothing fused.
            (Some(w), Some(b)) => {
                for (j, (&v, dst)) in row.iter().zip(result).enumerate() {
                    *dst = (v + shift) * scale * w[j] + b[j];
                }
            }
            _ => {
                for (j, (&v, dst)) in row.iter().zip(result).enumerate() {
                    let g = weight.map_or(1.0, |w| w[j]);
                    let b = bias.map_or(0.0, |b| b[j]);
                    let t = (v + shift) * scale;
                    *dst = if F_SECOND { t.mul_add(g, b) } else { t * g + b };
                }
            }
        }
    }
    Ok(())
}
pub fn forward(
    x: &TensorData,
    weight: Option<&TensorData>,
    bias: Option<&TensorData>,
    eps: f32,
) -> Result<TensorData, String> {
    crate::dispatch::layer_norm(x, weight, bias, eps)
}
pub(crate) fn forward_cpu(
    x: &TensorData,
    weight: Option<&TensorData>,
    bias: Option<&TensorData>,
    eps: f32,
) -> Result<TensorData, String> {
    let (m, n) = dimensions(x, weight, bias, eps)?;
    let mut out = Vec::new();
    forward_into(
        x.values(),
        weight.map(TensorData::values),
        bias.map(TensorData::values),
        m,
        n,
        eps,
        &mut out,
    )?;
    TensorData::new(out, x.shape().to_vec())
}
#[derive(Clone, Copy)]
pub enum Gradient {
    Input,
    Weight,
    Bias,
}
/// `vec::map2_reduce_all` / `map3_reduce_all` / `reduce_all` with `+`.
fn reduce_all(values: impl Fn(usize) -> f32, size: usize) -> f32 {
    crate::torch_softmax::reduce_all(values, size, crate::torch_softmax::add)
}
#[allow(clippy::too_many_arguments)]
pub fn gradient_into(
    x: Option<&[f32]>,
    weight: Option<&[f32]>,
    g: &[f32],
    m: usize,
    n: usize,
    eps: f32,
    kind: Gradient,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    validate_epsilon(eps)?;
    let count = m.checked_mul(n).ok_or("layer_norm size overflow")?;
    if n == 0
        || g.len() != count
        || weight.is_some_and(|w| w.len() != n)
        || (!matches!(kind, Gradient::Bias) && x.is_none_or(|x| x.len() != count))
    {
        return Err("layer_norm gradient length mismatch".into());
    }
    if matches!(kind, Gradient::Weight | Gradient::Bias) {
        // Two-pass reduction: rows split over the threads, one buffer per thread, then the
        // buffers added in thread order.
        let threads = crate::torch_sum::num_threads();
        let mut buffers = vec![vec![0.0f32; n]; threads];
        let rows_per = if m == 0 {
            0
        } else {
            m.div_ceil(threads.min(m))
        };
        for i in 0..m {
            let buffer = &mut buffers[i / rows_per.max(1)];
            let seeds = &g[i * n..(i + 1) * n];
            if matches!(kind, Gradient::Bias) {
                for (dst, &dy) in buffer.iter_mut().zip(seeds) {
                    *dst += dy;
                }
            } else {
                let row = &x.expect("validated normalization input")[i * n..(i + 1) * n];
                let (mean, rstd) = statistics(row, eps);
                let (a, b) = (rstd, -rstd * mean);
                for ((dst, &dy), &v) in buffer.iter_mut().zip(seeds).zip(row) {
                    *dst += dy * (a * v + b);
                }
            }
        }
        out.clear();
        out.extend((0..n).map(|j| buffers.iter().fold(0.0f32, |acc, b| acc + b[j])));
        return Ok(());
    }
    out.resize(count, 0.);
    let x = x.expect("validated normalization input");
    let scale = 1.0 / n as f32;
    for (i, (row, seeds)) in x.chunks_exact(n).zip(g.chunks_exact(n)).enumerate() {
        let (mean, rstd) = statistics(row, eps);
        let (ds, db) = match weight {
            None => (
                reduce_all(|j| seeds[j] * row[j], n),
                reduce_all(|j| seeds[j], n),
            ),
            Some(w) => (
                reduce_all(|j| seeds[j] * row[j] * w[j], n),
                reduce_all(|j| seeds[j] * w[j], n),
            ),
        };
        let a = rstd;
        // clang contracts both statements (found by testing against PyTorch):
        // b = (db * mean - ds) * a^3 * scale, c = -b * mean - db * a * scale.
        let b = db.mul_add(mean, -ds) * a * a * a * scale;
        let c = (-b).mul_add(mean, -(db * a * scale));
        let dst = &mut out[i * n..(i + 1) * n];
        match weight {
            None => {
                for (j, d) in dst.iter_mut().enumerate() {
                    *d = a * seeds[j] + b * row[j] + c;
                }
            }
            Some(w) => {
                for (j, d) in dst.iter_mut().enumerate() {
                    *d = a * seeds[j] * w[j] + b * row[j] + c;
                }
            }
        }
    }
    Ok(())
}
