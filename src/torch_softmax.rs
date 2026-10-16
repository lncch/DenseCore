//! PyTorch's float32 CPU softmax and log_softmax over the last axis, bit for bit.
//!
//! Ported from PyTorch 2.14.1 (git 5c48869): `_vec_softmax_lastdim`,
//! `_vec_host_softmax_backward_lastdim` (aten/src/ATen/native/cpu/SoftMaxKernel.cpp),
//! `serial_vec_log_softmax_lastdim_range` (LogSoftmaxKernelImpl.h) and the `vec::reduce_all`
//! family (aten/src/ATen/cpu/vec/functional_base.h). PyTorch is BSD-3; see
//! THIRD_PARTY_NOTICES.md.

const VS: usize = 4; // Vectorized<float>::size() on aarch64

/// `vec::reduce_all` / `map_reduce_all` / `map2_reduce_all` with a lambda reduction: 4-lane
/// partial results (the last partial vector only updates its first lanes), then the NEON
/// tree op(op(a0, a2), op(a1, a3)); fewer than 4 elements combine lane by lane in order.
pub(crate) fn reduce_all(
    values: impl Fn(usize) -> f32,
    size: usize,
    op: impl Fn(f32, f32) -> f32,
) -> f32 {
    if size < VS {
        let mut acc = values(0);
        for i in 1..size {
            acc = op(acc, values(i));
        }
        return acc;
    }
    let mut acc: [f32; VS] = std::array::from_fn(&values);
    let whole = size - size % VS;
    let mut d = VS;
    while d < whole {
        acc = std::array::from_fn(|l| op(acc[l], values(d + l)));
        d += VS;
    }
    for (l, slot) in acc.iter_mut().enumerate().take(size - d) {
        *slot = op(*slot, values(d + l));
    }
    op(op(acc[0], acc[2]), op(acc[1], acc[3]))
}

pub(crate) fn add(a: f32, b: f32) -> f32 {
    a + b
}

/// `vec::maximum` on NEON (FMAX): NaN if either is NaN, and max(+0, -0) is +0.
pub(crate) fn maximum(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_positive() { a } else { b }
    } else if a > b {
        a
    } else {
        b
    }
}

/// Softmax of each `width`-long row.
pub fn softmax_lastdim(x: &[f32], width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(width) {
        let max = reduce_all(|i| row[i], width, maximum);
        let shifted: Vec<f32> = row.iter().map(|&v| v - max).collect();
        let e = crate::sleef::exp_array(&shifted);
        let inv = 1.0 / reduce_all(|i| e[i], width, add);
        out.extend(e.iter().map(|&v| v * inv));
    }
    out
}

/// Log-softmax of each `width`-long row.
pub fn log_softmax_lastdim(x: &[f32], width: usize) -> Vec<f32> {
    let rows = x.len() / width.max(1);
    let mut maxima = Vec::with_capacity(rows);
    let mut sums = Vec::with_capacity(rows);
    for row in x.chunks_exact(width) {
        let max = reduce_all(|i| row[i], width, maximum);
        let shifted: Vec<f32> = row.iter().map(|&v| v - max).collect();
        let e = crate::sleef::exp_array(&shifted);
        maxima.push(max);
        sums.push(reduce_all(|i| e[i], width, add));
    }
    let logs = crate::sleef::log_array(&sums);
    let mut out = Vec::with_capacity(x.len());
    for ((row, &max), &log_sum) in x.chunks_exact(width).zip(&maxima).zip(&logs) {
        out.extend(row.iter().map(|&v| v - max - log_sum));
    }
    out
}

/// Softmax gradient per row: (g - sum(g * y)) * y.
pub fn softmax_backward_lastdim(g: &[f32], y: &[f32], width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(g.len());
    for (gr, yr) in g.chunks_exact(width).zip(y.chunks_exact(width)) {
        let sum = reduce_all(|i| gr[i] * yr[i], width, add);
        out.extend(gr.iter().zip(yr).map(|(&g, &y)| (g - sum) * y));
    }
    out
}

/// Log-softmax gradient per row: g - exp(y) * sum(g).
pub fn log_softmax_backward_lastdim(g: &[f32], y: &[f32], width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(g.len());
    for (gr, yr) in g.chunks_exact(width).zip(y.chunks_exact(width)) {
        let sum = reduce_all(|i| gr[i], width, add);
        let e = crate::sleef::exp_array(yr);
        out.extend(gr.iter().zip(&e).map(|(&g, &e)| g - e * sum));
    }
    out
}
