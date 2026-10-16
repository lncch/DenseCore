//! PyTorch's float32 CPU flash attention (no mask, no dropout), bit for bit.
//!
//! Ported from PyTorch 2.14.1 (git 5c48869): `cpu_flash_attention` and
//! `cpu_flash_attention_backward` (aten/src/ATen/native/cpu/FlashAttentionKernel.cpp), the float
//! path without packing. PyTorch is BSD-3; see THIRD_PARTY_NOTICES.md.
//!
//! Tensors here are contiguous (batch, heads, length, head_dim); `lse` is (batch, heads, length).
//! Every (batch, head) task is independent, so the thread split never changes the bits.

use crate::sleef::{V, exp_u20, expf, libm_exp, libm_log};
use crate::torch_blas::gemm_raw;
use crate::torch_softmax::{add, maximum, reduce_all};

const KV_SPLIT: usize = 512;

#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub batch: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub q_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
}

/// `flash_attention_kernel_impl`'s q split from the query length (the kv split is 512).
fn q_split(q_len: usize) -> usize {
    if q_len >= 768 {
        256
    } else if q_len >= 192 {
        64
    } else {
        32
    }
}

/// `sdp::calculate_scale`: computed in double, stored as float.
pub fn scale_factor(head_dim: usize, scale: Option<f64>) -> f32 {
    scale.unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt()) as f32
}

/// `std::max`: the first argument unless it is less than the second.
fn std_max(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

/// `_mul_reduce_max_fusion_kernel`: row *= scale in place, returns the row max.
fn mul_reduce_max(row: &mut [f32], scale: f32) -> f32 {
    let size = row.len();
    let (end_n, end) = (size / 8 * 8, size / 4 * 4);
    let mut acc = [[f32::NEG_INFINITY; 4]; 2];
    let mut tail = [f32::NEG_INFINITY; 4];
    let mut i = 0;
    while i < end_n {
        for (h, a) in acc.iter_mut().enumerate() {
            for (l, slot) in a.iter_mut().enumerate() {
                let x = row[i + 4 * h + l] * scale;
                row[i + 4 * h + l] = x;
                *slot = maximum(*slot, x);
            }
        }
        i += 8;
    }
    while i < end {
        for (l, slot) in tail.iter_mut().enumerate() {
            let x = row[i + l] * scale;
            row[i + l] = x;
            *slot = maximum(*slot, x);
        }
        i += 4;
    }
    for l in 0..4 {
        acc[0][l] = maximum(acc[0][l], tail[l]);
    }
    let mut tmp_max = f32::NEG_INFINITY;
    for x in &mut row[i..] {
        *x *= scale;
        tmp_max = std_max(tmp_max, *x);
    }
    let v: [f32; 4] = std::array::from_fn(|l| maximum(acc[0][l], acc[1][l]));
    let reduced = maximum(maximum(v[0], v[2]), maximum(v[1], v[3]));
    if reduced.is_nan() {
        f32::NAN
    } else {
        std_max(tmp_max, reduced)
    }
}

/// `_exp_reduce_sum_fusion_kernel` (float): row = exp(row - max) in place, returns the sum.
fn exp_reduce_sum(row: &mut [f32], max: f32) -> f32 {
    let end = row.len() / 4 * 4;
    let mut acc = [0.0f32; 4];
    for chunk in row[..end].as_chunks_mut::<4>().0 {
        let e = exp_u20(V(chunk.map(|x| x - max))).0;
        acc = std::array::from_fn(|l| acc[l] + e[l]);
        *chunk = e;
    }
    let mut sum = (acc[0] + acc[2]) + (acc[1] + acc[3]);
    for x in &mut row[end..] {
        let e = libm_exp(*x - max);
        sum += e;
        *x = e;
    }
    sum
}

/// Splits `data` into `parts` equal chunks and runs `f(index, chunk)` over all threads.
fn par_chunks(data: &mut [f32], parts: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    if parts == 0 {
        return;
    }
    let size = data.len() / parts;
    let threads = crate::torch_sum::num_threads().min(parts).max(1);
    let mut buckets: Vec<Vec<(usize, &mut [f32])>> = (0..threads).map(|_| Vec::new()).collect();
    for (i, chunk) in data.chunks_mut(size.max(1)).take(parts).enumerate() {
        buckets[i * threads / parts].push((i, chunk));
    }
    std::thread::scope(|s| {
        for bucket in buckets {
            let f = &f;
            s.spawn(move || {
                for (i, chunk) in bucket {
                    f(i, chunk);
                }
            });
        }
    });
}

/// `_scale_attn_mask_fusion_kernel` (float): row = row * scale + mask, unfused in 4-lane
/// groups and fused in the scalar tail.
fn scale_add_mask(row: &mut [f32], mask: &[f32], scale: f32) {
    let whole = row.len() / 4 * 4;
    for (i, (x, &m)) in row.iter_mut().zip(mask).enumerate() {
        *x = if i < whole {
            *x * scale + m
        } else {
            x.mul_add(scale, m)
        };
    }
}

/// Forward: (output, logsumexp). `mask` is an additive float mask already broadcast to
/// (batch, heads, q_len, kv_len).
#[allow(clippy::too_many_arguments)]
pub fn forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    d: Dims,
    causal: bool,
    scale: Option<f64>,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let e = d.head_dim;
    let scale = scale_factor(e, scale);
    let qs = q_split(d.q_len).min(d.q_len);
    let kvs = KV_SPLIT.min(d.kv_len);
    let repeat = d.heads / d.kv_heads;
    // Each (batch, head) task writes its output rows then its logsumexp row.
    let task = d.q_len * e + d.q_len;
    let mut both = vec![0.0f32; d.batch * d.heads * task];
    let failed = std::sync::Mutex::new(None);
    par_chunks(&mut both, d.batch * d.heads, |t, slot| {
        let (b, h) = (t / d.heads, t % d.heads);
        let qb_base = t * d.q_len * e;
        let kv_base = (b * d.kv_heads + h / repeat) * d.kv_len * e;
        let (out, lse) = slot.split_at_mut(d.q_len * e);
        let mut qk = vec![0.0f32; qs * kvs];
        let mut dst = vec![0.0f32; qs * e];
        let (mut qk_max, mut qk_sum) = (vec![0.0f32; qs], vec![0.0f32; qs]);
        for m in (0..d.q_len).step_by(qs.max(1)) {
            let qb = qs.min(d.q_len - m);
            qk_max[..qb].fill(f32::NEG_INFINITY);
            qk_sum[..qb].fill(0.0);
            let num_keys = if causal {
                (m + qb).min(d.kv_len)
            } else {
                d.kv_len
            };
            for n in (0..num_keys).step_by(kvs.max(1)) {
                let kvb = kvs.min(d.kv_len - n);
                let r = gemm_raw(
                    true,
                    false,
                    kvb,
                    qb,
                    e,
                    1.0,
                    &k[kv_base + n * e..],
                    e,
                    &q[qb_base + m * e..],
                    e,
                    0.0,
                    &mut qk,
                    kvb,
                );
                if let Err(err) = r {
                    *failed.lock().unwrap() = Some(err);
                    return;
                }
                if causal && num_keys - n <= kvs {
                    for row in 0..qb {
                        let from = (m + row + 1).saturating_sub(n).min(kvb);
                        qk[row * kvb + from..(row + 1) * kvb].fill(f32::NEG_INFINITY);
                    }
                }
                for row in 0..qb {
                    let r = &mut qk[row * kvb..(row + 1) * kvb];
                    let tmp_max = match mask {
                        Some(mask) => {
                            let at = (t * d.q_len + m + row) * d.kv_len + n;
                            scale_add_mask(r, &mask[at..at + kvb], scale);
                            reduce_all(|i| r[i], kvb, maximum)
                        }
                        None => mul_reduce_max(r, scale),
                    };
                    let tmp_max = if qk_max[row] > tmp_max {
                        qk_max[row]
                    } else {
                        tmp_max
                    };
                    if tmp_max == f32::NEG_INFINITY {
                        r.fill(0.0);
                        continue;
                    }
                    let tmp_sum = exp_reduce_sum(r, tmp_max);
                    let exp_tmp = libm_exp(qk_max[row] - tmp_max);
                    qk_sum[row] = exp_tmp.mul_add(qk_sum[row], tmp_sum);
                    qk_max[row] = tmp_max;
                    if n > 0 {
                        for x in &mut dst[row * e..(row + 1) * e] {
                            *x *= exp_tmp;
                        }
                    }
                }
                let beta = if n == 0 { 0.0 } else { 1.0 };
                let r = gemm_raw(
                    false,
                    false,
                    e,
                    qb,
                    kvb,
                    1.0,
                    &v[kv_base + n * e..],
                    e,
                    &qk,
                    kvb,
                    beta,
                    &mut dst,
                    e,
                );
                if let Err(err) = r {
                    *failed.lock().unwrap() = Some(err);
                    return;
                }
            }
            for row in 0..qb {
                if qk_max[row] == f32::NEG_INFINITY {
                    qk_max[row] = 0.0;
                }
                if qk_sum[row] == 0.0 {
                    qk_sum[row] = 1.0;
                }
                let recip = 1.0 / qk_sum[row];
                let o = &mut out[(m + row) * e..(m + row + 1) * e];
                for (o, x) in o.iter_mut().zip(&dst[row * e..(row + 1) * e]) {
                    *o = x * recip;
                }
                lse[m + row] = qk_max[row] + libm_log(qk_sum[row]);
            }
        }
    });
    if let Some(err) = failed.into_inner().unwrap() {
        return Err(err);
    }
    let mut out = Vec::with_capacity(d.batch * d.heads * d.q_len * e);
    let mut lse = Vec::with_capacity(d.batch * d.heads * d.q_len);
    for slot in both.chunks_exact(task.max(1)) {
        out.extend_from_slice(&slot[..d.q_len * e]);
        lse.extend_from_slice(&slot[d.q_len * e..]);
    }
    Ok((out, lse))
}

/// (grad_q, grad_k, grad_v)
type Grads = (Vec<f32>, Vec<f32>, Vec<f32>);

/// Backward: (grad_q, grad_k, grad_v).
#[allow(clippy::too_many_arguments)]
pub fn backward(
    grad_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    lse: &[f32],
    mask: Option<&[f32]>,
    d: Dims,
    causal: bool,
    scale: Option<f64>,
) -> Result<Grads, String> {
    let e = d.head_dim;
    let scale = scale_factor(e, scale);
    let qs = q_split(d.q_len).min(d.q_len);
    let kvs = KV_SPLIT.min(d.kv_len);
    let repeat = d.heads / d.kv_heads;
    // Each (batch, kv head) task owns its repeat query heads' grad_q plus its grad_k and grad_v.
    let (gq_len, gkv_len) = (repeat * d.q_len * e, d.kv_len * e);
    let task = gq_len + 2 * gkv_len;
    let mut all = vec![0.0f32; d.batch * d.kv_heads * task];
    let failed = std::sync::Mutex::new(None);
    par_chunks(&mut all, d.batch * d.kv_heads, |t, slot| {
        let kv_base = t * d.kv_len * e;
        let (gq_all, rest) = slot.split_at_mut(gq_len);
        let (gk, gv) = rest.split_at_mut(gkv_len);
        let mut attn = vec![0.0f32; qs * kvs];
        let mut grad_attn = vec![0.0f32; qs * kvs];
        let mut dsum = vec![0.0f32; qs];
        let mut run = || -> Result<(), String> {
            for r in 0..repeat {
                let j = t * repeat + r; // (batch, head) index of this query head
                let q_base = j * d.q_len * e;
                let gq = &mut gq_all[r * d.q_len * e..(r + 1) * d.q_len * e];
                for m in (0..d.q_len).step_by(qs.max(1)) {
                    let qb = qs.min(d.q_len - m);
                    for (row, slot) in dsum.iter_mut().enumerate().take(qb) {
                        let at = q_base + (m + row) * e;
                        *slot = reduce_all(|i| grad_out[at + i] * out[at + i], e, add);
                    }
                    let num_keys = if causal {
                        (m + qb).min(d.kv_len)
                    } else {
                        d.kv_len
                    };
                    for n in (0..num_keys).step_by(kvs.max(1)) {
                        let kvb = kvs.min(d.kv_len - n);
                        gemm_raw(
                            true,
                            false,
                            kvb,
                            qb,
                            e,
                            scale,
                            &k[kv_base + n * e..],
                            e,
                            &q[q_base + m * e..],
                            e,
                            0.0,
                            &mut attn,
                            kvb,
                        )?;
                        if let Some(mask) = mask {
                            // attn * 1 + mask: fused or not, the same bits.
                            for row in 0..qb {
                                let at = (j * d.q_len + m + row) * d.kv_len + n;
                                let r = &mut attn[row * kvb..(row + 1) * kvb];
                                for (x, &mv) in r.iter_mut().zip(&mask[at..at + kvb]) {
                                    *x += mv;
                                }
                            }
                        }
                        for row in 0..qb {
                            let norm = lse[j * d.q_len + m + row];
                            let r = &mut attn[row * kvb..(row + 1) * kvb];
                            for c in r.chunks_mut(4) {
                                let mut x = [0.0f32; 4];
                                for (l, &a) in c.iter().enumerate() {
                                    x[l] = a - norm;
                                }
                                let y = expf(V(x)).0;
                                c.copy_from_slice(&y[..c.len()]);
                            }
                        }
                        if causal && num_keys - n <= kvs {
                            for row in 0..qb {
                                let from = (m + row + 1).saturating_sub(n).min(kvb);
                                attn[row * kvb + from..(row + 1) * kvb].fill(0.0);
                            }
                        }
                        gemm_raw(
                            false,
                            true,
                            e,
                            kvb,
                            qb,
                            1.0,
                            &grad_out[q_base + m * e..],
                            e,
                            &attn,
                            kvb,
                            1.0,
                            &mut gv[n * e..],
                            e,
                        )?;
                        gemm_raw(
                            true,
                            false,
                            kvb,
                            qb,
                            e,
                            1.0,
                            &v[kv_base + n * e..],
                            e,
                            &grad_out[q_base + m * e..],
                            e,
                            0.0,
                            &mut grad_attn,
                            kvb,
                        )?;
                        for (row, &dr) in dsum.iter().enumerate().take(qb) {
                            let span = row * kvb..(row + 1) * kvb;
                            for (g, &a) in grad_attn[span.clone()].iter_mut().zip(&attn[span]) {
                                *g = a * (*g - dr);
                            }
                        }
                        gemm_raw(
                            false,
                            false,
                            e,
                            qb,
                            kvb,
                            scale,
                            &k[kv_base + n * e..],
                            e,
                            &grad_attn,
                            kvb,
                            1.0,
                            &mut gq[m * e..],
                            e,
                        )?;
                        gemm_raw(
                            false,
                            true,
                            e,
                            kvb,
                            qb,
                            scale,
                            &q[q_base + m * e..],
                            e,
                            &grad_attn,
                            kvb,
                            1.0,
                            &mut gk[n * e..],
                            e,
                        )?;
                    }
                }
            }
            Ok(())
        };
        if let Err(err) = run() {
            *failed.lock().unwrap() = Some(err);
        }
    });
    if let Some(err) = failed.into_inner().unwrap() {
        return Err(err);
    }
    let mut gq = Vec::with_capacity(d.batch * d.heads * d.q_len * e);
    let mut gk = Vec::with_capacity(d.batch * d.kv_heads * gkv_len);
    let mut gv = Vec::with_capacity(d.batch * d.kv_heads * gkv_len);
    for slot in all.chunks_exact(task.max(1)) {
        gq.extend_from_slice(&slot[..gq_len]);
        gk.extend_from_slice(&slot[gq_len..gq_len + gkv_len]);
        gv.extend_from_slice(&slot[gq_len + gkv_len..]);
    }
    Ok((gq, gk, gv))
}
