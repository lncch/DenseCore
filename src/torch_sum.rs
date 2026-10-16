//! PyTorch's float32 CPU sum, reproduced bit for bit.
//!
//! Ported from PyTorch 2.14.1 (git 5c48869): the cascade sum in
//! `aten/src/ATen/native/cpu/SumKernel.cpp`, the dimension order and coalescing of
//! `TensorIterator`, and the thread split of `TensorIteratorReduce.cpp`. PyTorch is
//! Copyright (c) 2016- Facebook, Inc and contributors, BSD-3-Clause; see THIRD_PARTY_NOTICES.md.
//!
//! The thread count only decides how the work is split (and so the rounding); the sums run on
//! one thread here and give the bits PyTorch gives with that many threads.
use std::sync::atomic::{AtomicUsize, Ordering};

/// `Vectorized<float>::size()` on aarch64.
const VS: usize = 4;
/// `at::internal::GRAIN_SIZE`.
const GRAIN: usize = 32768;

static THREADS: AtomicUsize = AtomicUsize::new(0);

/// PyTorch's default intra-op thread count: the performance cores on Apple silicon.
fn default_threads() -> usize {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                old: *mut std::ffi::c_void,
                len: *mut usize,
                new: *mut std::ffi::c_void,
                newlen: usize,
            ) -> std::ffi::c_int;
        }
        let mut value: std::ffi::c_int = 0;
        let mut len = std::mem::size_of::<std::ffi::c_int>();
        // SAFETY: the name is NUL-terminated and `value`/`len` describe a valid c_int buffer.
        let status = unsafe {
            sysctlbyname(
                c"hw.perflevel0.physicalcpu".as_ptr(),
                (&raw mut value).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if status == 0 && value > 0 {
            return value as usize;
        }
    }
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

pub fn num_threads() -> usize {
    match THREADS.load(Ordering::Relaxed) {
        0 => {
            let n = default_threads();
            THREADS.store(n, Ordering::Relaxed);
            n
        }
        n => n,
    }
}

pub fn set_num_threads(n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("number of threads must be positive".into());
    }
    THREADS.store(n, Ordering::Relaxed);
    Ok(())
}

trait Acc: Copy {
    const ZERO: Self;
    fn plus(self, other: Self) -> Self;
}

impl Acc for f32 {
    const ZERO: f32 = 0.0;
    fn plus(self, other: f32) -> f32 {
        self + other
    }
}

#[derive(Clone, Copy)]
struct Lanes([f32; VS]);

impl Acc for Lanes {
    const ZERO: Lanes = Lanes([0.0; VS]);
    fn plus(self, other: Lanes) -> Lanes {
        let mut out = self.0;
        for (a, b) in out.iter_mut().zip(other.0) {
            *a += b;
        }
        Lanes(out)
    }
}

fn lanes(src: &[f32], at: usize) -> Lanes {
    let mut out = [0.0; VS];
    out.copy_from_slice(&src[at..at + VS]);
    Lanes(out)
}

/// `utils::CeilLog2`.
fn ceil_log2(x: usize) -> usize {
    if x <= 2 {
        1
    } else {
        (usize::BITS - (x - 1).leading_zeros()) as usize
    }
}

/// `multi_row_sum`: `load(i, k)` is row i of lane-row k.
fn multi_row_sum<A: Acc, const N: usize>(size: usize, load: impl Fn(usize, usize) -> A) -> [A; N] {
    const LEVELS: usize = 4;
    let level_power = 4.max(ceil_log2(size) / LEVELS);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut acc = [[A::ZERO; N]; LEVELS];
    let mut i = 0;
    while i + level_step <= size {
        for _ in 0..level_step {
            for (k, a) in acc[0].iter_mut().enumerate() {
                *a = a.plus(load(i, k));
            }
            i += 1;
        }
        for j in 1..LEVELS {
            let (lower, upper) = acc.split_at_mut(j);
            for (high, low) in upper[0].iter_mut().zip(lower[j - 1].iter_mut()) {
                *high = high.plus(*low);
                *low = A::ZERO;
            }
            if i & (level_mask << (j * level_power)) != 0 {
                break;
            }
        }
    }
    while i < size {
        for (k, a) in acc[0].iter_mut().enumerate() {
            *a = a.plus(load(i, k));
        }
        i += 1;
    }
    let (first, rest) = acc.split_at_mut(1);
    for level in rest.iter() {
        for (total, &v) in first[0].iter_mut().zip(level) {
            *total = total.plus(v);
        }
    }
    acc[0]
}

/// `row_sum` with an ILP factor of 4.
fn row_sum<A: Acc>(size: usize, load: impl Fn(usize) -> A) -> A {
    let size_ilp = size / 4;
    let mut partial = multi_row_sum::<A, 4>(size_ilp, |i, k| load(i * 4 + k));
    for i in size_ilp * 4..size {
        partial[0] = partial[0].plus(load(i));
    }
    for k in 1..4 {
        partial[0] = partial[0].plus(partial[k]);
    }
    partial[0]
}

/// One row of `vectorized_inner_sum` (row.len() >= VS).
fn inner_sum(row: &[f32]) -> f32 {
    let vec_size = row.len() / VS;
    let vector = row_sum::<Lanes>(vec_size, |i| lanes(row, i * VS));
    let mut total = 0.0f32;
    for &x in &row[vec_size * VS..] {
        total += x;
    }
    for x in vector.0 {
        total += x;
    }
    total
}

/// One iterator dimension: size, input stride and output stride, in elements.
#[derive(Clone, Copy, Debug)]
struct Dim {
    size: usize,
    input: usize,
    output: usize,
}

/// The `cascade_sum` loop body for the two innermost dimensions; accumulates into `out`.
fn loop2d(src: &[f32], ib: usize, out: &mut [f32], ob: usize, d0: Dim, d1: Dim) {
    let (mut d0, mut d1) = (d0, d1);
    // Move the reduction to be the first dimension.
    if d0.output != 0 && d1.output == 0 {
        std::mem::swap(&mut d0, &mut d1);
    }
    let (size0, size1) = (d0.size, d1.size);
    let (in0, in1, out1) = (d0.input, d1.input, d1.output);
    if d0.output != 0 && d1.output != 0 {
        // Not a true reduction: out += in, elementwise.
        for j in 0..size1 {
            for i in 0..size0 {
                out[ob + i * d0.output + j * out1] += src[ib + i * in0 + j * in1];
            }
        }
        return;
    }
    let mut store = |j: usize, value: f32| out[ob + j * out1] += value;
    if in0 == 1 && size0 >= VS {
        for j in 0..size1 {
            let start = ib + j * in1;
            store(j, inner_sum(&src[start..start + size0]));
        }
    } else if in1 == 1 && size1 >= VS {
        let mut j = 0;
        while j + 4 * VS <= size1 {
            let sums =
                multi_row_sum::<Lanes, 4>(size0, |i, k| lanes(src, ib + i * in0 + j + k * VS));
            for (k, sum) in sums.iter().enumerate() {
                for (l, &v) in sum.0.iter().enumerate() {
                    store(j + k * VS + l, v);
                }
            }
            j += 4 * VS;
        }
        while j + VS <= size1 {
            let sum = row_sum::<Lanes>(size0, |i| lanes(src, ib + i * in0 + j));
            for (l, &v) in sum.0.iter().enumerate() {
                store(j + l, v);
            }
            j += VS;
        }
        while j < size1 {
            store(j, row_sum::<f32>(size0, |i| src[ib + i * in0 + j]));
            j += 1;
        }
    } else if in0 < in1 {
        for j in 0..size1 {
            store(j, row_sum::<f32>(size0, |i| src[ib + j * in1 + i * in0]));
        }
    } else {
        let mut j = 0;
        while j + 3 < size1 {
            let sums = multi_row_sum::<f32, 4>(size0, |i, k| src[ib + (j + k) * in1 + i * in0]);
            for (k, &v) in sums.iter().enumerate() {
                store(j + k, v);
            }
            j += 4;
        }
        while j < size1 {
            store(j, row_sum::<f32>(size0, |i| src[ib + j * in1 + i * in0]));
            j += 1;
        }
    }
}

/// `serial_for_each` over all dimensions: loop2d on the first two, an odometer over the rest
/// (dimension 2 fastest).
fn for_each(src: &[f32], ib: usize, out: &mut [f32], ob: usize, dims: &[Dim]) {
    let unit = Dim {
        size: 1,
        input: 0,
        output: 0,
    };
    let d0 = dims.first().copied().unwrap_or(unit);
    let d1 = dims.get(1).copied().unwrap_or(unit);
    let outer = dims.get(2..).unwrap_or(&[]);
    if outer.iter().any(|d| d.size == 0) {
        return;
    }
    let mut index = vec![0usize; outer.len()];
    let (mut ib, mut ob) = (ib, ob);
    loop {
        loop2d(src, ib, out, ob, d0, d1);
        let mut axis = 0;
        loop {
            if axis == outer.len() {
                return;
            }
            index[axis] += 1;
            ib += outer[axis].input;
            ob += outer[axis].output;
            if index[axis] < outer[axis].size {
                break;
            }
            ib -= outer[axis].input * outer[axis].size;
            ob -= outer[axis].output * outer[axis].size;
            index[axis] = 0;
            axis += 1;
        }
    }
}

/// `TensorIterator` reorder (reduced dims first, then ascending stride, ties by size) and
/// coalescing, for a contiguous row-major input.
fn iterator_dims(shape: &[usize], reduced: &[bool]) -> Vec<Dim> {
    let rank = shape.len();
    let mut input = vec![1usize; rank];
    let mut output = vec![0usize; rank];
    let (mut si, mut so) = (1usize, 1usize);
    for axis in (0..rank).rev() {
        input[axis] = si;
        si *= shape[axis];
        if !reduced[axis] {
            output[axis] = so;
            so *= shape[axis];
        }
    }
    let should_swap = |a: usize, b: usize| -> i32 {
        let (oa, ob) = (output[a], output[b]);
        if (oa == 0) != (ob == 0) {
            return if ob == 0 { 1 } else { -1 };
        }
        for (sa, sb) in [(oa, ob), (input[a], input[b])] {
            if sa == 0 || sb == 0 {
                continue;
            }
            if sa < sb {
                return -1;
            }
            if sa > sb {
                return 1;
            }
            if shape[a] > shape[b] {
                return 1;
            }
        }
        0
    };
    let mut perm: Vec<usize> = (0..rank).rev().collect();
    for i in 1..rank {
        let mut d1 = i;
        for d0 in (0..i).rev() {
            match should_swap(perm[d0], perm[d1]) {
                c if c > 0 => {
                    perm.swap(d0, d1);
                    d1 = d0;
                }
                c if c < 0 => break,
                _ => {}
            }
        }
    }
    let mut dims: Vec<Dim> = Vec::with_capacity(rank);
    for &axis in &perm {
        if shape[axis] == 1 {
            continue;
        }
        let next = Dim {
            size: shape[axis],
            input: input[axis],
            output: output[axis],
        };
        if let Some(last) = dims.last_mut()
            && next.input == last.size * last.input
            && next.output == last.size * last.output
        {
            last.size *= next.size;
            continue;
        }
        dims.push(next);
    }
    dims
}

/// `find_split_dim`: the outer-most kept dimension at least as large as the thread count,
/// else the largest kept one met before the first reduced dimension (from the outside).
fn split_dim(dims: &[Dim], threads: usize) -> usize {
    let mut best = dims.len() - 1;
    for d in (0..dims.len()).rev() {
        if dims[d].output == 0 {
            break;
        }
        if dims[d].size >= threads {
            return d;
        }
        if dims[d].size > dims[best].size {
            best = d;
        }
    }
    best
}

/// OpenMP `invoke_parallel` chunks of [0, n) for `threads` threads and a grain size.
fn chunks(n: usize, threads: usize, grain: usize) -> Vec<(usize, usize)> {
    let count = threads.min(n.div_ceil(grain)).max(1);
    let size = n.div_ceil(count);
    (0..count)
        .map(|t| (t * size, n.min((t + 1) * size)))
        .filter(|(b, e)| b < e)
        .collect()
}

/// `torch.sum(x, axes)` of contiguous row-major float32 values. `reduced[axis]` marks the
/// summed axes; the result is row-major over the kept axes.
pub fn sum(values: &[f32], shape: &[usize], reduced: &[bool]) -> Vec<f32> {
    let threads = num_threads();
    let kept: usize = shape
        .iter()
        .zip(reduced)
        .filter(|(_, r)| !**r)
        .map(|(n, _)| *n)
        .product();
    let mut out = vec![0.0f32; kept];
    let numel = values.len();
    if numel == 0 || kept == 0 {
        return out;
    }
    let dims = iterator_dims(shape, reduced);
    if numel < GRAIN || threads == 1 {
        for_each(values, 0, &mut out, 0, &dims);
    } else if kept == 1 {
        // Two-pass reduction: one partial per thread over a contiguous range, then their sum.
        let mut buffer = vec![0.0f32; threads];
        for (t, (begin, end)) in chunks(numel, threads, GRAIN).into_iter().enumerate() {
            let part = [Dim {
                size: end - begin,
                input: 1,
                output: 0,
            }];
            for_each(values, begin, &mut buffer, t, &part);
        }
        let whole = [Dim {
            size: threads,
            input: 1,
            output: 0,
        }];
        for_each(&buffer, 0, &mut out, 0, &whole);
    } else {
        let d = split_dim(&dims, threads);
        let cols = dims[d].size;
        // Round column ranges to 128 bytes when the split dimension is contiguous in the input.
        let per = (dims[d].input == 1).then_some(128 / std::mem::size_of::<f32>());
        for (begin, end) in chunks(cols, threads, 1) {
            let (begin, end) = match per {
                Some(p) => (
                    begin - begin % p,
                    if end == cols { end } else { end - end % p },
                ),
                None => (begin, end),
            };
            if begin == end {
                continue;
            }
            let mut narrowed = dims.clone();
            narrowed[d].size = end - begin;
            let (ib, ob) = (begin * dims[d].input, begin * dims[d].output);
            for_each(values, ib, &mut out, ob, &narrowed);
        }
    }
    out
}

/// `linalg.vector_norm(x, 2)` over every element of a contiguous float32 tensor: the vectorized
/// last-dim path of `norm_kernel_tensor_iterator_impl` (4 lanes of unfused x*x sums, lanes
/// added in order, then a fused scalar tail). A single element takes the `abs` shortcut.
pub fn norm2(values: &[f32]) -> f32 {
    if values.len() == 1 {
        return values[0].abs();
    }
    let whole = values.len() / 4 * 4;
    let mut acc = [0.0f32; 4];
    for chunk in values[..whole].as_chunks::<4>().0 {
        acc = std::array::from_fn(|l| acc[l] + chunk[l] * chunk[l]);
    }
    let mut total = acc[0] + acc[1] + acc[2] + acc[3];
    for &x in &values[whole..] {
        total = x.mul_add(x, total);
    }
    total.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_matches_pytorch_utils() {
        let want = [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
            (1 << 20, 20),
            ((1 << 20) + 1, 21),
        ];
        for (x, y) in want {
            assert_eq!(ceil_log2(x), y, "{x}");
        }
    }

    #[test]
    fn reduced_dims_come_first_then_by_stride() {
        // (12, 4, 64, 32) summed over axes 0 and 2: both reduced dims lead, kept dims follow.
        let dims = iterator_dims(&[12, 4, 64, 32], &[true, false, true, false]);
        let sizes: Vec<usize> = dims.iter().map(|d| d.size).collect();
        assert_eq!(sizes, vec![64, 12, 32, 4]);
    }
}
