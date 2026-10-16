//! PyTorch's CPU float32 matrix multiply, reproduced bit for bit.
//!
//! Ported from PyTorch 2.14.1 (git 5c48869): `addmm_impl_cpu_` in
//! `aten/src/ATen/native/LinearAlgebra.cpp` (which operand is transposed, copied or swapped,
//! from the strides) and `cpublas::gemm` in `aten/src/ATen/native/CPUBlas.cpp`
//! (`normalize_last_dims`, then Accelerate's Fortran `sgemm_`). The same BLAS call on the same
//! machine gives the same bits. PyTorch is Copyright (c) 2016- Facebook, Inc and contributors,
//! BSD-3-Clause; see THIRD_PARTY_NOTICES.md.

/// A 2-D strided float32 view: element (i, j) is `data[offset + i * strides[0] + j * strides[1]]`.
#[derive(Clone, Copy)]
pub struct Mat<'a> {
    pub data: &'a [f32],
    pub offset: usize,
    pub sizes: [usize; 2],
    pub strides: [usize; 2],
}

/// PyTorch's contiguous strides: the last is 1, earlier ones skip size-0 dims as 1.
fn contiguous_strides(sizes: [usize; 2]) -> [usize; 2] {
    [sizes[1].max(1), 1]
}

impl<'a> Mat<'a> {
    /// A row-major contiguous matrix.
    pub fn rows(data: &'a [f32], rows: usize, cols: usize) -> Self {
        Mat {
            data,
            offset: 0,
            sizes: [rows, cols],
            strides: contiguous_strides([rows, cols]),
        }
    }
    /// The `.t()` view: same storage, sizes and strides swapped.
    pub fn t(self) -> Self {
        Mat {
            sizes: [self.sizes[1], self.sizes[0]],
            strides: [self.strides[1], self.strides[0]],
            ..self
        }
    }
    /// `clone(at::MemoryFormat::Contiguous)` values, row-major.
    fn to_contiguous(self) -> Vec<f32> {
        let [r, c] = self.sizes;
        let mut out = Vec::with_capacity(r * c);
        for i in 0..r {
            for j in 0..c {
                out.push(self.data[self.offset + i * self.strides[0] + j * self.strides[1]]);
            }
        }
        out
    }
    /// True when PyTorch's `mm_mat*_backward` treats it as column-major.
    pub fn is_column_major(self) -> bool {
        self.strides[0] == 1 && self.strides[1] == self.sizes[0]
    }
}

/// `cpublas::internal::normalize_last_dims`.
/// `ld` is (lda, ldb, ldc).
fn normalize_last_dims(ta: bool, tb: bool, m: usize, n: usize, k: usize, ld: &mut [usize; 3]) {
    if n == 1 {
        ld[2] = m;
    }
    if ta {
        if m == 1 {
            ld[0] = k;
        }
    } else if k == 1 {
        ld[0] = m;
    }
    if tb {
        if k == 1 {
            ld[1] = n;
        }
    } else if n == 1 {
        ld[1] = k;
    }
}

#[cfg(target_os = "macos")]
mod accelerate {
    use std::ffi::{c_char, c_int};
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        /// Fortran BLAS sgemm (column-major), as PyTorch calls it on macOS.
        pub fn sgemm_(
            transa: *const c_char,
            transb: *const c_char,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            alpha: *const f32,
            a: *const f32,
            lda: *const c_int,
            b: *const f32,
            ldb: *const c_int,
            beta: *const f32,
            c: *mut f32,
            ldc: *const c_int,
        );
        /// Fortran BLAS sgemv, as PyTorch's `gemv<float>` calls it.
        pub fn sgemv_(
            trans: *const c_char,
            m: *const c_int,
            n: *const c_int,
            alpha: *const f32,
            a: *const f32,
            lda: *const c_int,
            x: *const f32,
            incx: *const c_int,
            beta: *const f32,
            y: *mut f32,
            incy: *const c_int,
        );
        pub fn cblas_sdot(n: c_int, x: *const f32, incx: c_int, y: *const f32, incy: c_int) -> f32;
    }
}

/// `addmm_impl_cpu_` for a fresh row-major result: `beta * self + alpha * (m1 @ m2)`.
/// `init` is `self` already broadcast to the result shape (row-major); it is used only when
/// beta is nonzero, as in PyTorch.
pub fn addmm(
    m1: Mat,
    m2: Mat,
    init: Option<&[f32]>,
    beta: f32,
    alpha: f32,
) -> Result<Vec<f32>, String> {
    let (rows, cols) = (m1.sizes[0], m2.sizes[1]);
    if m1.sizes[1] != m2.sizes[0] {
        return Err("matmul inner dimensions differ".into());
    }
    let mut result = vec![0.0f32; rows * cols];
    if result.is_empty() {
        return Ok(result);
    }
    if m1.sizes[1] == 0 {
        if beta != 0.0
            && let Some(init) = init
        {
            for (r, &v) in result.iter_mut().zip(init) {
                *r = v * beta;
            }
        }
        return Ok(result);
    }
    if beta != 0.0
        && let Some(init) = init
    {
        result.copy_from_slice(init);
    }
    let result_sizes = [rows, cols];
    let result_strides = contiguous_strides(result_sizes);
    let (mut m1, mut m2) = (m1, m2);
    let transpose_c = if result_strides[0] == 1
        && (result_sizes[1] == 1 || result_strides[1] >= result_sizes[0].max(1))
    {
        false
    } else {
        // A row-major result always takes this branch otherwise (strides[1] == 1).
        std::mem::swap(&mut m1, &mut m2);
        true
    };
    let (i0, i1) = if transpose_c { (1, 0) } else { (0, 1) };
    let m = result_sizes[i0];
    let n = result_sizes[i1];
    let k = m1.sizes[i1];

    let pick = |mat: Mat<'_>, lead: usize, other: usize| -> (bool, Option<Vec<f32>>) {
        if mat.strides[i0] == 1 && mat.strides[i1] >= lead.max(1) {
            (false, None)
        } else if mat.strides[i1] == 1 && mat.strides[i0] >= other.max(1) {
            (true, None)
        } else {
            (!transpose_c, Some(mat.to_contiguous()))
        }
    };
    let (ta, a_copy) = pick(m1, m, k);
    let (tb, b_copy) = pick(m2, k, n);
    let strides_of = |mat: Mat<'_>, copied: bool| {
        if copied {
            contiguous_strides(mat.sizes)
        } else {
            mat.strides
        }
    };
    let a_strides = strides_of(m1, a_copy.is_some());
    let b_strides = strides_of(m2, b_copy.is_some());
    let lda = a_strides[if ta == transpose_c { 1 } else { 0 }];
    let ldb = b_strides[if tb == transpose_c { 1 } else { 0 }];
    let ldc = result_strides[if transpose_c { 0 } else { 1 }];
    let mut ld = [lda, ldb, ldc];
    normalize_last_dims(ta, tb, m, n, k, &mut ld);
    let [lda, ldb, ldc] = ld;
    let fits = [m, n, k, lda, ldb, ldc]
        .iter()
        .all(|&v| v <= i32::MAX as usize)
        && lda >= (if ta { k } else { m }).max(1)
        && ldb >= (if tb { n } else { k }).max(1)
        && ldc >= m.max(1);
    if !fits {
        return Err("matrix layout outside PyTorch's BLAS path".into());
    }
    let a_ptr = match &a_copy {
        Some(v) => v.as_ptr(),
        None => m1.data[m1.offset..].as_ptr(),
    };
    let b_ptr = match &b_copy {
        Some(v) => v.as_ptr(),
        None => m2.data[m2.offset..].as_ptr(),
    };
    #[cfg(target_os = "macos")]
    {
        let (ta_c, tb_c) = (
            if ta { b'T' } else { b'N' } as std::ffi::c_char,
            if tb { b'T' } else { b'N' } as std::ffi::c_char,
        );
        let (m_, n_, k_) = (m as i32, n as i32, k as i32);
        let (lda_, ldb_, ldc_) = (lda as i32, ldb as i32, ldc as i32);
        // SAFETY: every dimension and leading dimension fits i32 and satisfies BLAS's bounds
        // (checked above); a and b point into live buffers covering the strided views, the
        // result buffer holds rows * cols floats, and nothing aliases it.
        unsafe {
            accelerate::sgemm_(
                &ta_c,
                &tb_c,
                &m_,
                &n_,
                &k_,
                &alpha,
                a_ptr,
                &lda_,
                b_ptr,
                &ldb_,
                &beta,
                result.as_mut_ptr(),
                &ldc_,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (a_ptr, b_ptr, ta, tb, lda, ldb, ldc);
        return Err("PyTorch's BLAS path is wired for macOS Accelerate only".into());
    }
    Ok(result)
}

/// `mm`: addmm with beta 0 and alpha 1.
pub fn mm(m1: Mat, m2: Mat) -> Result<Vec<f32>, String> {
    addmm(m1, m2, None, 0.0, 1.0)
}

/// `mm_mat1_backward(grad, mat2, mat1)`: grad of mat1 for mat1 @ mat2, as a row-major matrix.
pub fn mm_mat1_backward(grad: Mat, mat2: Mat, mat1: Mat) -> Result<Vec<f32>, String> {
    if mat1.is_column_major() {
        // mat2.mm(grad.t()).t(): computed (k, m), returned transposed.
        let r = mm(mat2, grad.t())?;
        Ok(Mat::rows(&r, mat2.sizes[0], grad.sizes[0])
            .t()
            .to_contiguous())
    } else {
        mm(grad, mat2.t())
    }
}

/// `mm_mat2_backward(grad, mat1, mat2)`: grad of mat2 for mat1 @ mat2, as a row-major matrix.
pub fn mm_mat2_backward(grad: Mat, mat1: Mat, mat2: Mat) -> Result<Vec<f32>, String> {
    if mat2.is_column_major() {
        // grad.t().mm(mat1).t()
        let r = mm(grad.t(), mat1)?;
        Ok(Mat::rows(&r, grad.sizes[1], mat1.sizes[1])
            .t()
            .to_contiguous())
    } else {
        mm(mat1.t(), grad)
    }
}

/// A 3-D strided float32 view (batch, rows, cols).
#[derive(Clone, Copy)]
pub struct Batch<'a> {
    pub data: &'a [f32],
    pub offset: usize,
    pub sizes: [usize; 3],
    pub strides: [usize; 3],
}

impl<'a> Batch<'a> {
    fn select(self, b: usize) -> Mat<'a> {
        Mat {
            data: self.data,
            offset: self.offset + b * self.strides[0],
            sizes: [self.sizes[1], self.sizes[2]],
            strides: [self.strides[1], self.strides[2]],
        }
    }
}

/// `bmm_out_or_baddbmm_` for bmm (no MKL, as in PyTorch's macOS build): a plain loop when
/// k * rows * cols < 400, else one `addmm_impl_cpu_` per batch item. Result is row-major.
pub fn bmm(b1: Batch, b2: Batch) -> Result<Vec<f32>, String> {
    let [bs, rows, k] = b1.sizes;
    let cols = b2.sizes[2];
    if b2.sizes[0] != bs || b2.sizes[1] != k {
        return Err("bmm shapes do not match".into());
    }
    let mut out = vec![0.0f32; bs * rows * cols];
    if out.is_empty() || k == 0 {
        return Ok(out);
    }
    if k * rows * cols < 400 {
        // baddbmm_cpu_kernel as PyTorch's macOS build compiled it (read from libtorch_cpu):
        // `acc += s * m` is fused, except that with k > 3 and both operands contiguous along
        // k the loop is vectorized: the first k - k % 4 products are rounded and added in
        // order, the rest fused.
        let unfused = if k > 3 && b1.strides[2] == 1 && b2.strides[1] == 1 {
            k - k % 4
        } else {
            0
        };
        for b in 0..bs {
            let (s, m) = (b1.select(b), b2.select(b));
            for i in 0..rows {
                for j in 0..cols {
                    let mut acc = 0.0f32;
                    for t in 0..k {
                        let x = s.data[s.offset + i * s.strides[0] + t * s.strides[1]];
                        let y = m.data[m.offset + t * m.strides[0] + j * m.strides[1]];
                        acc = if t < unfused {
                            acc + x * y
                        } else {
                            x.mul_add(y, acc)
                        };
                    }
                    out[(b * rows + i) * cols + j] = acc;
                }
            }
        }
        return Ok(out);
    }
    for b in 0..bs {
        let r = mm(b1.select(b), b2.select(b))?;
        out[b * rows * cols..(b + 1) * rows * cols].copy_from_slice(&r);
    }
    Ok(out)
}

/// A 1-D strided float32 view.
#[derive(Clone, Copy)]
pub struct Vector<'a> {
    pub data: &'a [f32],
    pub offset: usize,
    pub len: usize,
    pub stride: usize,
}

/// `addmv_out_cpu` for a fresh result: `beta * self + alpha * (mat @ vec)` through `sgemv_`.
pub fn addmv(
    mat: Mat,
    vec: Vector,
    init: Option<&[f32]>,
    beta: f32,
    alpha: f32,
) -> Result<Vec<f32>, String> {
    let [rows, cols] = mat.sizes;
    if vec.len != cols {
        return Err("mv size mismatch".into());
    }
    let mut result = vec![0.0f32; rows];
    if rows * cols == 0 {
        if beta != 0.0
            && let Some(init) = init
        {
            for (r, &v) in result.iter_mut().zip(init) {
                *r = v * beta;
            }
        }
        return Ok(result);
    }
    if beta != 0.0
        && let Some(init) = init
    {
        result.copy_from_slice(init);
    }
    let lda_cond = |m: usize, n: usize, lda: usize| n == 1 || lda >= m.max(1);
    let copy;
    let (trans, m, n, a_ptr, mut lda) =
        if mat.strides[0] == 1 && lda_cond(rows, cols, mat.strides[1]) {
            (
                b'n',
                rows,
                cols,
                mat.data[mat.offset..].as_ptr(),
                mat.strides[1],
            )
        } else if mat.strides[1] == 1 && lda_cond(cols, rows, mat.strides[0]) {
            (
                b't',
                cols,
                rows,
                mat.data[mat.offset..].as_ptr(),
                mat.strides[0],
            )
        } else {
            copy = mat.to_contiguous();
            (b't', cols, rows, copy.as_ptr(), cols.max(1))
        };
    // gemv<float>: a single column makes lda = m.
    if n == 1 {
        lda = m;
    }
    let fits = [m, n, lda, vec.stride]
        .iter()
        .all(|&v| v <= i32::MAX as usize)
        && vec.stride > 0;
    if !fits {
        return Err("vector layout outside PyTorch's BLAS path".into());
    }
    #[cfg(target_os = "macos")]
    {
        let t = trans as std::ffi::c_char;
        let (m_, n_, lda_, incx, incy) = (m as i32, n as i32, lda as i32, vec.stride as i32, 1i32);
        // SAFETY: sizes fit i32 (checked); the matrix and vector pointers cover their strided
        // views; the result holds `rows` floats and is not aliased.
        unsafe {
            accelerate::sgemv_(
                &t,
                &m_,
                &n_,
                &alpha,
                a_ptr,
                &lda_,
                vec.data[vec.offset..].as_ptr(),
                &incx,
                &beta,
                result.as_mut_ptr(),
                &incy,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (trans, a_ptr, lda);
        return Err("PyTorch's BLAS path is wired for macOS Accelerate only".into());
    }
    Ok(result)
}

/// `mv`: addmv with beta 0 and alpha 1.
pub fn mv(mat: Mat, vec: Vector) -> Result<Vec<f32>, String> {
    addmv(mat, vec, None, 0.0, 1.0)
}

/// `dot_impl<float>`: BLAS sdot (a length-1 dot uses unit increments).
pub fn dot(x: Vector, y: Vector) -> Result<f32, String> {
    if x.len != y.len {
        return Err("dot size mismatch".into());
    }
    if x.len == 0 {
        return Ok(0.0);
    }
    let (incx, incy) = if x.len == 1 {
        (1, 1)
    } else {
        (x.stride, y.stride)
    };
    if [x.len, incx, incy].iter().any(|&v| v > i32::MAX as usize) {
        return Err("dot layout outside PyTorch's BLAS path".into());
    }
    #[cfg(target_os = "macos")]
    // SAFETY: both pointers cover `len` elements at their increments; sizes fit i32.
    let value = unsafe {
        accelerate::cblas_sdot(
            x.len as i32,
            x.data[x.offset..].as_ptr(),
            incx as i32,
            y.data[y.offset..].as_ptr(),
            incy as i32,
        )
    };
    #[cfg(not(target_os = "macos"))]
    let value = return Err("PyTorch's BLAS path is wired for macOS Accelerate only".into());
    Ok(value)
}

/// `cpublas::gemm` for float32 with explicit leading dimensions (column-major, as BLAS):
/// `c = alpha * op(a) * op(b) + beta * c`. `a`, `b` and `c` start at the matrices' first
/// elements. Used by kernels PyTorch writes directly against cpublas (flash attention).
#[allow(clippy::too_many_arguments)]
pub fn gemm_raw(
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) -> Result<(), String> {
    let mut ld = [lda, ldb, ldc];
    normalize_last_dims(ta, tb, m, n, k, &mut ld);
    let [lda, ldb, ldc] = ld;
    let fits = [m, n, k, lda, ldb, ldc]
        .iter()
        .all(|&v| v <= i32::MAX as usize)
        && lda >= (if ta { k } else { m }).max(1)
        && ldb >= (if tb { n } else { k }).max(1)
        && ldc >= m.max(1);
    if !fits {
        return Err("gemm layout outside PyTorch's BLAS path".into());
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        let (ta_c, tb_c) = (
            if ta { b'T' } else { b'N' } as std::ffi::c_char,
            if tb { b'T' } else { b'N' } as std::ffi::c_char,
        );
        let (m_, n_, k_) = (m as i32, n as i32, k as i32);
        let (lda_, ldb_, ldc_) = (lda as i32, ldb as i32, ldc as i32);
        // SAFETY: dimensions and leading dimensions fit i32 and satisfy BLAS's bounds; the
        // slices start at the matrices and the caller sized them for these strides.
        unsafe {
            accelerate::sgemm_(
                &ta_c,
                &tb_c,
                &m_,
                &n_,
                &k_,
                &alpha,
                a.as_ptr(),
                &lda_,
                b.as_ptr(),
                &ldb_,
                &beta,
                c.as_mut_ptr(),
                &ldc_,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (a, b, c, alpha, beta);
        return Err("PyTorch's BLAS path is wired for macOS Accelerate only".into());
    }
    Ok(())
}
