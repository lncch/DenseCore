//! Checked synchronous float32 CPU matrix multiplication.
pub fn backend() -> &'static str {
    if cfg!(target_os = "macos") {
        "accelerate"
    } else {
        "reference"
    }
}
pub fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, String> {
    gemm_transpose(a, b, m, k, n, false, false)
}
pub fn gemm_transpose(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    ta: bool,
    tb: bool,
) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    gemm_into(a, b, m, k, n, ta, tb, None, &mut out)?;
    Ok(out)
}
#[allow(clippy::too_many_arguments)]
pub fn gemm_into(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    ta: bool,
    tb: bool,
    bias: Option<&[f32]>,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    let alen = m.checked_mul(k).ok_or("matrix input size overflow")?;
    let blen = k.checked_mul(n).ok_or("matrix input size overflow")?;
    let size = m.checked_mul(n).ok_or("matrix output size overflow")?;
    if a.len() != alen || b.len() != blen {
        return Err("matrix input length mismatch".into());
    }
    if bias.is_some_and(|v| v.len() != n) {
        return Err("matrix bias length mismatch".into());
    }
    if size == 0 {
        out.clear();
        return Ok(());
    }
    for dim in [m, k, n] {
        i32::try_from(dim).map_err(|_| "nonempty matrix dimensions must fit int32")?;
    }
    let bytes = size
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("matrix allocation size overflow")?;
    if bytes > isize::MAX as usize {
        return Err("matrix allocation exceeds addressable size".into());
    }
    out.resize(size, 0.0);
    if let Some(bias) = bias {
        for row in out.chunks_exact_mut(n) {
            row.copy_from_slice(bias);
        }
    } else if k == 0 || !cfg!(target_os = "macos") {
        // Nonempty Accelerate GEMM uses beta=0 and overwrites initialized output.
        // Empty products and the accumulating reference kernel still need zeros.
        out.fill(0.0);
    }
    if k == 0 {
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: checked lengths/products cover physical transposed layouts.
        // All dimensions/strides fit C int, contiguous borrowed inputs remain
        // alive synchronously, and safe Rust excludes aliasing with output.
        unsafe {
            accelerate::cblas_sgemm(
                101,
                if ta { 112 } else { 111 },
                if tb { 112 } else { 111 },
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                if ta { m } else { k } as i32,
                b.as_ptr(),
                if tb { k } else { n } as i32,
                if bias.is_some() { 1.0 } else { 0.0 },
                out.as_mut_ptr(),
                n as i32,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for t in 0..k {
                sum += a[if ta { t * m + i } else { i * k + t }]
                    * b[if tb { j * k + t } else { t * n + j }];
            }
            out[i * n + j] += sum;
        }
    }
    Ok(())
}
/// Add op(a) * op(b) into an existing m*n row-major destination (BLAS beta=1).
#[allow(clippy::too_many_arguments)]
pub fn gemm_accumulate(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    ta: bool,
    tb: bool,
    out: &mut [f32],
) -> Result<(), String> {
    let alen = m.checked_mul(k).ok_or("matrix input size overflow")?;
    let blen = k.checked_mul(n).ok_or("matrix input size overflow")?;
    let size = m.checked_mul(n).ok_or("matrix output size overflow")?;
    if a.len() != alen || b.len() != blen || out.len() != size {
        return Err("matrix input length mismatch".into());
    }
    if size == 0 || k == 0 {
        return Ok(());
    }
    for dim in [m, k, n] {
        i32::try_from(dim).map_err(|_| "nonempty matrix dimensions must fit int32")?;
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: lengths were checked against the logical layouts above, all
        // dimensions fit C int, and the borrow checker excludes output aliasing.
        unsafe {
            accelerate::cblas_sgemm(
                101,
                if ta { 112 } else { 111 },
                if tb { 112 } else { 111 },
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                if ta { m } else { k } as i32,
                b.as_ptr(),
                if tb { k } else { n } as i32,
                1.0,
                out.as_mut_ptr(),
                n as i32,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for t in 0..k {
                sum += a[if ta { t * m + i } else { i * k + t }]
                    * b[if tb { j * k + t } else { t * n + j }];
            }
            out[i * n + j] += sum;
        }
    }
    Ok(())
}
#[cfg(target_os = "macos")]
mod accelerate {
    use std::ffi::c_int;
    // SDK legacy LP32 ABI, available at the wheel's macOS11 deployment target.
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_and_shape_validation() {
        let a = [1., 2., 3., 4., 5., 6.];
        let b = [1., 2., 3., 4., 5., 6.];
        assert_eq!(gemm(&a, &b, 2, 3, 2).unwrap(), vec![22., 28., 49., 64.]);
        assert!(gemm(&a, &b, 2, 4, 2).is_err());
        assert!(gemm(&[], &[], usize::MAX, 2, 0).is_err());
        assert!(gemm(&[], &[], i32::MAX as usize + 1, 0, 1).is_err());
    }
    #[test]
    fn zero_dimensions_never_call_blas() {
        assert_eq!(gemm(&[], &[], 2, 0, 3).unwrap(), vec![0.; 6]);
        assert!(gemm(&[], &[], 0, usize::MAX, 0).unwrap().is_empty());
    }
    #[test]
    fn transposed_layouts_bias_and_buffer_reuse() {
        let a = [1., 2., 3., 4., 5., 6.];
        let b_transposed = [1., 3., 5., 2., 4., 6.];
        let mut out = vec![99.; 4];
        gemm_into(
            &a,
            &b_transposed,
            2,
            3,
            2,
            false,
            true,
            Some(&[1., 2.]),
            &mut out,
        )
        .unwrap();
        assert_eq!(out, vec![23., 30., 50., 66.]);
        gemm_into(&a, &b_transposed, 2, 3, 2, false, true, None, &mut out).unwrap();
        assert_eq!(out, vec![22., 28., 49., 64.]);
        let a_transposed = [1., 4., 2., 5., 3., 6.];
        assert_eq!(
            gemm_transpose(&a_transposed, &b_transposed, 2, 3, 2, true, true).unwrap(),
            vec![22., 28., 49., 64.]
        );
    }
    #[test]
    fn accumulate_adds_into_existing_values_for_all_layouts() {
        for ta in [false, true] {
            for tb in [false, true] {
                for k in [0, 5] {
                    let (m, n) = (3, 7);
                    let a: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 - 2.).collect();
                    let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 - 3.).collect();
                    let start: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.5).collect();
                    let mut expected = start.clone();
                    for i in 0..m {
                        for j in 0..n {
                            for t in 0..k {
                                expected[i * n + j] += a[if ta { t * m + i } else { i * k + t }]
                                    * b[if tb { j * k + t } else { t * n + j }];
                            }
                        }
                    }
                    let mut out = start.clone();
                    gemm_accumulate(&a, &b, m, k, n, ta, tb, &mut out).unwrap();
                    assert_eq!(out, expected);
                    gemm_accumulate(&a, &b, m, k, n, ta, tb, &mut out).unwrap();
                    let twice: Vec<f32> = expected
                        .iter()
                        .zip(&start)
                        .map(|(e, s)| 2. * e - s)
                        .collect();
                    assert_eq!(out, twice);
                }
            }
        }
        let mut short = vec![0.; 3];
        assert!(gemm_accumulate(&[1.; 6], &[1.; 6], 2, 3, 2, false, false, &mut short).is_err());
        assert!(gemm_accumulate(&[], &[], 0, 4, 0, false, false, &mut []).is_ok());
    }
    #[test]
    fn reused_nan_buffers_are_overwritten_for_all_layouts_and_empty_products() {
        for ta in [false, true] {
            for tb in [false, true] {
                for k in [0, 5] {
                    for bias in [false, true] {
                        let (m, n) = (3, 7);
                        let a: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 - 2.).collect();
                        let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 - 3.).collect();
                        let bias_values = vec![0.25; n];
                        let mut expected = vec![if bias { 0.25 } else { 0. }; m * n];
                        for i in 0..m {
                            for j in 0..n {
                                for t in 0..k {
                                    expected[i * n + j] += a
                                        [if ta { t * m + i } else { i * k + t }]
                                        * b[if tb { j * k + t } else { t * n + j }];
                                }
                            }
                        }
                        let mut out = vec![f32::NAN; m * n];
                        for length in [m * n, 1, m * n + 4] {
                            out.resize(length, f32::NAN);
                            out.fill(f32::NAN);
                            gemm_into(
                                &a,
                                &b,
                                m,
                                k,
                                n,
                                ta,
                                tb,
                                bias.then_some(bias_values.as_slice()),
                                &mut out,
                            )
                            .unwrap();
                            assert_eq!(out, expected);
                        }
                    }
                }
            }
        }
    }
}
