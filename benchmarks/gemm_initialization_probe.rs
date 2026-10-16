#[allow(dead_code)]
mod baseline {
    include!("baseline.rs");
}
#[allow(dead_code)]
mod candidate {
    include!("candidate.rs");
}
use std::{hint::black_box, time::Instant};
fn reference(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    ta: bool,
    tb: bool,
    bias: bool,
) -> Vec<f32> {
    let mut out = vec![0.; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = if bias { 0.125 } else { 0. };
            for t in 0..k {
                s += a[if ta { t * m + i } else { i * k + t }]
                    * b[if tb { j * k + t } else { t * n + j }];
            }
            out[i * n + j] = s;
        }
    }
    out
}
fn run(label: &str, m: usize, k: usize, n: usize, ta: bool, tb: bool, bias: bool, timing: bool) {
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 17) as f32 - 8.) * 0.015625)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i % 19) as f32 - 9.) * 0.015625)
        .collect();
    let bias_values = vec![0.125; n];
    let bopt = if bias {
        Some(bias_values.as_slice())
    } else {
        None
    };
    let mut out = vec![f32::NAN; m * n];
    let expected = reference(&a, &b, m, k, n, ta, tb, bias);
    let invoke = |out: &mut Vec<f32>| {
        if label == "baseline" {
            baseline::gemm_into(&a, &b, m, k, n, ta, tb, bopt, out)
        } else {
            candidate::gemm_into(&a, &b, m, k, n, ta, tb, bopt, out)
        }
    };
    invoke(&mut out).unwrap();
    assert_eq!(out.len(), expected.len());
    for (&x, &y) in out.iter().zip(&expected) {
        assert!(
            x.is_finite() && (x - y).abs() <= 1e-5 + 1e-4 * y.abs(),
            "numerical gate failed"
        );
    }
    // Shrink, grow, NaN poison and repeat before any timing.
    out.resize(1, f32::NAN);
    out.fill(f32::NAN);
    invoke(&mut out).unwrap();
    for (&x, &y) in out.iter().zip(&expected) {
        assert!(x.is_finite() && (x - y).abs() <= 1e-5 + 1e-4 * y.abs());
    }
    if !timing {
        return;
    }
    for _ in 0..30 {
        invoke(&mut out).unwrap();
        black_box(&out);
    }
    let start = Instant::now();
    for _ in 0..500 {
        invoke(black_box(&mut out)).unwrap();
        black_box(&out);
    }
    println!(
        "{{\"label\":\"{}\",\"m\":{},\"k\":{},\"n\":{},\"ta\":{},\"tb\":{},\"bias\":{},\"us\":{}}}",
        label,
        m,
        k,
        n,
        ta,
        tb,
        bias,
        start.elapsed().as_secs_f64() * 1e6 / 500.
    );
}
fn main() {
    // All correctness gates before any timed loop.
    for label in ["baseline", "candidate"] {
        for ta in [false, true] {
            for tb in [false, true] {
                for bias in [false, true] {
                    for (m, k, n) in [(3, 5, 7), (3, 0, 7), (0, 5, 7), (3, 5, 0)] {
                        run(label, m, k, n, ta, tb, bias, false);
                    }
                }
            }
        }
    }
    eprintln!("64 correctness gates passed");
    for repeat in 0..5 {
        for (m, k, n, ta, tb) in [
            (1024, 32, 256, true, false),
            (32, 256, 1024, false, true),
            (256, 32, 1024, true, false),
        ] {
            for bias in [false, true] {
                for label in if repeat % 2 == 0 {
                    ["baseline", "candidate"]
                } else {
                    ["candidate", "baseline"]
                } {
                    run(label, m, k, n, ta, tb, bias, true);
                }
            }
        }
    }
}
