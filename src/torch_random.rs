//! PyTorch's CPU random generator and sampling transforms, bit for bit.
//!
//! Ported from PyTorch 2.14.1 (git 5c48869): `mt19937` (aten/src/ATen/core/MT19937RNGEngine.h),
//! `CPUGeneratorImpl` (random, random64, the cached double normal), the distributions in
//! core/DistributionsHelper.h and core/TransformationHelper.h, the CPU kernels in
//! native/cpu/DistributionTemplates.h (`normal_fill` with its 16-wide Box-Muller) and
//! `randperm_cpu` (native/TensorFactories.cpp). PyTorch is BSD-3; see THIRD_PARTY_NOTICES.md.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UMASK: u32 = 0x8000_0000;
const LMASK: u32 = 0x7fff_ffff;

/// `default_rng_seed_val`: the seed of the default generator and of a new `torch.Generator()`.
pub const DEFAULT_SEED: u64 = 67_280_421_310_721;

#[derive(Clone)]
pub struct Generator {
    seed: u64,
    state: [u32; N],
    left: usize,
    next: usize,
    next_double_normal: Option<f64>,
    next_float_normal: Option<f32>,
}

impl Generator {
    pub fn new(seed: u64) -> Self {
        let mut g = Generator {
            seed: 0,
            state: [0; N],
            left: 1,
            next: 0,
            next_double_normal: None,
            next_float_normal: None,
        };
        g.manual_seed(seed);
        g
    }

    /// `set_current_seed`: a fresh engine and no cached normal samples.
    pub fn manual_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.state[0] = seed as u32;
        for j in 1..N {
            let prev = self.state[j - 1];
            self.state[j] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(j as u32);
        }
        self.left = 1;
        self.next = 0;
        self.next_double_normal = None;
        self.next_float_normal = None;
    }

    pub fn initial_seed(&self) -> u64 {
        self.seed
    }

    fn twist(u: u32, v: u32) -> u32 {
        (((u & UMASK) | (v & LMASK)) >> 1) ^ if v & 1 == 1 { MATRIX_A } else { 0 }
    }

    fn next_state(&mut self) {
        let s = &mut self.state;
        self.left = N;
        self.next = 0;
        for j in 0..N - M {
            s[j] = s[j + M] ^ Self::twist(s[j], s[j + 1]);
        }
        for j in N - M..N - 1 {
            s[j] = s[j + M - N] ^ Self::twist(s[j], s[j + 1]);
        }
        s[N - 1] = s[M - 1] ^ Self::twist(s[N - 1], s[0]);
    }

    pub fn random(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub fn random64(&mut self) -> u64 {
        let hi = self.random() as u64;
        let lo = self.random() as u64;
        (hi << 32) | lo
    }

    /// `uniform_real_distribution<float>`: 24 random bits, then fused x * (to - from) + from.
    fn uniform_f32(&mut self, from: f32, to: f32) -> f32 {
        let x = (self.random() & ((1 << 24) - 1)) as f32 * (1.0 / (1u32 << 24) as f32);
        x.mul_add(to - from, from)
    }

    /// `uniform_real_distribution<double>`: 53 random bits from random64.
    fn uniform_f64(&mut self, from: f64, to: f64) -> f64 {
        let x = (self.random64() & ((1 << 53) - 1)) as f64 * (1.0 / (1u64 << 53) as f64);
        x.mul_add(to - from, from)
    }

    /// `normal_distribution<double>` with the generator's cached second sample.
    fn normal_f64(&mut self, mean: f64, std: f64) -> f64 {
        if let Some(v) = self.next_double_normal.take() {
            return v.mul_add(std, mean);
        }
        let u1 = self.uniform_f64(0.0, 1.0);
        let u2 = self.uniform_f64(0.0, 1.0);
        let r = (-2.0 * (-u2).ln_1p()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u1;
        self.next_double_normal = Some(r * theta.sin());
        (r * theta.cos()).mul_add(std, mean)
    }

    /// `uniform_kernel` for float32: a draw equal to `to` becomes `from`.
    pub fn uniform(&mut self, n: usize, from: f64, to: f64) -> Vec<f32> {
        let (f, t) = (from as f32, to as f32);
        (0..n)
            .map(|_| {
                let v = self.uniform_f32(f, t);
                if v == t { f } else { v }
            })
            .collect()
    }

    /// `uniform_kernel` for float64.
    pub fn uniform_double(&mut self, n: usize, from: f64, to: f64) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let v = self.uniform_f64(from, to);
                if v == to { from } else { v }
            })
            .collect()
    }

    /// `normal_kernel` for a contiguous float32 tensor of `n` elements.
    pub fn normal(&mut self, n: usize, mean: f64, std: f64) -> Vec<f32> {
        if n < 16 {
            return (0..n).map(|_| self.normal_f64(mean, std) as f32).collect();
        }
        let (mean, std) = (mean as f32, std as f32);
        let fill16 = |d: &mut [f32]| {
            for j in 0..8 {
                let u1 = 1.0 - d[j];
                let u2 = d[j + 8];
                let radius = (-2.0 * u1.ln()).sqrt();
                let theta = (2.0f32 as f64 * std::f64::consts::PI * u2 as f64) as f32;
                d[j] = (radius * theta.cos()).mul_add(std, mean);
                d[j + 8] = (radius * theta.sin()).mul_add(std, mean);
            }
        };
        let mut data: Vec<f32> = (0..n).map(|_| self.uniform_f32(0.0, 1.0)).collect();
        let mut i = 0;
        while i + 16 <= n {
            fill16(&mut data[i..i + 16]);
            i += 16;
        }
        if !n.is_multiple_of(16) {
            let tail = &mut data[n - 16..];
            for x in tail.iter_mut() {
                *x = self.uniform_f32(0.0, 1.0);
            }
            fill16(tail);
        }
        data
    }

    /// `normal_kernel` for a contiguous float64 tensor.
    pub fn normal_double(&mut self, n: usize, mean: f64, std: f64) -> Vec<f64> {
        if n < 16 {
            return (0..n).map(|_| self.normal_f64(mean, std)).collect();
        }
        let fill16 = |d: &mut [f64]| {
            for j in 0..8 {
                let u1 = 1.0 - d[j];
                let u2 = d[j + 8];
                let radius = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0f32 as f64 * std::f64::consts::PI * u2;
                d[j] = (radius * theta.cos()).mul_add(std, mean);
                d[j + 8] = (radius * theta.sin()).mul_add(std, mean);
            }
        };
        let mut data: Vec<f64> = (0..n).map(|_| self.uniform_f64(0.0, 1.0)).collect();
        let mut i = 0;
        while i + 16 <= n {
            fill16(&mut data[i..i + 16]);
            i += 16;
        }
        if !n.is_multiple_of(16) {
            let tail = &mut data[n - 16..];
            for x in tail.iter_mut() {
                *x = self.uniform_f64(0.0, 1.0);
            }
            fill16(tail);
        }
        data
    }

    /// `random_from_to_kernel` (int64): `(r % range) + low`, with random64 once range >= 2^28.
    pub fn randint(&mut self, n: usize, low: i64, high: i64) -> Vec<i64> {
        let range = high.wrapping_sub(low) as u64;
        (0..n)
            .map(|_| {
                let r = if range >= 1 << 28 {
                    self.random64()
                } else {
                    self.random() as u64
                };
                ((r % range) as i64).wrapping_add(low)
            })
            .collect()
    }

    /// `randperm_cpu`.
    pub fn randperm(&mut self, n: usize) -> Vec<i64> {
        let mut r: Vec<i64> = (0..n as i64).collect();
        if (n as u64) < u32::MAX as u64 / 20 {
            for i in 0..n.saturating_sub(1) {
                let z = self.random() as usize % (n - i);
                r.swap(i, z + i);
            }
        } else {
            for i in 0..n {
                let z = (self.random64() % (i as u64 + 1)) as usize;
                r[i] = r[z];
                r[z] = i as i64;
            }
        }
        r
    }

    /// `bernoulli_kernel` with a scalar p: a double uniform below p gives 1.
    pub fn bernoulli(&mut self, n: usize, p: f64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                if self.uniform_f64(0.0, 1.0) < p {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// `exponential_kernel`, `geometric_kernel`, `cauchy_kernel`, `log_normal_kernel`: each
    /// draws in double (`*_distribution<double>`) and the caller casts to the tensor dtype.
    pub fn sample(&mut self, kind: &str, n: usize, a: f64, b: f64) -> Result<Vec<f64>, String> {
        let pi = std::f64::consts::PI;
        let draw: Box<dyn Fn(&mut Self) -> f64> = match kind {
            "exponential" => Box::new(move |g| (-1.0 / a) * (-g.uniform_f64(0.0, 1.0)).ln_1p()),
            "geometric" => Box::new(move |g| (g.uniform_f64(0.0, 1.0).ln() / (-a).ln_1p()).ceil()),
            "cauchy" => {
                Box::new(move |g| b.mul_add((pi * (g.uniform_f64(0.0, 1.0) - 0.5)).tan(), a))
            }
            "log_normal" => Box::new(move |g| g.normal_f64(a, b).exp()),
            _ => return Err(format!("unknown distribution {kind}")),
        };
        Ok((0..n).map(|_| draw(self)).collect())
    }

    /// Uniform doubles in [0, 1), as `uniform_real_distribution<double>(0, 1)` draws them.
    pub fn uniform_doubles(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform_f64(0.0, 1.0)).collect()
    }

    /// Engine state for get_state/set_state (DenseCore's own layout, not torch's bytes).
    pub fn to_words(&self) -> Vec<u64> {
        let mut w = vec![
            self.seed,
            self.left as u64,
            self.next as u64,
            self.next_double_normal.map_or(u64::MAX, f64::to_bits),
            self.next_double_normal.is_some() as u64,
            self.next_float_normal.map_or(0, |v| v.to_bits() as u64),
            self.next_float_normal.is_some() as u64,
        ];
        w.extend(self.state.iter().map(|&s| s as u64));
        w
    }

    pub fn from_words(w: &[u64]) -> Result<Self, String> {
        if w.len() != 7 + N || w[1] == 0 || w[1] > N as u64 || w[2] > N as u64 {
            return Err("invalid generator state".into());
        }
        let mut state = [0u32; N];
        for (s, &v) in state.iter_mut().zip(&w[7..]) {
            *s = u32::try_from(v).map_err(|_| "invalid generator state")?;
        }
        Ok(Generator {
            seed: w[0],
            state,
            left: w[1] as usize,
            next: w[2] as usize,
            next_double_normal: (w[4] == 1).then(|| f64::from_bits(w[3])),
            next_float_normal: (w[6] == 1).then(|| f32::from_bits(w[5] as u32)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_mt19937() {
        // std::mt19937 with the default seed 5489: the 10000th output is 4123659995.
        let mut g = Generator::new(5489);
        let mut last = 0;
        for _ in 0..10_000 {
            last = g.random();
        }
        assert_eq!(last, 4_123_659_995);
    }
}
