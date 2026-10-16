//! SLEEF's single-precision functions as PyTorch calls them on aarch64 (`Sleef_*f4_u10`),
//! ported so each result has the same bits.
//!
//! Ported from SLEEF (https://github.com/shibatch/sleef) at commit 5a1d179, the version
//! PyTorch 2.14.1 pins: `src/libm/sleefsimdsp.c`, `src/common/df.h` and the AdvSIMD helpers
//! (`src/arch/helperadvsimd.h`: CONFIG 1, ENABLE_FMA_SP, FULL_FP_ROUNDING). SLEEF is
//! Copyright Naoki Shibata and contributors, Boost Software License 1.0; see
//! THIRD_PARTY_NOTICES.md.
//!
//! The code runs on 4-lane groups (`V`), as the NEON build does: some functions pick their
//! argument reduction once per group, so a lane's bits can depend on its neighbours.
//! `map4` applies a function to an array the way PyTorch's loop does (groups of 4 from the
//! start, the last group padded with zeros).
//!
//! SLEEF writes most coefficients as C double literals that are converted to float; they are
//! written here as `f64 as f32` (two roundings, as in C). `f` suffixed C constants are f32.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy)]
pub struct V(pub [f32; 4]);
#[derive(Clone, Copy)]
struct I([i32; 4]);
#[derive(Clone, Copy)]
struct M([bool; 4]);
#[derive(Clone, Copy)]
struct V2 {
    x: V,
    y: V,
}

#[inline(always)]
fn c(v: f32) -> V {
    // Written out: `[v; 4]` becomes a memset_pattern16 call that keeps lanes out of registers.
    V([v, v, v, v])
}
#[inline(always)]
fn ci(v: i32) -> I {
    I([v, v, v, v])
}
#[inline(always)]
fn d2f(v: f64) -> f32 {
    v as f32
}

impl V {
    #[inline(always)]
    fn map(self, f: impl Fn(f32) -> f32) -> V {
        V(self.0.map(f))
    }
    #[inline(always)]
    fn zip(self, o: V, f: impl Fn(f32, f32) -> f32) -> V {
        V(std::array::from_fn(|i| f(self.0[i], o.0[i])))
    }
    #[inline(always)]
    fn bits(self) -> I {
        I(self.0.map(|x| x.to_bits() as i32))
    }
    #[inline(always)]
    fn abs(self) -> V {
        self.map(f32::abs)
    }
    #[inline(always)]
    fn div(self, o: V) -> V {
        self.zip(o, |a, b| a / b)
    }
    #[inline(always)]
    fn rec(self) -> V {
        c(1.0).div(self)
    }
    #[inline(always)]
    fn sqrt(self) -> V {
        self.map(f32::sqrt)
    }
    /// `vrndnq_f32`: round half to even.
    #[inline(always)]
    fn rint(self) -> V {
        self.map(f32::round_ties_even)
    }
    /// `vcvtq_s32_f32(vrndnq_f32(d))`.
    #[inline(always)]
    fn rint_i(self) -> I {
        I(self.0.map(|x| x.round_ties_even() as i32))
    }
    /// `vcvtq_s32_f32`: truncate, saturating, NaN to 0.
    #[inline(always)]
    fn trunc_i(self) -> I {
        I(self.0.map(|x| x as i32))
    }
    #[inline(always)]
    fn lt(self, o: V) -> M {
        M(std::array::from_fn(|i| self.0[i] < o.0[i]))
    }
    #[inline(always)]
    fn le(self, o: V) -> M {
        M(std::array::from_fn(|i| self.0[i] <= o.0[i]))
    }
    #[inline(always)]
    fn gt(self, o: V) -> M {
        o.lt(self)
    }
    #[inline(always)]
    fn ge(self, o: V) -> M {
        o.le(self)
    }
    #[inline(always)]
    fn eq(self, o: V) -> M {
        M(std::array::from_fn(|i| self.0[i] == o.0[i]))
    }
    #[inline(always)]
    fn isnan(self) -> M {
        M(self.0.map(f32::is_nan))
    }
    #[inline(always)]
    fn isinf(self) -> M {
        self.abs().eq(c(f32::INFINITY))
    }
    #[inline(always)]
    fn ispinf(self) -> M {
        self.eq(c(f32::INFINITY))
    }
    #[inline(always)]
    fn isnegzero(self) -> M {
        M(self.0.map(|x| x.to_bits() == (-0.0f32).to_bits()))
    }
    #[inline(always)]
    fn signbit(self) -> M {
        M(self.0.map(|x| x.is_sign_negative()))
    }
    /// `vmulsign_vf_vf_vf`: x with its sign flipped where y is negative.
    #[inline(always)]
    fn mulsign(self, y: V) -> V {
        self.zip(y, |x, y| {
            f32::from_bits(x.to_bits() ^ (y.to_bits() & 0x8000_0000))
        })
    }
    /// x with every bit set (NaN) where m holds: `vor_vm_vo32_vm`.
    #[inline(always)]
    fn or_mask(self, m: M) -> V {
        V(std::array::from_fn(|i| {
            if m.0[i] {
                f32::from_bits(u32::MAX)
            } else {
                self.0[i]
            }
        }))
    }
    /// x with all bits cleared (+0.0) where m holds: `vandnot_vm_vo32_vm`.
    #[inline(always)]
    fn andnot_mask(self, m: M) -> V {
        V(std::array::from_fn(
            |i| if m.0[i] { 0.0 } else { self.0[i] },
        ))
    }
    /// x kept where m holds, +0.0 elsewhere: `vand_vm_vo32_vm`.
    #[inline(always)]
    fn and_mask(self, m: M) -> V {
        V(std::array::from_fn(
            |i| if m.0[i] { self.0[i] } else { 0.0 },
        ))
    }
    /// Flip the sign where m holds.
    #[inline(always)]
    fn xor_sign(self, m: M) -> V {
        V(std::array::from_fn(|i| {
            if m.0[i] {
                f32::from_bits(self.0[i].to_bits() ^ 0x8000_0000)
            } else {
                self.0[i]
            }
        }))
    }
}

impl Add for V {
    type Output = V;
    #[inline(always)]
    fn add(self, o: V) -> V {
        self.zip(o, |a, b| a + b)
    }
}
impl Sub for V {
    type Output = V;
    #[inline(always)]
    fn sub(self, o: V) -> V {
        self.zip(o, |a, b| a - b)
    }
}
impl Mul for V {
    type Output = V;
    #[inline(always)]
    fn mul(self, o: V) -> V {
        self.zip(o, |a, b| a * b)
    }
}
impl Neg for V {
    type Output = V;
    #[inline(always)]
    fn neg(self) -> V {
        self.map(|a| -a)
    }
}

/// `vmla`/`vfma`: z + x * y, fused.
#[inline(always)]
fn mla(x: V, y: V, z: V) -> V {
    V(std::array::from_fn(|i| x.0[i].mul_add(y.0[i], z.0[i])))
}
/// `vfmanp`: z - x * y, fused.
#[inline(always)]
fn fmanp(x: V, y: V, z: V) -> V {
    V(std::array::from_fn(|i| (-x.0[i]).mul_add(y.0[i], z.0[i])))
}
/// `vfmapn`: x * y - z, fused.
#[inline(always)]
fn fmapn(x: V, y: V, z: V) -> V {
    mla(x, y, -z)
}
#[inline(always)]
fn sel(m: M, x: V, y: V) -> V {
    V(std::array::from_fn(
        |i| if m.0[i] { x.0[i] } else { y.0[i] },
    ))
}
#[inline(always)]
fn seli(m: M, x: I, y: I) -> I {
    I(std::array::from_fn(
        |i| if m.0[i] { x.0[i] } else { y.0[i] },
    ))
}
#[inline(always)]
fn sel2(m: M, x: V2, y: V2) -> V2 {
    V2 {
        x: sel(m, x.x, y.x),
        y: sel(m, x.y, y.y),
    }
}

impl M {
    #[inline(always)]
    fn or(self, o: M) -> M {
        M(std::array::from_fn(|i| self.0[i] || o.0[i]))
    }
    #[inline(always)]
    fn andnot(self, o: M) -> M {
        // vandnot_vo_vo_vo(x, y): y & !x
        M(std::array::from_fn(|i| !self.0[i] && o.0[i]))
    }
    #[inline(always)]
    fn all(self) -> bool {
        self.0.iter().all(|&b| b)
    }
}

impl I {
    #[inline(always)]
    fn map(self, f: impl Fn(i32) -> i32) -> I {
        I(self.0.map(f))
    }
    #[inline(always)]
    fn zip(self, o: I, f: impl Fn(i32, i32) -> i32) -> I {
        I(std::array::from_fn(|i| f(self.0[i], o.0[i])))
    }
    #[inline(always)]
    fn add(self, o: I) -> I {
        self.zip(o, i32::wrapping_add)
    }
    #[inline(always)]
    fn sub(self, o: I) -> I {
        self.zip(o, i32::wrapping_sub)
    }
    #[inline(always)]
    fn and(self, o: I) -> I {
        self.zip(o, |a, b| a & b)
    }
    #[inline(always)]
    fn neg(self) -> I {
        self.map(i32::wrapping_neg)
    }
    #[inline(always)]
    fn sra(self, n: u32) -> I {
        self.map(|a| a >> n)
    }
    #[inline(always)]
    fn eq(self, o: I) -> M {
        M(std::array::from_fn(|i| self.0[i] == o.0[i]))
    }
    #[inline(always)]
    fn gt(self, o: I) -> M {
        M(std::array::from_fn(|i| self.0[i] > o.0[i]))
    }
    #[inline(always)]
    fn to_f(self) -> V {
        V(self.0.map(|q| q as f32))
    }
    /// `vand_vi2_vo_vi2`: x where m holds, 0 elsewhere.
    #[inline(always)]
    fn and_mask(self, m: M) -> I {
        I(std::array::from_fn(|i| if m.0[i] { self.0[i] } else { 0 }))
    }
    /// `vandnot_vi2_vi2_vi2(x, y)`: y & !x.
    #[inline(always)]
    fn andnot(self, y: I) -> I {
        self.zip(y, |x, y| !x & y)
    }
}

#[inline(always)]
fn from_bits(q: I) -> V {
    V(q.0.map(|b| f32::from_bits(b as u32)))
}
/// `vpow2i_vf_vi2`.
#[inline(always)]
fn pow2i(q: I) -> V {
    from_bits(q.map(|q| q.wrapping_add(0x7f) << 23))
}
/// `vldexp2_vf_vf_vi2`.
#[inline(always)]
fn ldexp2(d: V, e: I) -> V {
    d * pow2i(e.sra(1)) * pow2i(e.sub(e.sra(1)))
}
/// `vldexp3_vf_vf_vi2`: add q to the exponent bits.
#[inline(always)]
fn ldexp3(d: V, q: I) -> V {
    from_bits(d.bits().add(q.map(|q| q << 23)))
}
/// `vilogbk_vi2_vf`.
#[inline(always)]
fn ilogbk(d: V) -> I {
    let o = d.lt(c(5.421_010_862_427_522e-20_f32));
    let d = sel(o, c(1.844_674_407_370_955_2e19_f32) * d, d);
    let q = d.bits().map(|b| ((b as u32 >> 23) & 0xff) as i32);
    q.sub(seli(o, ci(64 + 0x7f), ci(0x7f)))
}
/// `vilogb2k_vi2_vf`.
#[inline(always)]
fn ilogb2k(d: V) -> I {
    d.bits().map(|b| ((b as u32 >> 23) & 0xff) as i32 - 0x7f)
}

// ---- double-float arithmetic (df.h, FMA variants) -----------------------------------------

#[inline(always)]
fn v2(x: V, y: V) -> V2 {
    V2 { x, y }
}
#[inline(always)]
fn v2c(h: f32, l: f32) -> V2 {
    v2(c(h), c(l))
}
impl V2 {
    #[inline(always)]
    fn sum(self) -> V {
        self.x + self.y
    }
    #[inline(always)]
    fn neg(self) -> V2 {
        v2(-self.x, -self.y)
    }
    #[inline(always)]
    fn normalize(self) -> V2 {
        let s = self.x + self.y;
        v2(s, (self.x - s) + self.y)
    }
    #[inline(always)]
    fn scale(self, s: V) -> V2 {
        v2(self.x * s, self.y * s)
    }
}
#[inline(always)]
fn add_ff(x: V, y: V) -> V2 {
    let s = x + y;
    v2(s, (x - s) + y)
}
#[inline(always)]
fn add2_ff(x: V, y: V) -> V2 {
    let s = x + y;
    let v = s - x;
    v2(s, (x - (s - v)) + (y - v))
}
#[inline(always)]
fn add2_ff2(x: V, y: V2) -> V2 {
    let s = x + y.x;
    let v = s - x;
    v2(s, ((x - (s - v)) + (y.x - v)) + y.y)
}
#[inline(always)]
fn add_f2f(x: V2, y: V) -> V2 {
    let s = x.x + y;
    v2(s, ((x.x - s) + y) + x.y)
}
#[inline(always)]
fn sub_f2f(x: V2, y: V) -> V2 {
    let s = x.x - y;
    v2(s, ((x.x - s) - y) + x.y)
}
#[inline(always)]
fn add2_f2f(x: V2, y: V) -> V2 {
    let s = x.x + y;
    let v = s - x.x;
    let t = (x.x - (s - v)) + (y - v);
    v2(s, t + x.y)
}
#[inline(always)]
fn add_ff2(x: V, y: V2) -> V2 {
    let s = x + y.x;
    v2(s, ((x - s) + y.x) + y.y)
}
#[inline(always)]
fn add_f2f2(x: V2, y: V2) -> V2 {
    let s = x.x + y.x;
    v2(s, (((x.x - s) + y.x) + x.y) + y.y)
}
#[inline(always)]
fn add2_f2f2(x: V2, y: V2) -> V2 {
    let s = x.x + y.x;
    let v = s - x.x;
    let t = (x.x - (s - v)) + (y.x - v);
    v2(s, t + (x.y + y.y))
}
#[inline(always)]
fn sub_f2f2(x: V2, y: V2) -> V2 {
    let s = x.x - y.x;
    let mut t = x.x - s;
    t = t - y.x;
    t = t + x.y;
    v2(s, t - y.y)
}
#[inline(always)]
fn div_f2f2(n: V2, d: V2) -> V2 {
    let t = d.x.rec();
    let s = n.x * t;
    let u = fmapn(t, n.x, s);
    let v = fmanp(d.y, t, fmanp(d.x, t, c(1.0)));
    v2(s, mla(s, v, mla(n.y, t, u)))
}
#[inline(always)]
fn mul_ff(x: V, y: V) -> V2 {
    let s = x * y;
    v2(s, fmapn(x, y, s))
}
#[inline(always)]
fn squ_f2(x: V2) -> V2 {
    let s = x.x * x.x;
    v2(s, mla(x.x + x.x, x.y, fmapn(x.x, x.x, s)))
}
#[inline(always)]
fn mul_f2f2(x: V2, y: V2) -> V2 {
    let s = x.x * y.x;
    v2(s, mla(x.x, y.y, mla(x.y, y.x, fmapn(x.x, y.x, s))))
}
#[inline(always)]
fn mul_f2f2_f(x: V2, y: V2) -> V {
    mla(x.x, y.x, mla(x.y, y.x, x.x * y.y))
}
#[inline(always)]
fn mul_f2f(x: V2, y: V) -> V2 {
    let s = x.x * y;
    v2(s, mla(x.y, y, fmapn(x.x, y, s)))
}
#[inline(always)]
fn rec_f(d: V) -> V2 {
    let s = d.rec();
    v2(s, s * fmanp(d, s, c(1.0)))
}
#[inline(always)]
fn rec_f2(d: V2) -> V2 {
    let s = d.x.rec();
    v2(s, s * fmanp(d.y, s, fmanp(d.x, s, c(1.0))))
}
#[inline(always)]
fn sqrt_f(d: V) -> V2 {
    let t = d.sqrt();
    mul_f2f2(add2_ff2(d, mul_ff(t, t)), rec_f(t)).scale(c(0.5))
}

// ---- constants (misc.h) -----------------------------------------------------------------

const R_LN2F: f32 =
    1.442_695_040_888_963_407_359_924_681_001_892_137_426_645_954_152_985_934_135_449_406_931_f32;
const L2UF: f32 = 0.693_145_751_953_125_f32;
const L2LF: f32 = 1.428_606_765_330_187_045e-06_f32;
const M_1_PI: f64 = 0.318_309_886_183_790_671_537_767_526_745_028_724;
const M_PI: f64 = 3.141_592_653_589_793_238_462_643_383_279_502_884;
const PI_A2F: f32 = 3.141_479_492_187_5_f32;
const PI_B2F: f32 = 0.000_113_159_418_106_079_101_56_f32;
const PI_C2F: f32 = 1.984_187_258_941_005_893_6e-09_f32;
const TRIGRANGEMAX2F: f32 = 125.0;
const LOG1PF_BOUND: f32 = 1.0e38; // 0x1.2ced32p+126
const FLT_MIN: f32 = f32::MIN_POSITIVE;
/// `(float)(1 << 32) * (float)(1 << 32)`.
const TWO_64: f32 = 18_446_744_073_709_551_616.0;

// ---- exp / log ----------------------------------------------------------------------------

/// `xexpf` (1.0 ULP).
#[inline(always)]
pub(crate) fn expf(d: V) -> V {
    let q = (d * c(R_LN2F)).rint_i();
    let qf = q.to_f();
    let mut s = mla(qf, c(-L2UF), d);
    s = mla(qf, c(-L2LF), s);
    let mut u = c(d2f(0.000_198_527_617_612_853_646_278_381));
    u = mla(u, s, c(d2f(0.001_393_043_552_525_341_510_772_71)));
    u = mla(u, s, c(d2f(0.008_333_360_776_305_198_669_433_59)));
    u = mla(u, s, c(d2f(0.041_666_485_369_205_474_853_515_6)));
    u = mla(u, s, c(d2f(0.166_666_671_633_720_397_949_219)));
    u = mla(u, s, c(d2f(0.5)));
    u = c(1.0) + mla(s * s, u, s);
    u = ldexp2(u, q);
    u = u.andnot_mask(d.lt(c(-104.0)));
    sel(c(100.0).lt(d), c(f32::INFINITY), u)
}

/// `xexp2f` (1.0 ULP).
fn exp2f(d: V) -> V {
    let mut u = d.rint();
    let q = u.rint_i();
    let s = d - u;
    u = c(d2f(0.1535920892e-3));
    u = mla(u, s, c(d2f(0.1339262701e-2)));
    u = mla(u, s, c(d2f(0.9618384764e-2)));
    u = mla(u, s, c(d2f(0.5550347269e-1)));
    u = mla(u, s, c(d2f(0.2402264476e+0)));
    u = mla(u, s, c(d2f(0.6931471825e+0)));
    u = mla(u, s, c(1.0));
    u = ldexp2(u, q);
    u = sel(d.ge(c(128.0)), c(f32::INFINITY), u);
    u.andnot_mask(d.lt(c(-150.0)))
}

/// `expk2f`.
#[inline(always)]
fn expk2f(d: V2) -> V2 {
    let u = d.sum() * c(R_LN2F);
    let q = u.rint_i();
    let mut s = add2_f2f(d, q.to_f() * c(-L2UF));
    s = add2_f2f(s, q.to_f() * c(-L2LF));
    let mut u = c(0.1980960224e-3_f32);
    u = mla(u, s.x, c(0.1394256484e-2_f32));
    u = mla(u, s.x, c(0.8333456703e-2_f32));
    u = mla(u, s.x, c(0.4166637361e-1_f32));
    let mut t = add2_f2f(mul_f2f(s, u), c(0.166666659414234244790680580464e+0_f32));
    t = add2_f2f(mul_f2f2(s, t), c(d2f(0.5)));
    t = add2_f2f2(s, mul_f2f2(squ_f2(s), t));
    t = add_ff2(c(1.0), t);
    t = v2(ldexp2(t.x, q), ldexp2(t.y, q));
    let small = d.x.lt(c(-104.0));
    v2(t.x.andnot_mask(small), t.y.andnot_mask(small))
}

/// `xexpm1f` (1.0 ULP).
fn expm1f(a: V) -> V {
    let d = add2_f2f(expk2f(v2(a, c(0.0))), c(d2f(-1.0)));
    let mut x = d.sum();
    x = sel(a.gt(c(88.722_831_726_074_218_75_f32)), c(f32::INFINITY), x);
    x = sel(a.lt(c(-16.635_532_333_438_687_426_013_570_f32)), c(-1.0), x);
    sel(a.isnegzero(), c(-0.0), x)
}

/// Shared front end of the log family: scale subnormals, split off the exponent.
fn log_split(d: V) -> (V, V, I) {
    let o = d.lt(c(FLT_MIN));
    let d = sel(o, d * c(TWO_64), d);
    let e = ilogb2k(d * c(1.0_f32 / 0.75_f32));
    let m = ldexp3(d, e.neg());
    let e = seli(o, e.sub(ci(64)), e);
    (d, m, e)
}

fn log_specials(r: V, d: V) -> V {
    let r = sel(d.ispinf(), c(f32::INFINITY), r);
    let r = sel(d.lt(c(0.0)).or(d.isnan()), c(f32::NAN), r);
    sel(d.eq(c(0.0)), c(f32::NEG_INFINITY), r)
}

/// `xlogf_u1` (1.0 ULP).
fn logf(d: V) -> V {
    let (d, m, e) = log_split(d);
    let mut s = mul_f2f(
        v2c(
            0.693_147_182_464_599_609_38_f32,
            -1.904_654_323_148_236_017e-09_f32,
        ),
        e.to_f(),
    );
    let x = div_f2f2(add2_ff(c(-1.0), m), add2_ff(c(1.0), m));
    let x2 = x.x * x.x;
    let mut t = c(0.3027294874e+0_f32);
    t = mla(t, x2, c(0.3996108174e+0_f32));
    t = mla(t, x2, c(0.6666694880e+0_f32));
    s = add_f2f2(s, x.scale(c(2.0)));
    s = add_f2f(s, (x2 * x.x) * t);
    log_specials(s.sum(), d)
}

/// `xlog2f` (1.0 ULP).
fn log2f(d: V) -> V {
    let o = d.lt(c(FLT_MIN));
    let d = sel(o, d * c(TWO_64), d);
    let e = ilogb2k(d * c(d2f(1.0 / 0.75)));
    let m = ldexp3(d, e.neg());
    let e = seli(o, e.sub(ci(64)), e);
    let x = div_f2f2(add2_ff(c(-1.0), m), add2_ff(c(1.0), m));
    let x2 = x.x * x.x;
    let mut t = c(0.4374550283e+0_f32);
    t = mla(t, x2, c(0.5764790177e+0_f32));
    t = mla(t, x2, c(0.9618012905120_f32));
    let mut s = add2_ff2(
        e.to_f(),
        mul_f2f2(
            x,
            v2c(
                d2f(2.885_390_043_258_666_992_2),
                d2f(3.273_447_448_356_848_861_6e-08),
            ),
        ),
    );
    s = add2_f2f(s, (x2 * x.x) * t);
    log_specials(s.sum(), d)
}

/// `xlog10f` (1.0 ULP).
fn log10f(d: V) -> V {
    let o = d.lt(c(FLT_MIN));
    let d = sel(o, d * c(TWO_64), d);
    let e = ilogb2k(d * c(d2f(1.0 / 0.75)));
    let m = ldexp3(d, e.neg());
    let e = seli(o, e.sub(ci(64)), e);
    let x = div_f2f2(add2_ff(c(-1.0), m), add2_ff(c(1.0), m));
    let x2 = x.x * x.x;
    let mut t = c(d2f(0.1314289868e+0));
    t = mla(t, x2, c(d2f(0.1735493541e+0)));
    t = mla(t, x2, c(d2f(0.2895309627e+0)));
    let mut s = mul_f2f(v2c(d2f(0.30103001), d2f(-1.432098889e-08)), e.to_f());
    s = add_f2f2(s, mul_f2f2(x, v2c(d2f(0.868588984), d2f(-2.170757285e-08))));
    s = add_f2f(s, (x2 * x.x) * t);
    log_specials(s.sum(), d)
}

/// `xlog1pf` (1.0 ULP).
fn log1pf(d: V) -> V {
    let mut dp1 = d + c(1.0);
    let o = dp1.lt(c(FLT_MIN));
    dp1 = sel(o, dp1 * c(TWO_64), dp1);
    let mut e = ilogb2k(dp1 * c(1.0_f32 / 0.75_f32));
    let t = ldexp3(c(1.0), e.neg());
    let m = mla(d, t, t - c(1.0));
    e = seli(o, e.sub(ci(64)), e);
    let mut s = mul_f2f(
        v2c(
            0.693_147_182_464_599_609_38_f32,
            -1.904_654_323_148_236_017e-09_f32,
        ),
        e.to_f(),
    );
    let x = div_f2f2(v2(m, c(0.0)), add_ff(c(2.0), m));
    let x2 = x.x * x.x;
    let mut t = c(0.3027294874e+0_f32);
    t = mla(t, x2, c(0.3996108174e+0_f32));
    t = mla(t, x2, c(0.6666694880e+0_f32));
    s = add_f2f2(s, x.scale(c(2.0)));
    s = add_f2f(s, (x2 * x.x) * t);
    let mut r = s.sum();
    let core = d.le(c(LOG1PF_BOUND));
    if !core.all() {
        r = sel(core, r, logf(d));
    }
    r = r.or_mask(c(-1.0).gt(d));
    r = sel(d.eq(c(-1.0)), c(f32::NEG_INFINITY), r);
    sel(d.isnegzero(), c(-0.0), r)
}

/// `logk2f`.
fn logk2f(d: V2) -> V2 {
    let e = ilogbk(d.x * c(1.0_f32 / 0.75_f32));
    let m = d.scale(pow2i(e.neg()));
    let x = div_f2f2(add2_f2f(m, c(-1.0)), add2_f2f(m, c(1.0)));
    let x2 = squ_f2(x);
    let mut t = c(0.239_282_846_450_805_664_062_5_f32);
    t = mla(t, x2.x, c(0.285_182_118_415_832_519_531_25_f32));
    t = mla(t, x2.x, c(0.400_005_877_017_974_853_515_625_f32));
    t = mla(t, x2.x, c(0.666_666_686_534_881_591_796_875_f32));
    let mut s = mul_f2f(
        v2c(
            0.693_147_182_464_599_609_38_f32,
            -1.904_654_323_148_236_017e-09_f32,
        ),
        e.to_f(),
    );
    s = add_f2f2(s, x.scale(c(2.0)));
    add_f2f2(s, mul_f2f(mul_f2f2(x2, x), t))
}

/// `vcast_vf2_d`: a double split into a float pair.
fn v2d(d: f64) -> V2 {
    let h = d as f32;
    v2c(h, (d - f64::from(h)) as f32)
}
/// `vsel_vf_vo_vo_vo_f_f_f_f`: the first value whose mask holds, else the last.
fn sel4(o0: M, o1: M, o2: M, d: [f32; 4]) -> V {
    sel(o0, c(d[0]), sel(o1, c(d[1]), sel(o2, c(d[2]), c(d[3]))))
}
fn sel4_2(o0: M, o1: M, o2: M, d: [f64; 4]) -> V2 {
    sel2(
        o0,
        v2d(d[0]),
        sel2(o1, v2d(d[1]), sel2(o2, v2d(d[2]), v2d(d[3]))),
    )
}

/// `xerfcf_u15` (1.5 ULP).
fn erfcf(a: V) -> V {
    let s = a;
    let a = a.abs();
    let o0 = a.lt(c(1.0));
    let o1 = a.lt(c(2.2));
    let o2 = a.lt(c(4.3));
    let o3 = a.lt(c(10.1));
    let u = sel2(o1, v2(a, c(0.0)), div_f2f2(v2c(1.0, 0.0), v2(a, c(0.0))));
    let mut t = sel4(
        o0,
        o1,
        o2,
        [
            -0.8638041618e-4,
            -0.6236977242e-5,
            -0.3869504035e+0,
            0.1115344167e+1,
        ],
    );
    t = mla(
        t,
        u.x,
        sel4(
            o0,
            o1,
            o2,
            [
                0.6000166177e-3,
                0.5749821503e-4,
                0.1288077235e+1,
                -0.9454904199e+0,
            ],
        ),
    );
    t = mla(
        t,
        u.x,
        sel4(
            o0,
            o1,
            o2,
            [
                -0.1665703603e-2,
                0.6002851478e-5,
                -0.1816803217e+1,
                -0.3667259514e+0,
            ],
        ),
    );
    t = mla(
        t,
        u.x,
        sel4(
            o0,
            o1,
            o2,
            [
                0.1795156277e-3,
                -0.2851036377e-2,
                0.1249150872e+1,
                0.7155663371e+0,
            ],
        ),
    );
    t = mla(
        t,
        u.x,
        sel4(
            o0,
            o1,
            o2,
            [
                0.1914106123e-1,
                0.2260518074e-1,
                -0.1328857988e+0,
                -0.1262947265e-1,
            ],
        ),
    );
    let mut d = mul_f2f(u, t);
    d = add2_f2f2(
        d,
        sel4_2(
            o0,
            o1,
            o2,
            [
                -0.102_775_359_343_930_288_081_655_368_891e+0,
                -0.105_247_583_459_338_632_253_369_014_063e+0,
                -0.482_365_310_333_045_318_680_618_892_669e+0,
                -0.498_961_546_254_537_647_970_305_302_739e+0,
            ],
        ),
    );
    d = mul_f2f2(d, u);
    d = add2_f2f2(
        d,
        sel4_2(
            o0,
            o1,
            o2,
            [
                -0.636_619_483_208_481_931_303_752_546_439e+0,
                -0.635_609_463_574_589_034_216_723_775_292e+0,
                -0.134_450_203_224_533_979_217_859_332_703e-2,
                -0.471_199_543_422_848_492_080_722_832_666e-4,
            ],
        ),
    );
    d = mul_f2f2(d, u);
    d = add2_f2f2(
        d,
        sel4_2(
            o0,
            o1,
            o2,
            [
                -0.112_837_917_790_537_404_939_545_770_596e+1,
                -0.112_855_987_376_668_622_084_547_028_949e+1,
                -0.572_319_781_150_472_949_561_786_101_080e+0,
                -0.572_364_030_327_966_044_425_932_623_525e+0,
            ],
        ),
    );
    let mut x = mul_f2f(sel2(o1, d, v2(-a, c(0.0))), a);
    x = sel2(o1, x, add2_f2f2(x, d));
    x = expk2f(x);
    x = sel2(o1, x, mul_f2f2(x, u));
    let mut r = sel(o3, x.sum(), c(0.0));
    r = sel(s.signbit(), c(2.0) - r, r);
    sel(s.isnan(), c(f32::NAN), r)
}

// ---- hyperbolic -----------------------------------------------------------------------------

/// `xtanhf` (1.0 ULP).
fn tanhf(x: V) -> V {
    let mut y = x.abs();
    let mut d = expk2f(v2(y, c(0.0)));
    let e = rec_f2(d);
    d = div_f2f2(add_f2f2(d, e.neg()), add_f2f2(d, e));
    y = d.sum();
    y = sel(x.abs().gt(c(8.664_339_742_f32)).or(y.isnan()), c(1.0), y);
    y = y.mulsign(x);
    y.or_mask(x.isnan())
}

/// `xatanhf` (1.0 ULP).
fn atanhf(x: V) -> V {
    let mut y = x.abs();
    let d = logk2f(div_f2f2(add2_ff(c(1.0), y), add2_ff(c(1.0), -y)));
    let inner = sel(y.eq(c(d2f(1.0))), c(f32::INFINITY), d.sum() * c(d2f(0.5)));
    y = inner.or_mask(y.gt(c(d2f(1.0))));
    y = y.or_mask(x.isinf().or(y.isnan()));
    y = y.mulsign(x);
    y.or_mask(x.isnan())
}

// ---- trigonometric ----------------------------------------------------------------------------

#[rustfmt::skip]
static REMPITAB: [f32; 416] = include!("sleef_rempitab.in");

/// `rempisubf` with FULL_FP_ROUNDING.
fn rempisubf(x: V) -> (V, I) {
    let y = (x * c(4.0)).rint();
    let vi = (y - x.rint() * c(4.0)).trunc_i();
    (x - y * c(0.25), vi)
}

/// `rempif`: Payne-Hanek reduction, x = (a mod pi/2) as a double-float, with the quadrant.
fn rempif(a: V) -> (V2, I) {
    let mut ex = ilogb2k(a).sub(ci(25));
    let q = ci(-64).and_mask(ex.gt(ci(90 - 25)));
    let a = ldexp3(a, q);
    ex = ex.sra(31).andnot(ex);
    ex = ex.map(|e| e << 2);
    let gather = |k: usize| V(std::array::from_fn(|i| REMPITAB[ex.0[i] as usize + k]));
    let mut x = mul_ff(a, gather(0));
    let (dx, qi) = rempisubf(x.x);
    let mut q = qi;
    x = v2(dx, x.y).normalize();
    let y = mul_ff(a, gather(1));
    x = add2_f2f2(x, y);
    let (dx, qi) = rempisubf(x.x);
    q = q.add(qi);
    x = v2(dx, x.y).normalize();
    let y = mul_f2f(v2(gather(2), gather(3)), a);
    x = add2_f2f2(x, y).normalize();
    x = mul_f2f2(
        x,
        v2c(
            3.141_592_741_012_573_242_2_f32 * 2.0,
            -8.742_277_657_347_585_773_1e-08_f32 * 2.0,
        ),
    );
    x = sel2(a.abs().lt(c(0.7)), v2(a, c(0.0)), x);
    (x, q)
}

/// The shared sin/cos polynomial on the reduced argument t.
fn sincos_poly(t: V2) -> V {
    let s = squ_f2(t);
    let mut u = c(2.608_315_980_978_659_354_150_3e-06_f32);
    u = mla(u, s.x, c(-0.000_198_106_907_191_686_332_225_8_f32));
    u = mla(u, s.x, c(0.008_333_078_585_565_090_179_443_36_f32));
    let x = add_ff2(
        c(1.0),
        mul_f2f2(
            add_ff(c(-0.166_666_597_127_914_428_710_938_f32), u * s.x),
            s,
        ),
    );
    mul_f2f2_f(t, x)
}

/// `xsinf_u1` (1.0 ULP).
fn sinf(d: V) -> V {
    let (q, s);
    if d.abs().lt(c(TRIGRANGEMAX2F)).all() {
        let u = (d * c(d2f(M_1_PI))).rint();
        q = u.rint_i();
        let v = mla(u, c(-PI_A2F), d);
        let t = add2_ff(v, u * c(-PI_B2F));
        s = add_f2f(t, u * c(-PI_C2F));
    } else {
        let (df, i) = rempif(d);
        let mut qq = i.and(ci(3));
        qq = qq.add(qq).add(seli(df.x.gt(c(0.0)), ci(2), ci(1)));
        q = qq.sra(2);
        let o = i.and(ci(1)).eq(ci(1));
        let x = v2(
            c(3.141_592_741_012_573_242_2_f32 * -0.5).mulsign(df.x),
            c(-8.742_277_657_347_585_773_1e-08_f32 * -0.5).mulsign(df.x),
        );
        let x = add2_f2f2(df, x);
        let t = sel2(o, x, df).normalize();
        s = v2(t.x.or_mask(d.isinf().or(d.isnan())), t.y);
    }
    let u = sincos_poly(s).xor_sign(q.and(ci(1)).eq(ci(1)));
    sel(d.isnegzero(), d, u)
}

/// `xcosf_u1` (1.0 ULP).
fn cosf(d: V) -> V {
    let (q, s);
    if d.abs().lt(c(TRIGRANGEMAX2F)).all() {
        let dq = mla(mla(d, c(d2f(M_1_PI)), c(-0.5)).rint(), c(2.0), c(1.0));
        q = dq.rint_i();
        let t = add2_ff(d, dq * c(-PI_A2F * 0.5));
        let t = add2_f2f(t, dq * c(-PI_B2F * 0.5));
        s = add2_f2f(t, dq * c(-PI_C2F * 0.5));
    } else {
        let (df, i) = rempif(d);
        let mut qq = i.and(ci(3));
        qq = qq.add(qq).add(seli(df.x.gt(c(0.0)), ci(8), ci(7)));
        q = qq.sra(1);
        let o = i.and(ci(1)).eq(ci(0));
        let y = sel(df.x.gt(c(0.0)), c(0.0), c(-1.0));
        let x = v2(
            c(3.141_592_741_012_573_242_2_f32 * -0.5).mulsign(y),
            c(-8.742_277_657_347_585_773_1e-08_f32 * -0.5).mulsign(y),
        );
        let x = add2_f2f2(df, x);
        let t = sel2(o, x, df).normalize();
        s = v2(t.x.or_mask(d.isinf().or(d.isnan())), t.y);
    }
    sincos_poly(s).xor_sign(q.and(ci(2)).eq(ci(0)))
}

/// `xtanf_u1` (1.0 ULP).
fn tanf(d: V) -> V {
    let (q, mut s);
    if d.abs().lt(c(TRIGRANGEMAX2F)).all() {
        let u = (d * c(d2f(2.0 * M_1_PI))).rint();
        q = u.rint_i();
        let v = mla(u, c(-PI_A2F * 0.5), d);
        let t = add2_ff(v, u * c(-PI_B2F * 0.5));
        s = add_f2f(t, u * c(-PI_C2F * 0.5));
    } else {
        let (df, i) = rempif(d);
        q = i;
        let o = d.isinf().or(d.isnan());
        s = v2(df.x.or_mask(o), df.y.or_mask(o));
    }
    let o = q.and(ci(1)).eq(ci(1));
    s = v2(s.x.xor_sign(o), s.y.xor_sign(o));
    let t = s;
    s = squ_f2(s).normalize();
    let mut u = c(0.004_466_364_625_841_379_165_649_41_f32);
    u = mla(u, s.x, c(-8.392_018_207_814_544_439_315_8e-05_f32));
    u = mla(u, s.x, c(0.010_963_924_229_145_050_048_828_1_f32));
    u = mla(u, s.x, c(0.021_236_030_384_898_185_729_980_5_f32));
    u = mla(u, s.x, c(0.054_068_714_380_264_282_226_562_5_f32));
    let x = add_ff(c(0.133_325_666_189_193_725_585_938_f32), u * s.x);
    let x = add_ff2(
        c(1.0),
        mul_f2f2(
            add_ff2(c(0.333_333_611_488_342_285_156_25_f32), mul_f2f2(s, x)),
            s,
        ),
    );
    let x = mul_f2f2(t, x);
    let x = sel2(o, rec_f2(x), x);
    sel(d.isnegzero(), d, x.sum())
}

/// `atan2kf_u1`.
fn atan2kf_u1(y: V2, x: V2) -> V2 {
    let mut q = seli(x.x.lt(c(0.0)), ci(-2), ci(0));
    let p = x.x.lt(c(0.0));
    let x = v2(x.x.xor_sign(p), x.y.xor_sign(p));
    q = seli(x.x.lt(y.x), q.add(ci(1)), q);
    let p = x.x.lt(y.x);
    let s = sel2(p, x.neg(), y);
    let t = sel2(p, y, x);
    let s = div_f2f2(s, t);
    let t = squ_f2(s).normalize();
    let mut u = c(-0.001_763_979_089_446_365_833_282_47_f32);
    u = mla(u, t.x, c(0.010_790_090_076_625_347_137_451_2_f32));
    u = mla(u, t.x, c(-0.030_956_460_162_997_245_788_574_2_f32));
    u = mla(u, t.x, c(0.057_736_508_548_259_735_107_421_9_f32));
    u = mla(u, t.x, c(-0.083_895_072_340_965_270_996_093_8_f32));
    u = mla(u, t.x, c(0.109_463_557_600_975_036_621_094_f32));
    u = mla(u, t.x, c(-0.142_626_821_994_781_494_140_625_f32));
    u = mla(u, t.x, c(0.199_983_194_470_405_578_613_281_f32));
    let t = mul_f2f2(
        t,
        add_ff(c(-0.333_332_866_430_282_592_773_438_f32), u * t.x),
    );
    let t = mul_f2f2(s, add_ff2(c(1.0), t));
    add_f2f2(
        mul_f2f(
            v2c(
                1.570_796_370_506_286_621_1_f32,
                -4.371_138_828_673_792_886_5e-08_f32,
            ),
            q.to_f(),
        ),
        t,
    )
}

/// `visinf2_vf_vf_vf`: where d is infinite, m with d's sign or-ed in; +0.0 elsewhere.
fn isinf2(d: V, m: V) -> V {
    V(std::array::from_fn(|i| {
        if d.0[i].is_infinite() {
            f32::from_bits((d.0[i].to_bits() & 0x8000_0000) | m.0[i].to_bits())
        } else {
            0.0
        }
    }))
}

/// `xatan2f_u1` (1.0 ULP).
fn atan2f(y: V, x: V) -> V {
    let o = x.abs().lt(c(2.938_737_278_354_183_094_7e-39_f32));
    let x = sel(o, x * c(16_777_216.0), x);
    let y = sel(o, y * c(16_777_216.0), y);
    let d = atan2kf_u1(v2(y.abs(), c(0.0)), v2(x, c(0.0)));
    let mut r = d.sum();
    r = r.mulsign(x);
    let half_pi = c(d2f(M_PI / 2.0));
    r = sel(
        x.isinf().or(x.eq(c(0.0))),
        half_pi - isinf2(x, half_pi.mulsign(x)),
        r,
    );
    r = sel(
        y.isinf(),
        half_pi - isinf2(x, c(d2f(M_PI / 4.0)).mulsign(x)),
        r,
    );
    r = sel(y.eq(c(0.0)), c(M_PI as f32).and_mask(x.signbit()), r);
    r.mulsign(y).or_mask(x.isnan().or(y.isnan()))
}

/// `xasinf_u1` (1.0 ULP).
fn asinf(d: V) -> V {
    let o = d.abs().lt(c(0.5));
    let x2 = sel(o, d * d, (c(1.0) - d.abs()) * c(0.5));
    let mut x = sel2(o, v2(d.abs(), c(0.0)), sqrt_f(x2));
    x = sel2(d.abs().eq(c(1.0)), v2c(0.0, 0.0), x);
    let mut u = c(d2f(0.4197454825e-1));
    u = mla(u, x2, c(d2f(0.2424046025e-1)));
    u = mla(u, x2, c(d2f(0.4547423869e-1)));
    u = mla(u, x2, c(d2f(0.7495029271e-1)));
    u = mla(u, x2, c(d2f(0.1666677296e+0)));
    u = u * (x2 * x.x);
    let y = sub_f2f(
        sub_f2f2(
            v2c(
                3.141_592_741_012_573_242_2_f32 / 4.0,
                -8.742_277_657_347_585_773_1e-08_f32 / 4.0,
            ),
            x,
        ),
        u,
    );
    let r = sel(o, u + x.x, y.sum() * c(2.0));
    r.mulsign(d)
}

/// `xacosf_u1` (1.0 ULP).
fn acosf(d: V) -> V {
    let o = d.abs().lt(c(0.5));
    let x2 = sel(o, d * d, (c(1.0) - d.abs()) * c(0.5));
    let mut x = sel2(o, v2(d.abs(), c(0.0)), sqrt_f(x2));
    x = sel2(d.abs().eq(c(1.0)), v2c(0.0, 0.0), x);
    let mut u = c(d2f(0.4197454825e-1));
    u = mla(u, x2, c(d2f(0.2424046025e-1)));
    u = mla(u, x2, c(d2f(0.4547423869e-1)));
    u = mla(u, x2, c(d2f(0.7495029271e-1)));
    u = mla(u, x2, c(d2f(0.1666677296e+0)));
    u = u * (x2 * x.x);
    let mut y = sub_f2f2(
        v2c(
            3.141_592_741_012_573_242_2_f32 / 2.0,
            -8.742_277_657_347_585_773_1e-08_f32 / 2.0,
        ),
        add_ff(x.x.mulsign(d), u.mulsign(d)),
    );
    x = add_f2f(x, u);
    y = sel2(o, y, x.scale(c(2.0)));
    y = sel2(
        o.andnot(d.lt(c(0.0))),
        sub_f2f2(
            v2c(
                3.141_592_741_012_573_242_2_f32,
                -8.742_277_657_347_585_773_1e-08_f32,
            ),
            y,
        ),
        y,
    );
    y.sum()
}

/// `xatanf_u1` (1.0 ULP).
fn atanf(d: V) -> V {
    let d2 = atan2kf_u1(v2(d.abs(), c(0.0)), v2c(1.0, 0.0));
    let r = sel(
        d.isinf(),
        c(d2f(1.570_796_326_794_896_557_998_982)),
        d2.sum(),
    );
    r.mulsign(d)
}

// ---- functions PyTorch takes from the C library on this platform ------------------------------

mod libm {
    unsafe extern "C" {
        pub fn asinhf(x: f32) -> f32;
        pub fn acoshf(x: f32) -> f32;
        pub fn sinhf(x: f32) -> f32;
        pub fn coshf(x: f32) -> f32;
        pub fn erff(x: f32) -> f32;
        pub fn expf(x: f32) -> f32;
        pub fn logf(x: f32) -> f32;
        pub fn pow(x: f64, y: f64) -> f64;
        pub fn exp2f(x: f32) -> f32;
        pub fn atan2f(y: f32, x: f32) -> f32;
    }
}

fn libm_erf(x: f32) -> f32 {
    // SAFETY: a pure C math function on a plain float.
    unsafe { libm::erff(x) }
}
pub(crate) fn libm_exp(x: f32) -> f32 {
    // SAFETY: as above.
    unsafe { libm::expf(x) }
}
/// C `pow` on doubles, as Python's `float ** float` calls it.
pub(crate) fn libm_pow(x: f64, y: f64) -> f64 {
    // SAFETY: a pure C math function on plain doubles.
    unsafe { libm::pow(x, y) }
}
pub(crate) fn libm_log(x: f32) -> f32 {
    // SAFETY: as above.
    unsafe { libm::logf(x) }
}
fn libm_atan2(y: f32, x: f32) -> f32 {
    // SAFETY: a pure C math function on plain floats.
    unsafe { libm::atan2f(y, x) }
}
fn libm_exp2(x: f32) -> f32 {
    // SAFETY: as above.
    unsafe { libm::exp2f(x) }
}

/// PyTorch's `asinh`, `acosh`, `sinh` and `cosh` call the platform C library (`std::asinh`
/// and so on), not SLEEF; Rust's own `f32::asinh` is a different formula.
pub fn libm_asinh(x: f32) -> f32 {
    // SAFETY: a pure C math function on a plain float.
    unsafe { libm::asinhf(x) }
}
pub fn libm_acosh(x: f32) -> f32 {
    // SAFETY: as above.
    unsafe { libm::acoshf(x) }
}
pub fn libm_sinh(x: f32) -> f32 {
    // SAFETY: as above.
    unsafe { libm::sinhf(x) }
}
pub fn libm_cosh(x: f32) -> f32 {
    // SAFETY: as above.
    unsafe { libm::coshf(x) }
}

// ---- PyTorch's own aarch64 vector math (aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h) ----

const U20_BOUND: f32 = f32::from_bits(0x42AE_AF15); // 0x1.5d5e2ap+6
const U20_INV_LN2: f32 = f32::from_bits(0x3FB8_AA3B); // 0x1.715476p+0
const U20_LN2_HI: f32 = f32::from_bits(0x3F31_7200); // 0x1.62e4p-1
const U20_LN2_LO: f32 = f32::from_bits(0x35BF_BE8E); // 0x1.7f7d1cp-20
const U20_C0: f32 = f32::from_bits(0x3C07_2010); // 0x1.0e4020p-7
const U20_C1: f32 = f32::from_bits(0x3D2B_9F17); // 0x1.573e2ep-5
const U20_C2: f32 = f32::from_bits(0x3E2A_AF33); // 0x1.555e66p-3
const U20_C3: f32 = f32::from_bits(0x3EFF_FEDB); // 0x1.fffdb6p-2
const U20_C4: f32 = f32::from_bits(0x3F7F_FFF6); // 0x1.ffffecp-1

/// `vexpq_f32_u20` (Arm Optimized Routines expf): if any lane has |x| > 87.3..., the whole
/// group goes to SLEEF's expf instead.
#[inline(always)]
pub(crate) fn exp_u20(x: V) -> V {
    if x.abs().gt(c(U20_BOUND)).0.iter().any(|&b| b) {
        return expf(x);
    }
    let n = (x * c(U20_INV_LN2)).map(f32::round); // vrndaq: halves away from zero
    let r = fmanp(n, c(U20_LN2_HI), x);
    let r = fmanp(n, c(U20_LN2_LO), r);
    let scale = from_bits(n.trunc_i().map(|e| (e << 23).wrapping_add(0x3f80_0000)));
    let r2 = r * r;
    let p = mla(r, c(U20_C0), c(U20_C1));
    let q = mla(r, c(U20_C2), c(U20_C3));
    let q = mla(p, r2, q);
    let p = c(U20_C4) * r;
    let poly = mla(q, r2, p);
    mla(poly, scale, scale)
}

/// `Vectorized<float>::erf` (Abramowitz-Stegun 7.1.26 with exp_u20).
#[inline(always)]
fn erf_vec(x: V) -> V {
    let sign = x.signbit();
    let abs = x.abs();
    let t = c(1.0).div(mla(c(0.327_591_1_f32), abs, c(1.0)));
    let r = mla(c(1.061_405_429_f32), t, c(-1.453_152_027_f32));
    let r = mla(r, t, c(1.421_413_741_f32));
    let r = mla(r, t, c(-0.284_496_736_f32));
    let r = mla(r, t, c(0.254_829_592_f32));
    let neg_pow_2 = -(x * x);
    let tmp5 = -exp_u20(neg_pow_2);
    let tmp6 = t * tmp5;
    mla(tmp6, r, c(1.0)).xor_sign(sign)
}

/// The `[begin, end)` chunks `at::parallel_for` gives OpenMP threads for `n` items.
pub(crate) fn parallel_chunks(n: usize, grain: usize) -> Vec<(usize, usize)> {
    let threads = crate::torch_sum::num_threads();
    if n == 0 {
        return Vec::new();
    }
    if n <= grain || threads == 1 {
        return vec![(0, n)];
    }
    let count = threads.min(n.div_ceil(grain));
    let size = n.div_ceil(count);
    (0..count)
        .map(|t| (t * size, n.min((t + 1) * size)))
        .filter(|(b, e)| b < e)
        .collect()
}

/// PyTorch's VML unary path (`IMPLEMENT_FLOAT_KERNEL`, grain 2048): each thread chunk is
/// mapped in 4-lane groups from its own start, the last group zero padded.
pub fn vml(xs: &[f32], f: fn(V) -> V) -> Vec<f32> {
    let mut out = Vec::with_capacity(xs.len());
    for (b, e) in parallel_chunks(xs.len(), 2048) {
        out.extend(map4(&xs[b..e], f));
    }
    out
}

/// `cpu_kernel_vec` for a unary op: per thread chunk, pairs of 4-lane vectors, then the
/// scalar function for the last (fewer than 8) elements of the chunk.
fn kernel_vec(
    xs: &[f32],
    grain: usize,
    scalar: impl Fn(f32) -> f32,
    vector: fn(V) -> V,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(xs.len());
    for (b, e) in parallel_chunks(xs.len(), grain) {
        let chunk = &xs[b..e];
        let whole = chunk.len() / 8 * 8;
        for group in chunk[..whole].as_chunks::<4>().0 {
            out.extend_from_slice(&vector(V(std::array::from_fn(|i| group[i]))).0);
        }
        out.extend(chunk[whole..].iter().map(|&x| scalar(x)));
    }
    out
}

/// Two-input `cpu_kernel_vec` (both operands contiguous, same length).
fn kernel_vec2(
    a: &[f32],
    b: &[f32],
    grain: usize,
    scalar: impl Fn(f32, f32) -> f32,
    vector: fn(V, V) -> V,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len());
    for (s, e) in parallel_chunks(a.len(), grain) {
        let (ca, cb) = (&a[s..e], &b[s..e]);
        let whole = ca.len() / 8 * 8;
        for (ga, gb) in ca[..whole]
            .as_chunks::<4>()
            .0
            .iter()
            .zip(cb[..whole].as_chunks::<4>().0)
        {
            let lanes = |g: &[f32]| V(std::array::from_fn(|i| g[i]));
            out.extend_from_slice(&vector(lanes(ga), lanes(gb)).0);
        }
        out.extend(
            ca[whole..]
                .iter()
                .zip(&cb[whole..])
                .map(|(&x, &y)| scalar(x, y)),
        );
    }
    out
}

/// `exp2_kernel` (cpu_kernel_vec; the scalar tail is the C library's exp2f).
pub fn exp2_array(xs: &[f32]) -> Vec<f32> {
    kernel_vec(xs, 32768, libm_exp2, exp2f)
}

pub fn erfc_array(xs: &[f32]) -> Vec<f32> {
    vml(xs, erfcf)
}

/// `tanh_backward_kernel`: grad * (1 - y * y); clang fuses `1 - y * y` in the scalar tail.
fn tanh_backward(g: &[f32], y: &[f32]) -> Vec<f32> {
    kernel_vec2(
        g,
        y,
        32768,
        |g, y| g * (-y).mul_add(y, 1.0),
        |g, y| g * (c(1.0) - y * y),
    )
}

const GELU_ALPHA: f32 = std::f64::consts::FRAC_1_SQRT_2 as f32;
/// `M_2_SQRTPI * M_SQRT1_2 * 0.5`, computed in double and stored as float.
const GELU_BETA: f32 =
    (std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 * 0.5) as f32;

#[inline(always)]
fn gelu_vec(x: V) -> V {
    x * c(0.5) * (c(1.0) + erf_vec(x * c(GELU_ALPHA)))
}
fn gelu_scalar(x: f32) -> f32 {
    x * 0.5 * (1.0 + libm_erf(x * GELU_ALPHA))
}
/// d gelu / dx as PyTorch's backward forms it: cdf + x * pdf (dy multiplies afterwards).
#[inline(always)]
fn gelu_slope_vec(x: V) -> V {
    let cdf = c(0.5) * (c(1.0) + erf_vec(x * c(GELU_ALPHA)));
    let pdf = c(GELU_BETA) * expf(x * x * c(-0.5));
    cdf + x * pdf
}
fn gelu_slope_scalar(x: f32) -> f32 {
    let cdf = 0.5 * (1.0 + libm_erf(x * GELU_ALPHA));
    let pdf = GELU_BETA * libm_exp(x * x * -0.5);
    x.mul_add(pdf, cdf) // clang contracts `cdf + x * pdf` in the scalar lambda
}

/// `torch.nn.functional.gelu` (exact): `GeluKernelImpl` uses grain numel / threads above
/// 16384 elements.
pub fn gelu(xs: &[f32]) -> Vec<f32> {
    let threads = crate::torch_sum::num_threads();
    let grain = if xs.len() > 16384 {
        (xs.len() / threads).max(1)
    } else {
        32768
    };
    kernel_vec(xs, grain, gelu_scalar, gelu_vec)
}

/// The slope `GeluBackwardKernelImpl` multiplies the incoming gradient by (default grain).
pub fn gelu_slope(xs: &[f32]) -> Vec<f32> {
    kernel_vec(xs, 32768, gelu_slope_scalar, gelu_slope_vec)
}

pub fn erf_array(xs: &[f32]) -> Vec<f32> {
    vml(xs, erf_vec)
}

#[inline(always)]
fn sigmoid_vec(x: V) -> V {
    (c(1.0) + expf(-x)).rec()
}
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + libm_exp(-x))
}
/// `torch.sigmoid` (`sigmoid_kernel`: cpu_kernel_vec, default grain).
pub fn sigmoid_array(xs: &[f32]) -> Vec<f32> {
    kernel_vec(xs, 32768, sigmoid_scalar, sigmoid_vec)
}

/// PyTorch's gradient for a unary op (tools/autograd/derivatives.yaml), step by step as its
/// tensor ops round: `x` is the input, `g` the incoming gradient. None when not ported yet.
pub fn unary_backward(kind: crate::dispatch::UnaryMath, x: &[f32], g: &[f32]) -> Option<Vec<f32>> {
    use crate::dispatch::UnaryMath as U;
    let zip = |a: &[f32], f: &dyn Fn(f32, f32) -> f32| -> Vec<f32> {
        a.iter().zip(g).map(|(&a, &g)| f(a, g)).collect()
    };
    let ln2 = std::f64::consts::LN_2 as f32;
    let ln10 = std::f64::consts::LN_10 as f32;
    let two_sqrtpi = std::f64::consts::FRAC_2_SQRT_PI as f32;
    let rsqrt = |t: f32| 1.0 / t.sqrt();
    Some(match kind {
        U::Exp => zip(&exp_array(x), &|y, g| g * y),
        U::Exp2 => zip(&exp2_array(x), &|y, g| g * y * ln2),
        U::Expm1 => zip(&expm1_array(x), &|y, g| g * (y + 1.0)),
        U::Log => zip(x, &|x, g| g / x),
        U::Log2 => zip(x, &|x, g| g / (x * ln2)),
        U::Log10 => zip(x, &|x, g| g / (x * ln10)),
        U::Log1p => zip(x, &|x, g| g / (x + 1.0)),
        U::Sin => zip(&cos_array(x), &|cx, g| g * cx),
        U::Cos => zip(&sin_array(x), &|sx, g| g * -sx),
        U::Tan => zip(&tan_array(x), &|y, g| g * (1.0 + y * y)),
        U::Tanh => tanh_backward(g, &tanh_array(x)),
        U::Sigmoid => zip(&sigmoid_array(x), &|y, g| g * (1.0 - y) * y),
        U::Asin => zip(x, &|x, g| g * rsqrt(-x * x + 1.0)),
        U::Acos => zip(x, &|x, g| g * -rsqrt(-x * x + 1.0)),
        U::Atan => zip(x, &|x, g| g / (x * x + 1.0)),
        U::Sinh => zip(x, &|x, g| g * libm_cosh(x)),
        U::Cosh => zip(x, &|x, g| g * libm_sinh(x)),
        U::Asinh => zip(x, &|x, g| g * rsqrt(x * x + 1.0)),
        U::Acosh => zip(x, &|x, g| g * rsqrt(x * x - 1.0)),
        U::Atanh => zip(x, &|x, g| g / (1.0 - x * x)),
        U::Sqrt => zip(x, &|x, g| g / (x.sqrt() * 2.0)),
        U::Rsqrt => zip(x, &|x, g| {
            let y = rsqrt(x);
            g * -0.5 * (y * y * y)
        }),
        U::Reciprocal => zip(x, &|x, g| {
            let y = 1.0 / x;
            -g * (y * y)
        }),
        U::Erf | U::Erfc => {
            let neg_sq: Vec<f32> = x.iter().map(|&x| -(x * x)).collect();
            let scale = if matches!(kind, U::Erf) {
                two_sqrtpi
            } else {
                -two_sqrtpi
            };
            zip(&exp_array(&neg_sq), &|e, g| e * scale * g)
        }
        _ => return None,
    })
}

// ---- array entry points ----------------------------------------------------------------------

/// Apply a 4-lane function the way PyTorch's vectorized loop does: groups of 4 from the start;
/// the last group is padded with zeros.
pub fn map4(xs: &[f32], f: fn(V) -> V) -> Vec<f32> {
    let mut out = Vec::with_capacity(xs.len());
    for chunk in xs.chunks(4) {
        let mut lanes = [0.0f32; 4];
        lanes[..chunk.len()].copy_from_slice(chunk);
        out.extend_from_slice(&f(V(lanes)).0[..chunk.len()]);
    }
    out
}

macro_rules! array_fns {
    ($($name:ident => $f:ident),* $(,)?) => {
        $(pub fn $name(xs: &[f32]) -> Vec<f32> { vml(xs, $f) })*
    };
}
array_fns!(
    exp_array => expf, expm1_array => expm1f, log_array => logf,
    log2_array => log2f, log10_array => log10f, log1p_array => log1pf, tanh_array => tanhf,
    atanh_array => atanhf, sin_array => sinf, cos_array => cosf,
    tan_array => tanf, asin_array => asinf, acos_array => acosf, atan_array => atanf,
);

/// `atan2_kernel` (cpu_kernel_vec; the scalar tail is the C library's atan2f). Both operands
/// have the same length.
pub fn atan2_array(y: &[f32], x: &[f32]) -> Vec<f32> {
    kernel_vec2(y, x, 32768, libm_atan2, atan2f)
}

/// One element through a 4-lane function (zero padded), for scalar call sites.
pub fn scalar(x: f32, f: fn(&[f32]) -> Vec<f32>) -> f32 {
    f(&[x])[0]
}
