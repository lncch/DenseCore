//! Elementwise float32 binary math beyond + - * / (eager autograd only; the graph compiler
//! does not lower these). Semantics and gradients follow PyTorch's derivatives.yaml.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinaryMath {
    Pow,
    Maximum,
    Minimum,
    Fmax,
    Fmin,
    Atan2,
    Hypot,
    Fmod,
    Remainder,
    Copysign,
    Logaddexp,
    Logaddexp2,
    Xlogy,
}

const NAMES: &[(&str, BinaryMath)] = &[
    ("pow", BinaryMath::Pow),
    ("maximum", BinaryMath::Maximum),
    ("minimum", BinaryMath::Minimum),
    ("fmax", BinaryMath::Fmax),
    ("fmin", BinaryMath::Fmin),
    ("atan2", BinaryMath::Atan2),
    ("hypot", BinaryMath::Hypot),
    ("fmod", BinaryMath::Fmod),
    ("remainder", BinaryMath::Remainder),
    ("copysign", BinaryMath::Copysign),
    ("logaddexp", BinaryMath::Logaddexp),
    ("logaddexp2", BinaryMath::Logaddexp2),
    ("xlogy", BinaryMath::Xlogy),
];

/// Python-style remainder: the result takes the divisor's sign.
fn remainder(l: f32, r: f32) -> f32 {
    let m = l % r;
    if m != 0.0 && (m < 0.0) != (r < 0.0) {
        m + r
    } else {
        m
    }
}

fn logaddexp(l: f32, r: f32, base_log: f32, pow: fn(f32) -> f32) -> f32 {
    if l.is_infinite() && l == r {
        return l;
    }
    let m = l.max(r);
    m + (1.0 + pow(-(l - r).abs())).ln() / base_log
}

impl BinaryMath {
    pub fn by_name(name: &str) -> Result<Self, String> {
        NAMES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|&(_, op)| op)
            .ok_or_else(|| format!("unregistered binary operator: {name}"))
    }

    pub fn apply(self, l: f32, r: f32) -> f32 {
        match self {
            Self::Pow => l.powf(r),
            // NaN propagates, unlike f32::max/min.
            Self::Maximum => {
                if l.is_nan() || r.is_nan() {
                    f32::NAN
                } else if l > r {
                    l
                } else {
                    r
                }
            }
            Self::Minimum => {
                if l.is_nan() || r.is_nan() {
                    f32::NAN
                } else if l < r {
                    l
                } else {
                    r
                }
            }
            Self::Fmax => l.max(r),
            Self::Fmin => l.min(r),
            Self::Atan2 => l.atan2(r),
            Self::Hypot => l.hypot(r),
            Self::Fmod => l % r,
            Self::Remainder => remainder(l, r),
            Self::Copysign => l.copysign(r),
            Self::Logaddexp => logaddexp(l, r, 1.0, f32::exp),
            Self::Logaddexp2 => logaddexp(l, r, std::f32::consts::LN_2, f32::exp2),
            Self::Xlogy => {
                if l == 0.0 && !r.is_nan() {
                    0.0
                } else {
                    l * r.ln()
                }
            }
        }
    }

    /// d(out)/d(left) times the incoming gradient `s`.
    pub fn grad_left(self, l: f32, r: f32, s: f32) -> f32 {
        match self {
            Self::Pow => {
                if r == 0.0 {
                    0.0
                } else {
                    s * r * l.powf(r - 1.0)
                }
            }
            // Ties split the gradient in half and a NaN gives both sides all of it (PyTorch;
            // fmax/fmin share maximum/minimum's rule).
            Self::Maximum | Self::Fmax => tie_split(s, l == r, l < r),
            Self::Minimum | Self::Fmin => tie_split(s, l == r, l > r),
            Self::Atan2 => atan2_slope(s * r, l, r),
            Self::Hypot => s * l / l.hypot(r),
            Self::Fmod | Self::Remainder => s,
            Self::Copysign => {
                if l == 0.0 {
                    0.0
                } else {
                    s * (l.copysign(r) / l)
                }
            }
            Self::Logaddexp => s / (1.0 + (r - l).exp()),
            Self::Logaddexp2 => s / (1.0 + (r - l).exp2()),
            Self::Xlogy => {
                if l != 0.0 {
                    s * r.ln()
                } else if r.is_nan() {
                    f32::NAN
                } else {
                    0.0
                }
            }
        }
    }

    /// d(out)/d(right) times the incoming gradient `s`.
    pub fn grad_right(self, l: f32, r: f32, s: f32) -> f32 {
        match self {
            Self::Pow => {
                if l == 0.0 && r >= 0.0 {
                    0.0
                } else {
                    s * l.powf(r) * l.ln()
                }
            }
            Self::Maximum | Self::Fmax => tie_split(s, l == r, l > r),
            Self::Minimum | Self::Fmin => tie_split(s, l == r, l < r),
            Self::Atan2 => atan2_slope(s * -l, l, r),
            Self::Hypot => s * r / l.hypot(r),
            Self::Fmod => -s * (l / r).trunc(),
            Self::Remainder => -s * (l / r).floor(),
            Self::Copysign => 0.0,
            Self::Logaddexp => s / (1.0 + (l - r).exp()),
            Self::Logaddexp2 => s / (1.0 + (l - r).exp2()),
            Self::Xlogy => s * l / r,
        }
    }
}

/// numerator / (l^2 + r^2), with PyTorch's 0 at the origin instead of 0/0.
fn atan2_slope(numerator: f32, l: f32, r: f32) -> f32 {
    // PyTorch: recip = (self * self + other * other).reciprocal(), masked to 0 where the
    // denominator is 0, then (grad * other) * recip and (grad * -self) * recip.
    let denominator = l * l + r * r;
    let recip = if denominator == 0.0 {
        0.0
    } else {
        1.0 / denominator
    };
    numerator * recip
}

/// Gradient for one side of maximum/minimum: half on ties, none where the other side wins.
fn tie_split(s: f32, tie: bool, loses: bool) -> f32 {
    if tie {
        s / 2.0
    } else if loses {
        0.0
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn names_round_trip_and_special_values() {
        for &(name, op) in NAMES {
            assert_eq!(BinaryMath::by_name(name).unwrap(), op);
        }
        assert!(BinaryMath::by_name("nope").is_err());
        assert!(BinaryMath::Maximum.apply(f32::NAN, 1.0).is_nan());
        assert_eq!(BinaryMath::Fmax.apply(f32::NAN, 1.0), 1.0);
        assert_eq!(BinaryMath::Remainder.apply(-7.0, 3.0), 2.0);
        assert_eq!(BinaryMath::Fmod.apply(-7.0, 3.0), -1.0);
        assert_eq!(BinaryMath::Xlogy.apply(0.0, 0.0), 0.0);
        assert_eq!(
            BinaryMath::Logaddexp.apply(f32::INFINITY, f32::INFINITY),
            f32::INFINITY
        );
        assert_eq!(BinaryMath::Maximum.grad_left(2.0, 2.0, 1.0), 0.5);
    }
}
