//! Compact stride-based broadcasting for arbitrary NumPy-compatible shapes.
#[derive(Clone)]
pub struct CompiledBroadcast {
    shape: Vec<usize>,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
    len: usize,
}
impl CompiledBroadcast {
    pub fn new(a: &[usize], b: &[usize]) -> Result<Self, String> {
        let shape = crate::ops::broadcast_shape(a, b)?;
        let len = crate::tensor::TensorLayout::count(&shape)?;
        if len == 0 {
            // No logical address is reachable, so physical index strides are unused.
            return Ok(Self {
                shape,
                left_strides: Vec::new(),
                right_strides: Vec::new(),
                len,
            });
        }
        let left_strides = crate::ops::broadcast_strides(a, &shape)?;
        let right_strides = crate::ops::broadcast_strides(b, &shape)?;
        Ok(Self {
            shape,
            left_strides,
            right_strides,
            len,
        })
    }
    pub fn forward(&self, a: &[f32], b: &[f32], out: &mut [f32], op: crate::ops::BinaryOp) {
        self.for_each_index(|i, ai, bi| {
            out[i] = match op {
                crate::ops::BinaryOp::Sub => a[ai] - b[bi],
                crate::ops::BinaryOp::Div => a[ai] / b[bi],
                crate::ops::BinaryOp::Add => a[ai] + b[bi],
                crate::ops::BinaryOp::Mul => a[ai] * b[bi],
            };
        });
    }
    pub fn add_gradient(&self, left_side: bool, g: &[f32], target: &mut [f32]) {
        self.for_each_index(|i, ai, bi| {
            target[if left_side { ai } else { bi }] += g[i];
        });
    }
    pub fn mul_gradient(&self, left_side: bool, g: &[f32], target: &mut [f32], other: &[f32]) {
        self.for_each_index(|i, ai, bi| {
            let (target_index, other_index) = if left_side { (ai, bi) } else { (bi, ai) };
            target[target_index] += g[i] * other[other_index];
        });
    }
    pub fn sub_gradient(&self, left_side: bool, g: &[f32], target: &mut [f32]) {
        self.for_each_index(|i, ai, bi| {
            target[if left_side { ai } else { bi }] += if left_side { g[i] } else { -g[i] };
        });
    }
    pub fn div_gradient(
        &self,
        left_side: bool,
        g: &[f32],
        target: &mut [f32],
        a: &[f32],
        b: &[f32],
    ) {
        self.for_each_index(|i, ai, bi| {
            target[if left_side { ai } else { bi }] += if left_side {
                g[i] / b[bi]
            } else {
                -(g[i] * (a[ai] / b[bi])) / b[bi]
            };
        });
    }
    /// Iterate output and corresponding input indices without output-sized maps.
    pub fn for_each_index(&self, mut emit: impl FnMut(usize, usize, usize)) {
        for flat in 0..self.len {
            let mut remaining = flat;
            let mut ai = 0usize;
            let mut bi = 0usize;
            for axis in (0..self.shape.len()).rev() {
                let coordinate = remaining % self.shape[axis];
                remaining /= self.shape[axis];
                ai += coordinate * self.left_strides[axis];
                bi += coordinate * self.right_strides[axis];
            }
            emit(flat, ai, bi);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contiguous_forward_and_gradients_match_map_oracle_with_existing_gradients() {
        let shapes = [
            vec![],
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![0, 0],
            vec![0, 2],
            vec![2, 0],
            vec![1, 2],
            vec![2, 2],
            vec![3, 2],
            vec![2, 3],
            vec![3, 3],
            vec![2, 1, 3],
            vec![1, 4, 1],
            vec![1, 0, 3],
            vec![1, 1, 3],
        ];
        for a_shape in &shapes {
            for b_shape in &shapes {
                let Ok(maps) = crate::ops::broadcast_plan(a_shape, b_shape) else {
                    continue;
                };
                let descriptor = CompiledBroadcast::new(a_shape, b_shape).unwrap();
                let a: Vec<f32> = (0..a_shape.iter().product())
                    .map(|i| i as f32 * 0.2 - 0.3)
                    .collect();
                let b: Vec<f32> = (0..b_shape.iter().product())
                    .map(|i| i as f32 * 0.3 + 0.1)
                    .collect();
                let g: Vec<f32> = (0..maps.left.len())
                    .map(|i| i as f32 * 0.13 - 0.2)
                    .collect();
                for op in [
                    crate::ops::BinaryOp::Add,
                    crate::ops::BinaryOp::Mul,
                    crate::ops::BinaryOp::Sub,
                    crate::ops::BinaryOp::Div,
                ] {
                    let expected = crate::ops::binary(
                        &crate::tensor::TensorData::new(a.clone(), a_shape.clone()).unwrap(),
                        &crate::tensor::TensorData::new(b.clone(), b_shape.clone()).unwrap(),
                        op,
                    )
                    .unwrap();
                    let mut got = vec![0.; g.len()];
                    descriptor.forward(&a, &b, &mut got, op);
                    assert_eq!(got, expected.values());
                    for left_side in [true, false] {
                        let mut want = vec![0.25; if left_side { a.len() } else { b.len() }];
                        let mut got = want.clone();
                        for ((&ai, &bi), &seed) in maps.left.iter().zip(&maps.right).zip(&g) {
                            let index = if left_side { ai } else { bi };
                            want[index] += match op {
                                crate::ops::BinaryOp::Sub => {
                                    if left_side {
                                        seed
                                    } else {
                                        -seed
                                    }
                                }
                                crate::ops::BinaryOp::Div => {
                                    if left_side {
                                        seed / b[bi]
                                    } else {
                                        -(seed * (a[ai] / b[bi])) / b[bi]
                                    }
                                }
                                crate::ops::BinaryOp::Add => seed,
                                crate::ops::BinaryOp::Mul => {
                                    seed * if left_side { b[bi] } else { a[ai] }
                                }
                            };
                        }
                        match op {
                            crate::ops::BinaryOp::Sub => {
                                descriptor.sub_gradient(left_side, &g, &mut got)
                            }
                            crate::ops::BinaryOp::Div => {
                                descriptor.div_gradient(left_side, &g, &mut got, &a, &b)
                            }
                            crate::ops::BinaryOp::Add => {
                                descriptor.add_gradient(left_side, &g, &mut got)
                            }
                            crate::ops::BinaryOp::Mul => descriptor.mul_gradient(
                                left_side,
                                &g,
                                &mut got,
                                if left_side { &b } else { &a },
                            ),
                        }
                        for (&got, &want) in got.iter().zip(&want) {
                            assert!((got - want).abs() <= 1e-6 + 1e-5 * want.abs());
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn compact_indices_match_eager_maps_across_supported_shapes() {
        let shapes = [
            vec![],
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![0, 0],
            vec![0, 2],
            vec![2, 0],
            vec![1, 2],
            vec![2, 2],
            vec![3, 2],
            vec![2, 3],
            vec![3, 3],
            vec![2, 1, 3],
            vec![1, 4, 1],
            vec![1, 0, 3],
            vec![1, 1, 3],
        ];
        for a in &shapes {
            for b in &shapes {
                let expected = crate::ops::broadcast_plan(a, b);
                let compiled = CompiledBroadcast::new(a, b);
                match (expected, compiled) {
                    (Ok(expected), Ok(compiled)) => {
                        let mut indices = Vec::new();
                        compiled.for_each_index(|i, a, b| indices.push((i, a, b)));
                        assert_eq!(
                            indices,
                            expected
                                .left
                                .into_iter()
                                .zip(expected.right)
                                .enumerate()
                                .map(|(i, (a, b))| (i, a, b))
                                .collect::<Vec<_>>()
                        );
                    }
                    (Err(_), Err(_)) => {}
                    _ => panic!("broadcast support mismatch: {a:?}, {b:?}"),
                }
            }
        }
        assert!(CompiledBroadcast::new(&[usize::MAX, 2], &[]).is_err());
    }
}
