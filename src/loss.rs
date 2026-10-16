use crate::tensor::TensorData;
/// Decode detached float32 class indices from a compiled-graph input.
/// The labels never participate in differentiation.
pub fn tensor_labels(logits_shape: &[usize], target: &TensorData) -> Result<Vec<usize>, String> {
    crate::dispatch::CrossEntropySpec::new(
        logits_shape,
        target.shape(),
        crate::dispatch::CrossEntropyMode::CompiledTensor,
    )?;
    target
        .values()
        .iter()
        .map(|&v| crate::dispatch::decode_class_index(v, logits_shape[1]))
        .collect()
}

fn dimensions(x: &TensorData, labels: &[usize]) -> Result<(usize, usize), String> {
    let spec = crate::dispatch::CrossEntropySpec::new(
        x.shape(),
        &[labels.len()],
        crate::dispatch::CrossEntropyMode::EagerIndices,
    )?;
    spec.validate_indices(labels)?;
    Ok((spec.batch, spec.classes))
}
pub fn cross_entropy(x: &TensorData, labels: &[usize]) -> Result<TensorData, String> {
    crate::dispatch::cross_entropy(x, labels)
}
pub(crate) fn cross_entropy_cpu(x: &TensorData, labels: &[usize]) -> Result<TensorData, String> {
    let (n, c) = dimensions(x, labels)?;
    let value = cross_entropy_values(x.values(), n, c, |i| Ok(labels[i]))?;
    TensorData::new(vec![value], vec![])
}
/// A label PyTorch's ignore_index skips (the row still takes part in log_softmax).
pub const IGNORED: usize = usize::MAX;

/// PyTorch's cross_entropy for class indices, mean reduction: log_softmax over the last axis,
/// then nll_loss_out_frame's 8-level cascade over the batch (aten/src/ATen/native/LossNLL.cpp,
/// PyTorch 2.14.1), divided by the number of rows not ignored.
fn cross_entropy_values(
    x: &[f32],
    n: usize,
    c: usize,
    mut label: impl FnMut(usize) -> Result<usize, String>,
) -> Result<f32, String> {
    if c == 0 || n.checked_mul(c) != Some(x.len()) {
        return Err("cross_entropy input length mismatch".into());
    }
    let logp = crate::torch_softmax::log_softmax_lastdim(x, c);
    const LEVELS: usize = 8;
    let level_power = 4usize.max(ceil_log2(n) / LEVELS);
    let level_mask = (1usize << level_power) - 1;
    let mut partial = [0.0f32; LEVELS];
    let mut ignored = 0usize;
    for b in 0..n {
        let target = label(b)?;
        if target == IGNORED {
            ignored += 1;
            continue;
        }
        if target >= c {
            return Err("cross_entropy class index out of range".into());
        }
        partial[0] -= logp[b * c + target];
        for j in 0..LEVELS - 1 {
            if b & (level_mask << (j * level_power)) != 0 {
                break;
            }
            partial[j + 1] += partial[j];
            partial[j] = 0.0;
        }
    }
    let total = partial.iter().fold(0.0f32, |acc, &v| acc + v);
    Ok(total / (n - ignored) as f32)
}
/// `utils::CeilLog2`.
fn ceil_log2(x: usize) -> usize {
    if x <= 2 {
        1
    } else {
        (usize::BITS - (x - 1).leading_zeros()) as usize
    }
}
pub(crate) fn cross_entropy_tensor_values(
    x: &[f32],
    labels: &[f32],
    shape: &[usize],
    label_shape: &[usize],
) -> Result<f32, String> {
    let spec = crate::dispatch::CrossEntropySpec::new(
        shape,
        label_shape,
        crate::dispatch::CrossEntropyMode::CompiledTensor,
    )?;
    if labels.len() != spec.batch {
        return Err("cross_entropy label length mismatch".into());
    }
    cross_entropy_values(x, spec.batch, spec.classes, |i| {
        crate::dispatch::decode_class_index(labels[i], spec.classes)
    })
}
/// nll_loss_backward (-(seed / total) at each target) through log_softmax's backward.
pub fn cross_entropy_grad(
    x: &TensorData,
    labels: &[usize],
    seed: f32,
) -> Result<TensorData, String> {
    let (n, c) = dimensions(x, labels)?;
    let logp = crate::torch_softmax::log_softmax_lastdim(x.values(), c);
    let kept = labels.iter().filter(|&&t| t != IGNORED).count();
    let grad = -(seed / kept as f32);
    let mut dlogp = vec![0.0f32; n * c];
    for (i, &t) in labels.iter().enumerate() {
        if t != IGNORED {
            dlogp[i * c + t] = grad;
        }
    }
    let values = crate::torch_softmax::log_softmax_backward_lastdim(&dlogp, &logp, c);
    TensorData::new(values, x.shape().to_vec())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uniform_classes() {
        let x = TensorData::new(vec![0.0; 4], vec![2, 2]).unwrap();
        assert!((cross_entropy(&x, &[0, 1]).unwrap().values()[0] - 2f32.ln()).abs() < 1e-6);
        assert_eq!(
            cross_entropy_grad(&x, &[0, 1], 1.0).unwrap().values(),
            &[-0.25, 0.25, 0.25, -0.25]
        );
        assert!(cross_entropy(&x, &[2, 0]).is_err());
    }
}

/// Compute mean squared error without allocating intermediate tensors.
pub fn mse_loss(prediction: &TensorData, target: &TensorData) -> Result<TensorData, String> {
    crate::dispatch::mse_loss(prediction, target)
}
pub(crate) fn mse_loss_cpu(
    prediction: &TensorData,
    target: &TensorData,
) -> Result<TensorData, String> {
    crate::dispatch::MseSpec::new(prediction.shape(), target.shape())?;
    TensorData::new(
        vec![mse_values(prediction.values(), target.values())?],
        vec![],
    )
}
pub(crate) fn mse_values(prediction: &[f32], target: &[f32]) -> Result<f32, String> {
    if prediction.len() != target.len() {
        return Err("MSE input and target lengths must match".into());
    }
    if prediction.is_empty() {
        return Err("MSE requires at least one element".into());
    }
    let mut total = 0.0f32;
    for (&x, &y) in prediction.iter().zip(target) {
        let diff = x - y;
        total += diff * diff;
    }
    Ok(total / prediction.len() as f32)
}

/// Compute the MSE gradient for one operand.
///
/// target_gradient=false: gradient with respect to prediction.
/// target_gradient=true: gradient with respect to target.
pub fn mse_loss_grad(
    prediction: &TensorData,
    target: &TensorData,
    seed: f32,
    target_gradient: bool,
) -> Result<TensorData, String> {
    let n = crate::dispatch::MseSpec::new(prediction.shape(), target.shape())?.elements;

    let scale = if target_gradient {
        -2.0 * seed / n as f32
    } else {
        2.0 * seed / n as f32
    };

    let values = prediction
        .values()
        .iter()
        .zip(target.values())
        .map(|(&x, &y)| (x - y) * scale)
        .collect();

    TensorData::new(values, prediction.shape().to_vec())
}

#[cfg(test)]
mod mse_tests {
    use super::*;

    #[test]
    fn fused_mse_forward_backward() {
        let x = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let y = TensorData::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();

        let loss = mse_loss(&x, &y).unwrap();

        assert!((loss.values()[0] - 1.0).abs() < 1e-6);

        let dx = mse_loss_grad(&x, &y, 1.0, false).unwrap();
        let dy = mse_loss_grad(&x, &y, 1.0, true).unwrap();

        assert_eq!(dx.values(), &[0.5, 0.5, 0.5, 0.5]);
        assert_eq!(dy.values(), &[-0.5, -0.5, -0.5, -0.5]);

        assert!(mse_loss(&x, &TensorData::new(vec![1.0], vec![1]).unwrap()).is_err());
    }

    #[test]
    fn fused_mse_gradient_seed() {
        let x = TensorData::new(vec![2.0], vec![1]).unwrap();
        let y = TensorData::new(vec![1.0], vec![1]).unwrap();

        let dx = mse_loss_grad(&x, &y, 3.0, false).unwrap();

        assert_eq!(dx.values(), &[6.0]);
    }
}
