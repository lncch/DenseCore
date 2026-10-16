//! Strict rank-three batched matrix multiplication, without batch broadcasting.
use crate::tensor::{DType, Device, TensorData, TensorLayout};

struct BmmSpec {
    k: usize,
    shape: Vec<usize>,
    count: usize,
}
impl BmmSpec {
    fn new(a: &[usize], b: &[usize]) -> Result<Self, String> {
        if a.len() != 3 || b.len() != 3 {
            return Err("bmm requires rank-3 operands".into());
        }
        if a[0] != b[0] || a[2] != b[1] {
            return Err("bmm batch or inner dimensions mismatch".into());
        }
        TensorLayout::count(a)?;
        TensorLayout::count(b)?;
        let shape = vec![a[0], a[1], b[2]];
        let count = TensorLayout::count(&shape)?;
        Ok(Self {
            k: a[2],
            shape,
            count,
        })
    }
}
struct BmmRegistration {
    device: Device,
    dtype: DType,
    forward: fn(&TensorData, &TensorData) -> Result<TensorData, String>,
}
static BMM: &[BmmRegistration] = &[BmmRegistration {
    device: Device::Cpu,
    dtype: DType::Float32,
    forward: cpu,
}];
pub fn bmm(a: &TensorData, b: &TensorData) -> Result<TensorData, String> {
    let entry = BMM
        .iter()
        .find(|r| {
            r.device == a.device()
                && r.device == b.device()
                && r.dtype == a.dtype()
                && r.dtype == b.dtype()
        })
        .ok_or("no registered bmm kernel for device/dtype")?;
    (entry.forward)(a, b)
}
fn cpu(a: &TensorData, b: &TensorData) -> Result<TensorData, String> {
    let spec = BmmSpec::new(a.shape(), b.shape())?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(spec.count)
        .map_err(|_| "bmm allocation failed")?;
    output.resize(spec.count, 0.0);
    // Zero batches can hide unreachable, overflowing matrix dimensions.
    if spec.count == 0 || spec.k == 0 {
        return TensorData::new(output, spec.shape);
    }
    // PyTorch's bmm (crate::torch_blas::bmm) on the operands' strided storage.
    fn view(t: &TensorData) -> crate::torch_blas::Batch<'_> {
        let (st, sh) = (t.strides(), t.shape());
        crate::torch_blas::Batch {
            data: t.storage(),
            offset: t.storage_offset(),
            sizes: [sh[0], sh[1], sh[2]],
            strides: [st[0], st[1], st[2]],
        }
    }
    output = crate::torch_blas::bmm(view(a), view(b))?;
    TensorData::new(output, spec.shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_batch_skips_unreachable_matrix_products() {
        let a = TensorData::new(vec![], vec![0, usize::MAX, 2]).unwrap();
        let b = TensorData::new(vec![], vec![0, 2, usize::MAX]).unwrap();
        assert_eq!(bmm(&a, &b).unwrap().shape(), &[0, usize::MAX, usize::MAX]);
    }
    #[test]
    fn checks_output_overflow() {
        assert!(BmmSpec::new(&[1, usize::MAX, 0], &[1, 0, 2]).is_err());
    }
}
