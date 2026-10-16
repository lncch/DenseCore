//! Storage-only handles for dtypes beyond float32/bool: views work, math does not yet.
//! Values cross the Python boundary as little-endian bytes in NumPy's layout.
use crate::tensor::{BF16, C64, C128, F16, SliceSpec, TypedTensorData};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

trait Raw: crate::tensor::Element {
    const SIZE: usize;
    fn read(bytes: &[u8]) -> Self;
    fn write(self, out: &mut Vec<u8>);
}
macro_rules! raw {
    ($($t:ty, $size:expr, |$b:ident| $read:expr, |$v:ident| $write:expr);* $(;)?) => {$(
        impl Raw for $t {
            const SIZE: usize = $size;
            fn read($b: &[u8]) -> Self {
                $read
            }
            fn write(self, out: &mut Vec<u8>) {
                let $v = self;
                out.extend_from_slice(&$write);
            }
        }
    )*};
}
raw! {
    f64, 8, |b| f64::from_le_bytes(b.try_into().unwrap()), |v| v.to_le_bytes();
    i8, 1, |b| i8::from_le_bytes([b[0]]), |v| v.to_le_bytes();
    i16, 2, |b| i16::from_le_bytes(b.try_into().unwrap()), |v| v.to_le_bytes();
    i32, 4, |b| i32::from_le_bytes(b.try_into().unwrap()), |v| v.to_le_bytes();
    i64, 8, |b| i64::from_le_bytes(b.try_into().unwrap()), |v| v.to_le_bytes();
    u8, 1, |b| b[0], |v| [v];
    F16, 2, |b| F16(u16::from_le_bytes(b.try_into().unwrap())), |v| v.0.to_le_bytes();
    BF16, 2, |b| BF16(u16::from_le_bytes(b.try_into().unwrap())), |v| v.0.to_le_bytes();
    C64, 8, |b| C64(f32::from_le_bytes(b[..4].try_into().unwrap()), f32::from_le_bytes(b[4..].try_into().unwrap())),
        |v| { let mut out = [0u8; 8]; out[..4].copy_from_slice(&v.0.to_le_bytes()); out[4..].copy_from_slice(&v.1.to_le_bytes()); out };
    C128, 16, |b| C128(f64::from_le_bytes(b[..8].try_into().unwrap()), f64::from_le_bytes(b[8..].try_into().unwrap())),
        |v| { let mut out = [0u8; 16]; out[..8].copy_from_slice(&v.0.to_le_bytes()); out[8..].copy_from_slice(&v.1.to_le_bytes()); out };
}

fn decode<T: Raw>(bytes: &[u8], shape: Vec<usize>) -> Result<TypedTensorData<T>, String> {
    if !bytes.len().is_multiple_of(T::SIZE) {
        return Err("byte length is not a multiple of the element size".into());
    }
    TypedTensorData::new(bytes.chunks_exact(T::SIZE).map(T::read).collect(), shape)
}
fn encode<T: Raw>(t: &TypedTensorData<T>) -> Vec<u8> {
    let mut out = Vec::with_capacity(t.numel() * T::SIZE);
    for &v in t.values() {
        v.write(&mut out);
    }
    out
}

macro_rules! typed {
    ($($v:ident($t:ty) = $name:literal),*) => {
        #[derive(Clone)]
        enum Typed {
            $($v(TypedTensorData<$t>)),*
        }
        impl Typed {
            fn decode(bytes: &[u8], shape: Vec<usize>, dtype: &str) -> Result<Self, String> {
                match dtype {
                    $($name => decode(bytes, shape).map(Self::$v),)*
                    _ => Err(format!("unsupported storage dtype {dtype:?}")),
                }
            }
        }
        /// Run `$e` on the inner data whatever its element type.
        macro_rules! with {
            ($s:expr, $x:ident => $e:expr) => { match $s { $(Typed::$v($x) => $e),* } };
        }
        /// Run a view-producing `$e` and keep the variant.
        macro_rules! rewrap {
            ($s:expr, $x:ident => $e:expr) => { match $s { $(Typed::$v($x) => Typed::$v($e)),* } };
        }
        fn same_storage(a: &Typed, b: &Typed) -> bool {
            match (a, b) {
                $((Typed::$v(x), Typed::$v(y)) => x.shares_storage(y),)*
                _ => false,
            }
        }
    };
}
typed!(
    F64(f64) = "float64",
    F16(F16) = "float16",
    BF16(BF16) = "bfloat16",
    I8(i8) = "int8",
    I16(i16) = "int16",
    I32(i32) = "int32",
    I64(i64) = "int64",
    U8(u8) = "uint8",
    C64(C64) = "complex64",
    C128(C128) = "complex128"
);

#[pyclass(name = "_TypedTensor", frozen)]
pub(crate) struct NativeTypedTensor {
    inner: Typed,
}
impl NativeTypedTensor {
    fn view(&self, inner: Result<Typed, String>) -> PyResult<Self> {
        Ok(Self {
            inner: inner.map_err(PyValueError::new_err)?,
        })
    }
}
#[pymethods]
impl NativeTypedTensor {
    #[new]
    fn new(data: &[u8], shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Typed::decode(data, shape, dtype).map_err(PyValueError::new_err)?,
        })
    }
    /// Logical row-major values as little-endian bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &with!(&self.inner, t => encode(t)))
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        with!(&self.inner, t => t.dtype().name())
    }
    #[getter]
    fn device(&self) -> &'static str {
        with!(&self.inner, t => t.device().name())
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        with!(&self.inner, t => t.shape().to_vec())
    }
    #[getter]
    fn strides(&self) -> Vec<usize> {
        with!(&self.inner, t => t.strides().to_vec())
    }
    #[getter]
    fn storage_offset(&self) -> usize {
        with!(&self.inner, t => t.storage_offset())
    }
    #[getter]
    fn requires_grad(&self) -> bool {
        false
    }
    #[getter]
    fn is_leaf(&self) -> bool {
        true
    }
    #[getter]
    fn grad(&self) -> Option<Self> {
        None
    }
    fn numel(&self) -> usize {
        with!(&self.inner, t => t.numel())
    }
    fn is_contiguous(&self) -> bool {
        with!(&self.inner, t => t.is_contiguous())
    }
    fn shares_storage(&self, other: &Self) -> bool {
        same_storage(&self.inner, &other.inner)
    }
    fn detach(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.view((|| Ok(rewrap!(&self.inner, t => t.reshape(shape)?)))())
    }
    fn permute(&self, axes: Vec<usize>) -> PyResult<Self> {
        self.view((|| Ok(rewrap!(&self.inner, t => t.permute(&axes)?)))())
    }
    fn transpose(&self) -> PyResult<Self> {
        if self.shape().len() != 2 {
            return Err(PyValueError::new_err("transpose requires rank 2"));
        }
        self.permute(vec![1, 0])
    }
    fn slice(&self, specs: Vec<(usize, usize, usize, bool)>) -> PyResult<Self> {
        let specs: Vec<_> = specs
            .into_iter()
            .map(|(start, step, len, squeeze)| SliceSpec {
                start,
                step,
                len,
                squeeze,
            })
            .collect();
        self.view((|| Ok(rewrap!(&self.inner, t => t.slice(&specs)?)))())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bytes_round_trip_through_strided_views_for_every_dtype() {
        let shape = vec![2, 3];
        for (dtype, size) in [
            ("float64", 8),
            ("float16", 2),
            ("bfloat16", 2),
            ("int8", 1),
            ("int16", 2),
            ("int32", 4),
            ("int64", 8),
            ("uint8", 1),
            ("complex64", 8),
            ("complex128", 16),
        ] {
            let bytes: Vec<u8> = (0..6 * size).map(|i| (i * 37 % 251) as u8).collect();
            let t = Typed::decode(&bytes, shape.clone(), dtype).unwrap();
            assert_eq!(with!(&t, x => encode(x)), bytes);
            let back = rewrap!(&t, x => x.permute(&[1, 0]).unwrap());
            let transposed = rewrap!(&back, x => x.permute(&[1, 0]).unwrap());
            assert_eq!(with!(&transposed, x => encode(x)), bytes);
            let permuted = with!(&back, x => encode(x));
            for (j, chunk) in permuted.chunks(size).enumerate() {
                let (col, row) = (j / 2, j % 2);
                let source = (row * 3 + col) * size;
                assert_eq!(chunk, &bytes[source..source + size]);
            }
            assert!(Typed::decode(&bytes[1..], shape.clone(), dtype).is_err());
        }
        assert!(Typed::decode(&[], vec![0], "complex32").is_err());
    }
}
