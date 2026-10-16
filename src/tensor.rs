#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
}
impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Float32,
    Bool,
    Float64,
    Float16,
    BFloat16,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    Complex64,
    Complex128,
}
impl DType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Bool => "bool",
            Self::Float64 => "float64",
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt8 => "uint8",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
        }
    }
}
/// IEEE half precision, stored as raw bits; conversions happen outside Rust for now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F16(pub u16);
/// bfloat16 (the upper half of a float32), stored as raw bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BF16(pub u16);
/// complex64: (real, imaginary) float32 pair, NumPy's memory layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C64(pub f32, pub f32);
/// complex128: (real, imaginary) float64 pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C128(pub f64, pub f64);

use std::sync::{Arc, OnceLock};
#[cfg(test)]
use std::{any::Any, cell::RefCell, sync::Weak};

#[cfg(test)]
struct TrackedTensorStorage {
    identity: usize,
    capacity_bytes: usize,
    storage: Weak<dyn Any + Send + Sync>,
}

#[cfg(test)]
#[derive(Default)]
struct TensorStorageTrackerState {
    storages: Vec<TrackedTensorStorage>,
    peak_capacity_bytes: usize,
}

#[cfg(test)]
thread_local! {
    static TENSOR_STORAGE_TRACKER: RefCell<Option<TensorStorageTrackerState>> = const { RefCell::new(None) };
}

#[cfg(test)]
pub(crate) struct TensorStorageTracker;

#[cfg(test)]
impl TensorStorageTracker {
    pub(crate) fn start() -> Self {
        TENSOR_STORAGE_TRACKER.with(|tracker| {
            let mut tracker = tracker.borrow_mut();
            assert!(tracker.is_none(), "tensor storage tracker already active");
            *tracker = Some(TensorStorageTrackerState::default());
        });
        Self
    }

    pub(crate) fn live_capacity_bytes(&self) -> usize {
        TENSOR_STORAGE_TRACKER.with(|tracker| {
            let mut tracker = tracker.borrow_mut();
            let state = tracker.as_mut().expect("tensor storage tracker is active");
            state
                .storages
                .retain(|record| record.storage.strong_count() > 0);
            let live = state
                .storages
                .iter()
                .filter_map(|record| record.storage.upgrade().map(|_| record.capacity_bytes))
                .sum();
            state.peak_capacity_bytes = state.peak_capacity_bytes.max(live);
            live
        })
    }

    pub(crate) fn peak_capacity_bytes(&self) -> usize {
        self.live_capacity_bytes();
        TENSOR_STORAGE_TRACKER.with(|tracker| {
            tracker
                .borrow()
                .as_ref()
                .expect("tensor storage tracker is active")
                .peak_capacity_bytes
        })
    }

    pub(crate) fn reset_peak(&self) {
        let live = self.live_capacity_bytes();
        TENSOR_STORAGE_TRACKER.with(|tracker| {
            tracker
                .borrow_mut()
                .as_mut()
                .expect("tensor storage tracker is active")
                .peak_capacity_bytes = live;
        });
    }
}

#[cfg(test)]
impl Drop for TensorStorageTracker {
    fn drop(&mut self) {
        TENSOR_STORAGE_TRACKER.with(|tracker| *tracker.borrow_mut() = None);
    }
}

#[cfg(test)]
fn track_tensor_storage<T: Element>(storage: &Arc<Vec<T>>) {
    TENSOR_STORAGE_TRACKER.with(|tracker| {
        let mut tracker = tracker.borrow_mut();
        let Some(state) = tracker.as_mut() else {
            return;
        };
        let identity = Arc::as_ptr(storage) as usize;
        if state
            .storages
            .iter()
            .any(|record| record.identity == identity)
        {
            return;
        }
        let erased: Arc<dyn Any + Send + Sync> = storage.clone();
        state.storages.push(TrackedTensorStorage {
            identity,
            capacity_bytes: storage.capacity() * std::mem::size_of::<T>(),
            storage: Arc::downgrade(&erased),
        });
        let live = state
            .storages
            .iter()
            .filter_map(|record| record.storage.upgrade().map(|_| record.capacity_bytes))
            .sum();
        state.peak_capacity_bytes = state.peak_capacity_bytes.max(live);
    });
}

/// Logical compilation contract. Layout is deliberately excluded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    shape: Vec<usize>,
    device: Device,
    dtype: DType,
}
impl TensorSpec {
    pub fn new(shape: Vec<usize>, device: Device, dtype: DType) -> Result<Self, String> {
        TensorLayout::count(&shape)?;
        Ok(Self {
            shape,
            device,
            dtype,
        })
    }
    pub fn from_tensor<T: Element>(tensor: &TypedTensorData<T>) -> Self {
        Self {
            shape: tensor.shape().to_vec(),
            device: tensor.device(),
            dtype: tensor.dtype(),
        }
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn matches<T: Element>(&self, tensor: &TypedTensorData<T>) -> bool {
        self.shape == tensor.shape()
            && self.device == tensor.device()
            && self.dtype == tensor.dtype()
    }
    pub fn check<T: Element>(
        &self,
        tensor: &TypedTensorData<T>,
        label: &str,
    ) -> Result<(), String> {
        if self.shape != tensor.shape() {
            return Err(format!("{label} shape mismatch"));
        }
        if self.device != tensor.device() {
            return Err(format!("{label} device mismatch"));
        }
        if self.dtype != tensor.dtype() {
            return Err(format!("{label} dtype mismatch"));
        }
        Ok(())
    }
}

/// Immutable backing allocation. Replacements create new storage, never mutate aliases.
#[derive(Clone)]
pub struct TensorStorage<T: Element>(Arc<Vec<T>>);

/// Element strides and offset refer to storage, independently of tensor identity.
#[derive(Clone, Debug)]
pub struct TensorLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    count: usize,
    contiguous: bool,
}
impl TensorLayout {
    pub(crate) fn count(shape: &[usize]) -> Result<usize, String> {
        if shape.contains(&0) {
            return Ok(0);
        }
        shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| format!("shape product overflow: {shape:?}"))
    }
    fn contiguous(shape: Vec<usize>) -> Result<Self, String> {
        let count = Self::count(&shape)?;
        let mut strides = vec![0; shape.len()];
        let mut stride = 1usize;
        for axis in (0..shape.len()).rev() {
            strides[axis] = stride;
            stride = if count == 0 {
                // Empty tensors have no reachable addresses; saturated strides
                // preserve representable metadata without rejecting their shape.
                stride.saturating_mul(shape[axis].max(1))
            } else {
                stride
                    .checked_mul(shape[axis].max(1))
                    .ok_or("stride overflow")?
            };
        }
        Ok(Self {
            shape,
            strides,
            offset: 0,
            count,
            contiguous: true,
        })
    }
    fn checked(
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
        storage_len: usize,
    ) -> Result<Self, String> {
        if shape.len() != strides.len() {
            return Err("shape and stride rank mismatch".into());
        }
        let count = Self::count(&shape)?;
        if count == 0 {
            if offset > storage_len {
                return Err("empty layout offset exceeds storage".into());
            }
        } else {
            let last = shape
                .iter()
                .zip(&strides)
                .try_fold(offset, |n, (&d, &s)| n.checked_add((d - 1).checked_mul(s)?))
                .ok_or("layout address overflow")?;
            if last >= storage_len {
                return Err("layout exceeds storage bounds".into());
            }
        }
        let mut layout = Self {
            shape,
            strides,
            offset,
            count,
            contiguous: false,
        };
        layout.contiguous = layout.is_contiguous();
        Ok(layout)
    }
    fn is_contiguous(&self) -> bool {
        if self.count == 0 {
            return true;
        }
        let mut expected = 1usize;
        for (&dim, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if dim != 1 && stride != expected {
                return false;
            }
            // The shape product was checked at construction.
            expected *= dim;
        }
        true
    }
    /// Row-major gather: a stride walk on the last axis plus an odometer over the
    /// outer axes, so no element pays a divide/modulo per axis.
    fn gather<T: Copy>(&self, storage: &[T]) -> Vec<T> {
        if self.count > 0 && self.strides.last() == Some(&1) {
            // Unit-stride rows (a transposed batch of rows): copy each row whole.
            let inner = self.shape[self.shape.len() - 1];
            let mut out = Vec::with_capacity(self.count);
            self.for_each_row(|base| out.extend_from_slice(&storage[base..base + inner]));
            return out;
        }
        self.gather_with(|address| storage[address])
    }
    /// Map every logical element's storage address, in row-major order.
    fn gather_with<T>(&self, get: impl Fn(usize) -> T) -> Vec<T> {
        let mut out = Vec::with_capacity(self.count);
        let Some((&inner, outer)) = self.shape.split_last() else {
            if self.count == 1 {
                out.push(get(self.offset));
            }
            return out;
        };
        if self.count == 0 {
            return out;
        }
        let step = self.strides[outer.len()];
        self.for_each_row(|base| out.extend((0..inner).map(|i| get(base + i * step))));
        out
    }
    /// Visit the storage address of each last-axis row's first element, in row-major order.
    /// Needs rank >= 1 and a nonzero count.
    fn for_each_row(&self, mut visit: impl FnMut(usize)) {
        let outer = &self.shape[..self.shape.len() - 1];
        let mut index = vec![0; outer.len()];
        let mut base = self.offset;
        loop {
            visit(base);
            let mut axis = outer.len();
            loop {
                if axis == 0 {
                    return;
                }
                axis -= 1;
                index[axis] += 1;
                base += self.strides[axis];
                if index[axis] < outer[axis] {
                    break;
                }
                base -= self.strides[axis] * outer[axis];
                index[axis] = 0;
            }
        }
    }
    #[cfg(test)]
    fn address(&self, mut flat: usize) -> usize {
        let mut index = self.offset;
        for (&dim, &stride) in self.shape.iter().zip(&self.strides).rev() {
            index += (flat % dim) * stride;
            flat /= dim;
        }
        index
    }
}

/// `at::detail::computeStride` (PyTorch 2.14.1, aten/src/ATen/TensorUtils.cpp): the strides
/// that view `old_shape`/`old_strides` as `new_shape` without copying, if any.
fn compute_stride(
    old_shape: &[usize],
    old_strides: &[usize],
    new_shape: &[usize],
) -> Option<Vec<usize>> {
    if old_shape.is_empty() {
        return Some(vec![1; new_shape.len()]);
    }
    let numel: usize = old_shape.iter().product();
    if numel == 0 && old_shape == new_shape {
        return Some(old_strides.to_vec());
    }
    let mut new_strides = vec![0usize; new_shape.len()];
    if numel == 0 {
        for d in (0..new_shape.len()).rev() {
            new_strides[d] = if d + 1 == new_shape.len() {
                1
            } else {
                new_shape[d + 1].max(1) * new_strides[d + 1]
            };
        }
        return Some(new_strides);
    }
    let mut view_d = new_shape.len() as isize - 1;
    let mut chunk_base_stride = *old_strides.last()?;
    let (mut tensor_numel, mut view_numel) = (1usize, 1usize);
    for tensor_d in (0..old_shape.len()).rev() {
        tensor_numel *= old_shape[tensor_d];
        if tensor_d == 0
            || (old_shape[tensor_d - 1] != 1
                && old_strides[tensor_d - 1] != tensor_numel * chunk_base_stride)
        {
            while view_d >= 0 && (view_numel < tensor_numel || new_shape[view_d as usize] == 1) {
                new_strides[view_d as usize] = view_numel * chunk_base_stride;
                view_numel *= new_shape[view_d as usize];
                view_d -= 1;
            }
            if view_numel != tensor_numel {
                return None;
            }
            if tensor_d > 0 {
                chunk_base_stride = old_strides[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    (view_d == -1).then_some(new_strides)
}

/// A normalized positive-step basic index for one parent axis.
#[derive(Clone)]
pub struct SliceSpec {
    pub start: usize,
    pub step: usize,
    pub len: usize,
    pub squeeze: bool,
}

mod sealed {
    pub trait Sealed {}
}
pub trait Element: sealed::Sealed + Copy + Send + Sync + 'static {
    const DTYPE: DType;
}
macro_rules! element {
    ($($t:ty => $d:ident),*) => {$(
        impl sealed::Sealed for $t {}
        impl Element for $t {
            const DTYPE: DType = DType::$d;
        }
    )*};
}
element!(f32 => Float32, bool => Bool, f64 => Float64, F16 => Float16, BF16 => BFloat16,
    i8 => Int8, i16 => Int16, i32 => Int32, i64 => Int64, u8 => UInt8,
    C64 => Complex64, C128 => Complex128);
pub type TensorData = TypedTensorData<f32>;
pub type BoolTensorData = TypedTensorData<bool>;

#[derive(Clone)]
pub struct TypedTensorData<T: Element> {
    storage: TensorStorage<T>,
    layout: TensorLayout,
    logical: Arc<OnceLock<Vec<T>>>,
}
impl<T: Element> TypedTensorData<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, String> {
        let layout = TensorLayout::contiguous(shape)?;
        if layout.count != data.len() {
            return Err(format!(
                "shape {:?} requires {} elements, got {}",
                layout.shape,
                layout.count,
                data.len()
            ));
        }
        let storage = Arc::new(data);
        #[cfg(test)]
        track_tensor_storage(&storage);
        Ok(Self {
            storage: TensorStorage(storage),
            layout,
            logical: Arc::new(OnceLock::new()),
        })
    }
    fn view(&self, layout: TensorLayout) -> Self {
        Self {
            storage: self.storage.clone(),
            layout,
            logical: Arc::new(OnceLock::new()),
        }
    }
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }
    pub fn device(&self) -> Device {
        Device::Cpu
    }
    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.layout.strides
    }
    pub fn storage_offset(&self) -> usize {
        self.layout.offset
    }
    /// The whole backing buffer (index with `storage_offset` and `strides`).
    pub(crate) fn storage(&self) -> &[T] {
        &self.storage.0
    }
    pub fn numel(&self) -> usize {
        self.layout.count
    }
    pub fn is_contiguous(&self) -> bool {
        self.layout.contiguous
    }
    pub fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage.0, &other.storage.0)
    }
    #[cfg(test)]
    pub(crate) fn storage_capacity_bytes(&self) -> usize {
        self.storage.0.capacity() * std::mem::size_of::<T>()
    }
    #[cfg(test)]
    pub(crate) fn storage_identity(&self) -> usize {
        Arc::as_ptr(&self.storage.0) as usize
    }
    /// Legacy kernels receive logical row-major values, never raw strided storage.
    /// A strided view caches one materialization, shared by identical clones.
    pub fn values(&self) -> &[T] {
        if self.is_contiguous() {
            return &self.storage.0[self.layout.offset..self.layout.offset + self.numel()];
        }
        self.logical
            .get_or_init(|| self.layout.gather(&self.storage.0))
    }
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, String> {
        let mut layout = TensorLayout::contiguous(shape)?;
        if layout.count != self.numel() {
            return Err("reshape element count mismatch".into());
        }
        if self.is_contiguous() {
            layout.offset = self.layout.offset;
            return Ok(self.view(layout));
        }
        // As PyTorch's reshape: a view whenever the strides allow one, else a copy.
        if let Some(strides) = compute_stride(self.shape(), self.strides(), &layout.shape) {
            let view = TensorLayout::checked(
                layout.shape.clone(),
                strides,
                self.layout.offset,
                self.storage.0.len(),
            )?;
            return Ok(self.view(view));
        }
        Self::new(self.values().to_vec(), layout.shape)
    }
    pub fn slice(&self, specs: &[SliceSpec]) -> Result<Self, String> {
        if specs.len() != self.shape().len() {
            return Err("slice rank mismatch".into());
        }
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        let empty = specs.iter().any(|s| s.len == 0);
        let mut offset = self.storage_offset();
        for (axis, s) in specs.iter().enumerate() {
            let dim = self.shape()[axis];
            if s.step == 0 || (s.squeeze && s.len != 1) {
                return Err("invalid slice step or integer index".into());
            }
            if s.len == 0 {
                if s.start > dim {
                    return Err("slice start out of bounds".into());
                }
            } else {
                let last = (s.len - 1)
                    .checked_mul(s.step)
                    .and_then(|v| s.start.checked_add(v))
                    .ok_or("slice index overflow")?;
                if last >= dim {
                    return Err("slice index out of bounds".into());
                }
            }
            if !empty {
                offset = s
                    .start
                    .checked_mul(self.strides()[axis])
                    .and_then(|v| offset.checked_add(v))
                    .ok_or("slice offset overflow")?;
            }
            if !s.squeeze {
                shape.push(s.len);
                strides.push(if empty {
                    self.strides()[axis].saturating_mul(s.step)
                } else {
                    self.strides()[axis]
                        .checked_mul(s.step)
                        .ok_or("slice stride overflow")?
                });
            }
        }
        let layout = TensorLayout::checked(shape, strides, offset, self.storage.0.len())?;
        Ok(self.view(layout))
    }
    /// Parent indices are logical row-major indices, not backing storage addresses.
    pub fn slice_indices(&self, specs: &[SliceSpec]) -> Result<Vec<usize>, String> {
        let view = self.slice(specs)?;
        let parent = TensorLayout::contiguous(self.shape().to_vec())?;
        // The slice as a strided layout over the contiguous parent; walk it row by row.
        let (mut shape, mut strides, mut offset) = (Vec::new(), Vec::new(), 0);
        for (s, &stride) in specs.iter().zip(&parent.strides) {
            offset += s.start * stride;
            if !s.squeeze {
                shape.push(s.len);
                strides.push(s.step * stride);
            }
        }
        let layout = TensorLayout {
            shape,
            strides,
            offset,
            count: view.numel(),
            contiguous: false,
        };
        Ok(layout.gather_with(|address| address))
    }
    pub fn permute(&self, axes: &[usize]) -> Result<Self, String> {
        if axes.len() != self.shape().len() {
            return Err("permutation rank mismatch".into());
        }
        let mut seen = vec![false; axes.len()];
        for &axis in axes {
            if axis >= axes.len() || seen[axis] {
                return Err("invalid permutation axes".into());
            }
            seen[axis] = true;
        }
        let layout = TensorLayout::checked(
            axes.iter().map(|&a| self.shape()[a]).collect(),
            axes.iter().map(|&a| self.strides()[a]).collect(),
            self.storage_offset(),
            self.storage.0.len(),
        )?;
        Ok(self.view(layout))
    }
}
#[cfg(test)]
mod tests {
    use super::TensorData;
    #[test]
    fn validates_shape() {
        assert!(TensorData::new(vec![1.0], vec![2]).is_err());
        assert!(TensorData::new(vec![], vec![usize::MAX, 2]).is_err());
        assert_eq!(
            TensorData::new(vec![3.0], vec![]).unwrap().shape(),
            &[] as &[usize]
        );
        assert_eq!(
            TensorData::new(vec![], vec![0, 3]).unwrap().values(),
            &[] as &[f32]
        );
    }
    #[test]
    fn reshape_preserves_values() {
        let t = TensorData::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let r = t.reshape(vec![2]).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.values(), &[1.0, 2.0]);
        assert!(t.reshape(vec![3]).is_err());
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    #[test]
    fn empty_layout_accepts_large_unreachable_dimensions() {
        let x = TensorData::new(vec![], vec![0, usize::MAX, 2]).unwrap();
        assert!(x.is_contiguous());
        assert!(x.values().is_empty());
        assert!(x.permute(&[2, 1, 0]).unwrap().values().is_empty());
    }
    #[test]
    fn checked_layout_rejects_invalid_addresses() {
        assert!(TensorLayout::checked(vec![2], vec![], 0, 2).is_err());
        assert!(TensorLayout::checked(vec![2], vec![2], 0, 2).is_err());
        assert!(TensorLayout::checked(vec![2], vec![usize::MAX], 1, 2).is_err());
        assert!(TensorLayout::checked(vec![0], vec![1], 3, 2).is_err());
        assert!(TensorLayout::checked(vec![], vec![], 1, 1).is_err());
    }
    #[test]
    fn offset_contiguous_views_and_cached_clones() {
        let x = TensorData::new((0..12).map(|v| v as f32).collect(), vec![3, 4]).unwrap();
        let layout = TensorLayout::checked(vec![2, 2], vec![2, 1], 3, 12).unwrap();
        let y = x.view(layout);
        assert_eq!(y.values(), &[3., 4., 5., 6.]);
        assert_eq!(y.reshape(vec![4]).unwrap().storage_offset(), 3);
        let strided = x.view(TensorLayout::checked(vec![2, 2], vec![1, 4], 1, 12).unwrap());
        assert_eq!(strided.values(), &[1., 5., 2., 6.]);
        let p = x.permute(&[1, 0]).unwrap();
        let clone = p.clone();
        assert!(p.logical.get().is_none());
        assert_eq!(
            p.values(),
            &[0., 4., 8., 1., 5., 9., 2., 6., 10., 3., 7., 11.]
        );
        assert_eq!(p.values().as_ptr(), clone.values().as_ptr());
        assert!(x.shares_storage(&p));
        assert!(x.logical.get().is_none());
    }
    #[test]
    fn normalized_slice_validation_and_offsets() {
        let x = TensorData::new((0..12).map(|v| v as f32).collect(), vec![3, 4]).unwrap();
        let spec = |start, step, len, squeeze| SliceSpec {
            start,
            step,
            len,
            squeeze,
        };
        let y = x
            .slice(&[spec(1, 1, 2, false), spec(1, 2, 2, false)])
            .unwrap();
        assert_eq!(y.values(), &[5., 7., 9., 11.]);
        assert_eq!(y.storage_offset(), 5);
        assert!(x.shares_storage(&y));
        assert_eq!(
            x.slice_indices(&[spec(1, 1, 2, false), spec(1, 2, 2, false)])
                .unwrap(),
            vec![5, 7, 9, 11]
        );
        assert!(x.slice(&[]).is_err());
        for bad in [
            spec(0, 0, 1, false),
            spec(0, 1, 2, true),
            spec(3, 1, 1, false),
            spec(0, usize::MAX, 3, false),
        ] {
            assert!(x.slice(&[bad, spec(0, 1, 4, false)]).is_err());
        }
        let empty = x
            .slice(&[spec(3, 1, 0, false), spec(4, 1, 0, false)])
            .unwrap();
        assert!(empty.values().is_empty());
    }
    #[test]
    fn gather_matches_per_element_addresses_for_strided_views() {
        let x = TensorData::new((0..120).map(|v| v as f32).collect(), vec![2, 3, 4, 5]).unwrap();
        let step = |start, step, len| SliceSpec {
            start,
            step,
            len,
            squeeze: false,
        };
        let views = [
            x.permute(&[0, 2, 1, 3]).unwrap(),
            x.permute(&[3, 1, 0, 2]).unwrap(),
            x.slice(&[step(1, 1, 1), step(0, 2, 2), step(1, 2, 2), step(4, 1, 1)])
                .unwrap(),
            x.slice(&[step(0, 1, 2), step(2, 1, 1), step(0, 3, 2), step(0, 2, 3)])
                .unwrap()
                .permute(&[3, 2, 0, 1])
                .unwrap(),
            x.permute(&[1, 0, 2, 3])
                .unwrap()
                .slice(&[step(0, 1, 3), step(0, 1, 0), step(0, 1, 4), step(0, 1, 5)])
                .unwrap(),
        ];
        for view in views {
            let expected: Vec<f32> = (0..view.numel())
                .map(|i| x.values()[view.layout.address(i)])
                .collect();
            assert!(view.numel() == 0 || !view.is_contiguous());
            assert_eq!(view.values(), expected.as_slice());
        }
        let scalar = x
            .slice(&[step(1, 1, 1), step(2, 1, 1), step(3, 1, 1), step(4, 1, 1)])
            .unwrap()
            .reshape(vec![])
            .unwrap();
        assert_eq!(scalar.values(), &[119.]);
    }
    #[test]
    fn shared_storage_survives_threads_and_parents() {
        let x = TensorData::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
        let y = x.permute(&[1, 0]).unwrap();
        let z = y.clone();
        drop(x);
        let thread = std::thread::spawn(move || z.values().to_vec());
        assert_eq!(y.values(), thread.join().unwrap());
    }
}

#[cfg(test)]
mod boolean_storage_tests {
    use super::*;
    #[test]
    fn boolean_storage_views_preserve_dtype_and_logical_values() {
        let a =
            BoolTensorData::new(vec![true, false, false, true, true, false], vec![2, 3]).unwrap();
        assert_eq!(a.dtype(), DType::Bool);
        let spec = TensorSpec::from_tensor(&a);
        assert!(spec.matches(&a));
        assert!(
            spec.check(&TensorData::new(vec![0.; 6], vec![2, 3]).unwrap(), "input")
                .unwrap_err()
                .contains("dtype mismatch")
        );
        assert!(crate::graph::Graph::new(1).compile_specs(&[spec]).is_err());
        let t = a.permute(&[1, 0]).unwrap();
        assert!(a.shares_storage(&t));
        assert_eq!(t.values(), &[true, true, false, true, false, false]);
        let row = t
            .slice(&[
                SliceSpec {
                    start: 1,
                    step: 1,
                    len: 2,
                    squeeze: false,
                },
                SliceSpec {
                    start: 0,
                    step: 1,
                    len: 1,
                    squeeze: true,
                },
            ])
            .unwrap();
        assert!(a.shares_storage(&row));
        assert_eq!(row.values(), &[false, false]);
        let copy = t.reshape(vec![6]).unwrap();
        assert!(!a.shares_storage(&copy));
        assert_eq!(copy.values(), &[true, true, false, true, false, false]);
        assert!(a.shares_storage(&a.reshape(vec![6]).unwrap()));
    }
    #[test]
    fn boolean_storage_checks_shapes_and_empty_views() {
        assert!(BoolTensorData::new(vec![true], vec![2]).is_err());
        assert!(BoolTensorData::new(vec![], vec![usize::MAX, 2]).is_err());
        let a = BoolTensorData::new(vec![], vec![usize::MAX, 2, 0]).unwrap();
        assert!(a.permute(&[2, 0, 1]).unwrap().values().is_empty());
        assert_eq!(a.dtype(), DType::Bool);
        assert_eq!(std::mem::size_of::<bool>(), 1);
    }
}

#[cfg(test)]
mod storage_capacity_tracker_tests {
    use super::*;

    #[test]
    fn tracker_deduplicates_clones_and_observes_storage_release() {
        let tracker = TensorStorageTracker::start();
        let first = TensorData::new(vec![1., 2., 3.], vec![3]).unwrap();
        assert_eq!(tracker.peak_capacity_bytes(), 12);

        let clone = first.clone();
        assert_eq!(tracker.live_capacity_bytes(), 12);

        let second = TensorData::new(vec![4., 5.], vec![2]).unwrap();
        assert_eq!(tracker.peak_capacity_bytes(), 20);
        drop(second);
        assert_eq!(tracker.live_capacity_bytes(), 12);
        drop(clone);
        assert_eq!(tracker.live_capacity_bytes(), 12);
        drop(first);
        assert_eq!(tracker.live_capacity_bytes(), 0);
    }
}
