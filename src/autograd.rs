use crate::{
    ops::{self, BinaryOp, UnaryOp},
    tensor::{SliceSpec, TensorData},
};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
static ENGINE: Mutex<()> = Mutex::new(());
#[path = "autograd_session.rs"]
mod session;
pub(crate) use session::{BackwardSession, BackwardStep};
thread_local! { static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) }; }
#[cfg(test)]
thread_local! { static ACCUMULATE_PAYLOAD_PEAK_BYTES: Cell<usize> = const { Cell::new(0) }; }
#[cfg(test)]
thread_local! { static ACCUMULATE_MERGE_COUNT: Cell<usize> = const { Cell::new(0) }; }
#[cfg(test)]
fn reset_accumulate_payload_peak() {
    ACCUMULATE_PAYLOAD_PEAK_BYTES.with(|peak| peak.set(0));
    ACCUMULATE_MERGE_COUNT.with(|count| count.set(0));
}
#[cfg(test)]
fn accumulate_payload_peak() -> usize {
    ACCUMULATE_PAYLOAD_PEAK_BYTES.with(Cell::get)
}
#[cfg(test)]
fn accumulate_merge_count() -> usize {
    ACCUMULATE_MERGE_COUNT.with(Cell::get)
}
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|mode| mode.get())
}
pub fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|mode| mode.replace(enabled))
}
fn lock<T>(m: &Mutex<T>) -> Result<MutexGuard<'_, T>, String> {
    m.lock().map_err(|_| "autograd lock poisoned".into())
}
#[derive(Clone)]
pub struct TensorHandle(Arc<Node>);
#[derive(Clone)]
pub(crate) struct TensorIdentity(Weak<Node>);
impl TensorIdentity {
    pub(crate) fn id(&self) -> usize {
        self.0.as_ptr() as usize
    }
    pub(crate) fn alive(&self) -> bool {
        self.0.strong_count() != 0
    }
    pub(crate) fn set_hooks(&self, enabled: bool) {
        if let Some(node) = self.0.upgrade() {
            node.has_hooks.store(enabled, Ordering::SeqCst);
        }
    }
}
struct Node {
    value: TensorData,
    requires_grad: bool,
    is_leaf: bool,
    has_custom: bool,
    has_hooks: AtomicBool,
    state: Mutex<NodeState>,
    version: Arc<AtomicU64>,
    generation: u64,
    provenance: Vec<(Arc<AtomicU64>, u64)>,
}
struct NodeState {
    grad: Option<TensorData>,
    backward: BackwardState,
}
#[derive(Clone)]
enum BackwardState {
    Leaf,
    Ready(Recipe),
    Consumed,
}
#[derive(Clone)]
enum Recipe {
    Binary(BinaryOp, TensorHandle, TensorHandle),
    BinaryMath(crate::binary_math::BinaryMath, TensorHandle, TensorHandle),
    Unary(UnaryOp, TensorHandle),
    Slice(TensorHandle, Vec<SliceSpec>),
    Reduce(TensorHandle, Vec<usize>, bool, bool),
    /// amax/amin over axes: (input, axes, keepdims, max).
    Extreme(TensorHandle, Vec<usize>, bool, bool),
    Matmul(TensorHandle, TensorHandle),
    Bmm(TensorHandle, TensorHandle),
    Mv(TensorHandle, TensorHandle),
    Dot(TensorHandle, TensorHandle),
    Cat(Vec<TensorHandle>, usize),
    CrossEntropy(TensorHandle, Vec<usize>),
    Embedding(TensorHandle, Vec<usize>),
    MaskedFill(TensorHandle, Arc<Vec<bool>>),
    /// An elementwise result computed elsewhere: each input (same shape as the output) gets
    /// grad * its slope.
    Pointwise(Vec<(TensorHandle, Arc<Vec<f32>>)>),
    /// base + src added at flat positions of base (row-major logical order).
    ScatterAdd(TensorHandle, TensorHandle, Arc<Vec<usize>>),
    Mse(TensorHandle, TensorHandle),
    Linear(TensorHandle, TensorHandle, Option<TensorHandle>),
    Compiled(Arc<CompiledVjp>),
    Custom(Arc<CustomVjp>),
    LayerNorm(
        TensorHandle,
        Option<TensorHandle>,
        Option<TensorHandle>,
        f32,
    ),
}
struct CustomVjp {
    id: u64,
    parents: Vec<TensorHandle>,
    saved: Vec<TensorHandle>,
}
struct CompiledVjp {
    plan: Arc<crate::graph::BackwardPlan>,
    workspace: Mutex<crate::graph::TrainingWorkspace>,
    parents: Vec<TensorHandle>,
}
impl Recipe {
    fn parents(&self) -> Vec<TensorHandle> {
        match self {
            Self::Compiled(saved) => saved.parents.clone(),
            Self::Custom(saved) => saved.parents.clone(),
            Self::Cat(inputs, _) => inputs.clone(),
            Self::Pointwise(parts) => parts.iter().map(|(input, _)| input.clone()).collect(),
            Self::LayerNorm(x, w, b, _) => {
                let mut p = vec![x.clone()];
                p.extend(w.iter().cloned());
                p.extend(b.iter().cloned());
                p
            }
            Self::Linear(x, w, b) => {
                let mut p = vec![x.clone(), w.clone()];
                p.extend(b.iter().cloned());
                p
            }
            Self::Binary(_, a, b)
            | Self::BinaryMath(_, a, b)
            | Self::Matmul(a, b)
            | Self::Bmm(a, b)
            | Self::Mv(a, b)
            | Self::Dot(a, b)
            | Self::Mse(a, b)
            | Self::ScatterAdd(a, b, _) => {
                vec![a.clone(), b.clone()]
            }
            Self::Unary(_, a)
            | Self::Slice(a, _)
            | Self::CrossEntropy(a, _)
            | Self::Embedding(a, _)
            | Self::MaskedFill(a, _)
            | Self::Reduce(a, _, _, _)
            | Self::Extreme(a, _, _, _) => {
                vec![a.clone()]
            }
        }
    }
}
impl Recipe {
    fn check_saved(&self) -> Result<(), String> {
        if let Self::Custom(saved) = self {
            for tensor in &saved.saved {
                tensor.check_current()?;
            }
        }
        Ok(())
    }
}
// Drain uniquely owned ancestors iteratively when an unused graph is dropped.
impl Drop for Node {
    fn drop(&mut self) {
        fn take_parents(node: &mut Node) -> Vec<TensorHandle> {
            let state = node.state.get_mut().unwrap_or_else(|e| e.into_inner());
            match std::mem::replace(&mut state.backward, BackwardState::Consumed) {
                BackwardState::Ready(recipe) => recipe.parents(),
                _ => Vec::new(),
            }
        }
        let mut pending = take_parents(self);
        while let Some(parent) = pending.pop() {
            if let Ok(mut node) = Arc::try_unwrap(parent.0) {
                pending.extend(take_parents(&mut node));
            }
        }
    }
}
impl TensorHandle {
    pub(crate) fn compiled(
        value: TensorData,
        plan: Arc<crate::graph::BackwardPlan>,
        workspace: crate::graph::TrainingWorkspace,
        parents: Vec<Self>,
    ) -> Self {
        Self::result(
            value,
            Recipe::Compiled(Arc::new(CompiledVjp {
                plan,
                workspace: Mutex::new(workspace),
                parents,
            })),
        )
    }
    pub fn new(value: TensorData, requires_grad: bool) -> Self {
        Self(Arc::new(Node {
            value,
            requires_grad,
            is_leaf: true,
            has_custom: false,
            has_hooks: AtomicBool::new(false),
            version: Arc::new(AtomicU64::new(0)),
            generation: 0,
            provenance: Vec::new(),
            state: Mutex::new(NodeState {
                grad: None,
                backward: BackwardState::Leaf,
            }),
        }))
    }
    fn result(value: TensorData, recipe: Recipe) -> Self {
        if !is_grad_enabled() {
            return Self::new(value, false);
        }
        let requires_grad = recipe.parents().iter().any(|p| p.requires_grad());
        if !requires_grad {
            let mut result = Self::new(value, false);
            let node = Arc::get_mut(&mut result.0).expect("new node is uniquely owned");
            let mut seen = HashSet::new();
            for parent in recipe.parents() {
                for (token, generation) in
                    std::iter::once((parent.0.version.clone(), parent.0.generation))
                        .chain(parent.0.provenance.iter().cloned())
                {
                    if seen.insert((Arc::as_ptr(&token) as usize, generation)) {
                        node.provenance.push((token, generation));
                    }
                }
            }
            return result;
        }
        let has_custom =
            matches!(&recipe, Recipe::Custom(_)) || recipe.parents().iter().any(|p| p.has_custom());
        Self(Arc::new(Node {
            value,
            requires_grad,
            is_leaf: false,
            has_custom,
            has_hooks: AtomicBool::new(false),
            version: Arc::new(AtomicU64::new(0)),
            generation: 0,
            provenance: Vec::new(),
            state: Mutex::new(NodeState {
                grad: None,
                backward: BackwardState::Ready(recipe),
            }),
        }))
    }
    fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
    pub(crate) fn has_hooks(&self) -> bool {
        self.0.has_hooks.load(Ordering::SeqCst)
    }
    pub(crate) fn identity(&self) -> TensorIdentity {
        TensorIdentity(Arc::downgrade(&self.0))
    }
    pub fn value(&self) -> &TensorData {
        &self.0.value
    }
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad
    }
    pub fn is_leaf(&self) -> bool {
        self.0.is_leaf
    }
    pub(crate) fn has_custom(&self) -> bool {
        self.0.has_custom
    }
    pub(crate) fn saved_snapshot(&self) -> Result<Self, String> {
        self.check_current()?;
        Ok(Self(Arc::new(Node {
            value: self.value().clone(),
            requires_grad: false,
            is_leaf: true,
            has_custom: false,
            has_hooks: AtomicBool::new(false),
            state: Mutex::new(NodeState {
                grad: None,
                backward: BackwardState::Leaf,
            }),
            version: self.0.version.clone(),
            generation: self.0.generation,
            provenance: self.0.provenance.clone(),
        })))
    }
    pub(crate) fn custom(
        value: TensorData,
        parents: Vec<Self>,
        saved: &[Self],
        id: u64,
    ) -> Result<Self, String> {
        let _engine = engine_gate()?;
        for parent in &parents {
            parent.check_current()?;
        }
        let saved = saved
            .iter()
            .map(Self::saved_snapshot)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::result(
            value,
            Recipe::Custom(Arc::new(CustomVjp { id, parents, saved })),
        ))
    }
    pub(crate) fn logical_id(&self) -> usize {
        Arc::as_ptr(&self.0.version) as usize
    }
    pub(crate) fn check_current(&self) -> Result<(), String> {
        if self.0.version.load(Ordering::SeqCst) != self.0.generation
            || self
                .0
                .provenance
                .iter()
                .any(|(v, g)| v.load(Ordering::SeqCst) != *g)
        {
            return Err("tensor was modified after the saved computation".into());
        }
        Ok(())
    }
    pub(crate) fn grad_unlocked(&self) -> Result<Option<TensorData>, String> {
        Ok(lock(&self.0.state)?.grad.clone())
    }
    pub(crate) fn validate_replace(&self, value: &TensorData) -> Result<(), String> {
        self.check_current()?;
        if !self.is_leaf() || value.shape() != self.value().shape() {
            return Err("replacement requires a leaf with matching shape".into());
        }
        if self.0.generation == u64::MAX {
            return Err("parameter version overflow".into());
        }
        Ok(())
    }
    pub(crate) fn prepare_replacement_unlocked(&self, value: TensorData) -> Result<Self, String> {
        self.validate_replace(&value)?;
        let generation = self.0.generation + 1;
        let grad = self.grad_unlocked()?;
        let result = Self(Arc::new(Node {
            value,
            requires_grad: self.requires_grad(),
            is_leaf: true,
            has_custom: false,
            has_hooks: AtomicBool::new(false),
            version: self.0.version.clone(),
            generation,
            provenance: Vec::new(),
            state: Mutex::new(NodeState {
                grad,
                backward: BackwardState::Leaf,
            }),
        }));
        Ok(result)
    }
    pub(crate) fn commit_replacement(&self) {
        self.0.version.store(self.0.generation, Ordering::SeqCst);
    }
    pub fn grad(&self) -> Result<Option<TensorData>, String> {
        let _engine = lock(&ENGINE)?;
        self.check_current()?;
        Ok(lock(&self.0.state)?.grad.clone())
    }
    /// Assign (or clear) a leaf's accumulated gradient, as `tensor.grad = g` in PyTorch.
    pub fn set_grad(&self, grad: Option<TensorData>) -> Result<(), String> {
        let _engine = lock(&ENGINE)?;
        self.check_current()?;
        if !self.is_leaf() {
            return Err("grad can only be assigned on a leaf tensor".into());
        }
        if grad
            .as_ref()
            .is_some_and(|g| g.shape() != self.value().shape())
        {
            return Err("assigned grad shape must match the tensor shape".into());
        }
        lock(&self.0.state)?.grad = grad;
        Ok(())
    }
    pub fn zero_grad(&self) -> Result<(), String> {
        let _engine = lock(&ENGINE)?;
        self.check_current()?;
        if !self.is_leaf() {
            return Err("zero_grad requires a leaf tensor".into());
        }
        lock(&self.0.state)?.grad = None;
        Ok(())
    }
    pub fn binary(&self, b: &Self, op: BinaryOp) -> Result<Self, String> {
        Ok(Self::result(
            ops::binary(self.value(), b.value(), op)?,
            Recipe::Binary(op, self.clone(), b.clone()),
        ))
    }
    pub fn binary_math(
        &self,
        b: &Self,
        op: crate::binary_math::BinaryMath,
    ) -> Result<Self, String> {
        let (x, y) = (self.value(), b.value());
        let value = if matches!(op, crate::binary_math::BinaryMath::Atan2)
            && (x.shape() == y.shape() || x.numel() == 1 || y.numel() == 1)
        {
            // PyTorch's vectorized loop over one contiguous run (a single value broadcasts).
            let shape = ops::broadcast_shape(x.shape(), y.shape())?;
            let n = shape.iter().product::<usize>();
            let full = |t: &TensorData| -> Vec<f32> {
                if t.numel() == n {
                    t.values().to_vec()
                } else {
                    vec![t.values()[0]; n]
                }
            };
            TensorData::new(crate::sleef::atan2_array(&full(x), &full(y)), shape)?
        } else {
            ops::broadcast_map(x, y, |l, r| op.apply(l, r))?
        };
        Ok(Self::result(
            value,
            Recipe::BinaryMath(op, self.clone(), b.clone()),
        ))
    }
    pub fn unary(&self, op: UnaryOp) -> Result<Self, String> {
        if matches!(op, UnaryOp::Math(crate::dispatch::UnaryMath::Gelu))
            && is_grad_enabled()
            && self.requires_grad()
        {
            // Values and slopes share one erfc, so backward does not recompute it.
            let (values, slopes) = crate::dispatch::gelu_value_and_slope(self.value().values());
            let value = TensorData::new(values, self.value().shape().to_vec())?;
            return Self::pointwise(std::slice::from_ref(self), value, vec![slopes]);
        }
        Ok(Self::result(
            ops::unary(self.value(), &op)?,
            Recipe::Unary(op, self.clone()),
        ))
    }
    pub fn slice(&self, specs: Vec<SliceSpec>) -> Result<Self, String> {
        Ok(Self::result(
            self.value().slice(&specs)?,
            Recipe::Slice(self.clone(), specs),
        ))
    }
    pub fn reduce_axes(
        &self,
        axes: Vec<usize>,
        keepdims: bool,
        mean: bool,
    ) -> Result<Self, String> {
        let (value, _) = ops::reduce_axes(self.value(), &axes, keepdims, mean)?;
        Ok(Self::result(
            value,
            Recipe::Reduce(self.clone(), axes, keepdims, mean),
        ))
    }
    pub fn extreme(&self, axes: Vec<usize>, keepdims: bool, max: bool) -> Result<Self, String> {
        let (value, _) = extreme_forward(self.value(), &axes, keepdims, max)?;
        Ok(Self::result(
            value,
            Recipe::Extreme(self.clone(), axes, keepdims, max),
        ))
    }
    pub fn matmul(&self, b: &Self) -> Result<Self, String> {
        let value = match (mat(self.value()), mat(b.value())) {
            (Some(x), Some(y)) => {
                let out = crate::torch_blas::mm(x, y)?;
                TensorData::new(out, vec![x.sizes[0], y.sizes[1]])?
            }
            _ => ops::matmul(self.value(), b.value())?,
        };
        Ok(Self::result(value, Recipe::Matmul(self.clone(), b.clone())))
    }
    pub fn cat(inputs: Vec<Self>, axis: usize) -> Result<Self, String> {
        let data: Vec<_> = inputs.iter().map(|a| a.value()).collect();
        let value = crate::dispatch::cat(&data, axis)?;
        Ok(Self::result(value, Recipe::Cat(inputs, axis)))
    }
    pub fn bmm(&self, b: &Self) -> Result<Self, String> {
        Ok(Self::result(
            crate::dispatch::bmm(self.value(), b.value())?,
            Recipe::Bmm(self.clone(), b.clone()),
        ))
    }
    /// PyTorch's mv (addmv through sgemv_).
    pub fn mv(&self, v: &Self) -> Result<Self, String> {
        let (m, x) = match (mat(self.value()), vector(v.value())) {
            (Some(m), Some(x)) => (m, x),
            _ => return Err("mv expects a matrix and a vector".into()),
        };
        let out = crate::torch_blas::mv(m, x)?;
        Ok(Self::result(
            TensorData::new(out, vec![m.sizes[0]])?,
            Recipe::Mv(self.clone(), v.clone()),
        ))
    }
    /// PyTorch's dot (BLAS sdot).
    pub fn dot(&self, other: &Self) -> Result<Self, String> {
        let (x, y) = match (vector(self.value()), vector(other.value())) {
            (Some(x), Some(y)) => (x, y),
            _ => return Err("dot expects 1-D tensors".into()),
        };
        let value = crate::torch_blas::dot(x, y)?;
        Ok(Self::result(
            TensorData::new(vec![value], vec![])?,
            Recipe::Dot(self.clone(), other.clone()),
        ))
    }
    pub fn linear(&self, w: &Self, bias: Option<&Self>) -> Result<Self, String> {
        let value = match (mat(self.value()), mat(w.value())) {
            (Some(x), Some(wm))
                if x.sizes[1] == wm.sizes[1]
                    && bias.is_none_or(|b| b.value().shape() == [wm.sizes[0]]) =>
            {
                // PyTorch: addmm(bias, input, weight.t()) with a bias, else mm(input, weight.t()).
                let (rows, cols) = (x.sizes[0], wm.sizes[0]);
                let out = match bias {
                    Some(b) => {
                        let row = b.value().values();
                        let init: Vec<f32> = (0..rows).flat_map(|_| row.iter().copied()).collect();
                        crate::torch_blas::addmm(x, wm.t(), Some(&init), 1.0, 1.0)?
                    }
                    None => crate::torch_blas::mm(x, wm.t())?,
                };
                TensorData::new(out, vec![rows, cols])?
            }
            _ => ops::linear(self.value(), w.value(), bias.map(|b| b.value()))?,
        };
        Ok(Self::result(
            value,
            Recipe::Linear(self.clone(), w.clone(), bias.cloned()),
        ))
    }
    pub fn layer_norm(
        &self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        eps: f32,
    ) -> Result<Self, String> {
        Ok(Self::result(
            crate::normalization::forward(
                self.value(),
                weight.map(Self::value),
                bias.map(Self::value),
                eps,
            )?,
            Recipe::LayerNorm(self.clone(), weight.cloned(), bias.cloned(), eps),
        ))
    }
    pub fn mse_loss(&self, target: &Self) -> Result<Self, String> {
        Ok(Self::result(
            crate::loss::mse_loss(self.value(), target.value())?,
            Recipe::Mse(self.clone(), target.clone()),
        ))
    }

    /// Wrap externally computed elementwise values with their local derivatives.
    pub fn pointwise(
        inputs: &[Self],
        value: TensorData,
        slopes: Vec<Vec<f32>>,
    ) -> Result<Self, String> {
        if inputs.len() != slopes.len() {
            return Err("pointwise needs one slope array per input".into());
        }
        for (input, slope) in inputs.iter().zip(&slopes) {
            if input.value().shape() != value.shape() || slope.len() != value.numel() {
                return Err("pointwise inputs and slopes must match the output shape".into());
            }
        }
        let parts = inputs
            .iter()
            .cloned()
            .zip(slopes.into_iter().map(Arc::new))
            .collect();
        Ok(Self::result(value, Recipe::Pointwise(parts)))
    }
    pub fn scatter_add(&self, src: &Self, positions: Vec<usize>) -> Result<Self, String> {
        let (base, add) = (self.value(), src.value());
        if add.numel() != positions.len() {
            return Err("scatter_add needs one position per source element".into());
        }
        let mut out = base.values().to_vec();
        for (&p, &v) in positions.iter().zip(add.values()) {
            *out.get_mut(p).ok_or("scatter_add position out of range")? += v;
        }
        Ok(Self::result(
            TensorData::new(out, base.shape().to_vec())?,
            Recipe::ScatterAdd(self.clone(), src.clone(), Arc::new(positions)),
        ))
    }
    pub fn masked_fill(&self, mask: Vec<bool>, value: f32) -> Result<Self, String> {
        Ok(Self::result(
            ops::masked_fill(self.value(), &mask, value)?,
            Recipe::MaskedFill(self.clone(), Arc::new(mask)),
        ))
    }
    /// Gather weight rows; self is the (num_embeddings, embedding_dim) table.
    pub fn embedding(&self, indices: Vec<usize>, index_shape: &[usize]) -> Result<Self, String> {
        Ok(Self::result(
            ops::embedding(self.value(), &indices, index_shape)?,
            Recipe::Embedding(self.clone(), indices),
        ))
    }

    pub fn cross_entropy(&self, labels: Vec<usize>) -> Result<Self, String> {
        Ok(Self::result(
            crate::loss::cross_entropy(self.value(), &labels)?,
            Recipe::CrossEntropy(self.clone(), labels),
        ))
    }
    pub fn backward(&self, seed: Option<&TensorData>, retain_graph: bool) -> Result<(), String> {
        let mut session = BackwardSession::new(self, seed, retain_graph)?;
        match session.advance()? {
            BackwardStep::Done => Ok(()),
            BackwardStep::Custom { .. } | BackwardStep::Hook { .. } => {
                session.abort()?;
                Err("custom autograd graphs require Tensor.backward Python driver".into())
            }
        }
    }
}
pub(crate) fn engine_gate() -> Result<MutexGuard<'static, ()>, String> {
    lock(&ENGINE)
}
pub(crate) fn replace_batch(
    params: &[TensorHandle],
    values: Vec<TensorData>,
) -> Result<Vec<TensorHandle>, String> {
    let _engine = engine_gate()?;
    if params.len() != values.len() {
        return Err("replacement parameter count mismatch".into());
    }
    let mut seen = HashSet::new();
    for (p, v) in params.iter().zip(&values) {
        if !seen.insert(p.logical_id()) {
            return Err("duplicate parameter identity".into());
        }
        p.validate_replace(v)?;
    }
    let prepared = params
        .iter()
        .zip(values)
        .map(|(p, v)| p.prepare_replacement_unlocked(v))
        .collect::<Result<Vec<_>, _>>()?;
    for p in &prepared {
        p.commit_replacement();
    }
    Ok(prepared)
}
fn accumulate(
    grads: &mut HashMap<usize, TensorData>,
    id: usize,
    g: TensorData,
) -> Result<(), String> {
    if let Some(old) = grads.get(&id) {
        let combined = ops::binary(old, &g, BinaryOp::Add)?;
        #[cfg(test)]
        ACCUMULATE_PAYLOAD_PEAK_BYTES.with(|peak| {
            let mut seen = HashSet::new();
            let live = grads
                .values()
                .chain(std::iter::once(&g))
                .chain(std::iter::once(&combined))
                .filter(|tensor| seen.insert(tensor.storage_identity()))
                .map(TensorData::storage_capacity_bytes)
                .sum();
            peak.set(peak.get().max(live));
        });
        #[cfg(test)]
        ACCUMULATE_MERGE_COUNT.with(|count| count.set(count.get() + 1));
        grads.insert(id, combined);
    } else {
        grads.insert(id, g);
    }
    Ok(())
}
/// Gradients of a broadcast binary op: `da`/`db` map (left, right, incoming) to each
/// side's contribution, summed into zeros in output order.
fn broadcast_vjp(
    a: &TensorHandle,
    b: &TensorHandle,
    g: &TensorData,
    da: impl Fn(f32, f32, f32) -> f32,
    db: impl Fn(f32, f32, f32) -> f32,
) -> Result<Vec<(TensorHandle, TensorData)>, String> {
    let (sa, sb) = (a.value().shape(), b.value().shape());
    if sa == sb {
        vjp_with(a, b, g, |i| i, |i| i, &da, &db)
    } else if let Some(n) = ops::leading_repeat(sa, sb) {
        vjp_with(a, b, g, |i| i, |i| i % n, &da, &db)
    } else if let Some(n) = ops::leading_repeat(sb, sa) {
        vjp_with(a, b, g, |i| i % n, |i| i, &da, &db)
    } else {
        let p = ops::broadcast_plan(sa, sb)?;
        vjp_with(a, b, g, |i| p.left[i], |i| p.right[i], &da, &db)
    }
}
/// Output element `i` reads `a[ai(i)]` and `b[bi(i)]`. Each gradient is computed elementwise
/// over the output, then summed back to its input's shape as PyTorch's `sum_to_size` does.
fn vjp_with(
    a: &TensorHandle,
    b: &TensorHandle,
    g: &TensorData,
    ai: impl Fn(usize) -> usize,
    bi: impl Fn(usize) -> usize,
    da: &impl Fn(f32, f32, f32) -> f32,
    db: &impl Fn(f32, f32, f32) -> f32,
) -> Result<Vec<(TensorHandle, TensorData)>, String> {
    let (x, y, gv) = (a.value().values(), b.value().values(), g.values());
    let mut parts = Vec::with_capacity(2);
    if a.requires_grad() {
        let full = (0..gv.len())
            .map(|i| da(x[ai(i)], y[bi(i)], gv[i]))
            .collect();
        parts.push((a.clone(), sum_to(full, g.shape(), a.value().shape())?));
    }
    if b.requires_grad() {
        let full = (0..gv.len())
            .map(|i| db(x[ai(i)], y[bi(i)], gv[i]))
            .collect();
        parts.push((b.clone(), sum_to(full, g.shape(), b.value().shape())?));
    }
    Ok(parts)
}
/// A 1-D float32 tensor as a strided BLAS vector (None for other ranks).
fn vector(t: &TensorData) -> Option<crate::torch_blas::Vector<'_>> {
    (t.shape().len() == 1).then(|| crate::torch_blas::Vector {
        data: t.storage(),
        offset: t.storage_offset(),
        len: t.shape()[0],
        stride: t.strides()[0],
    })
}
/// A 2-D float32 tensor as a strided BLAS view (None for other ranks).
fn mat(t: &TensorData) -> Option<crate::torch_blas::Mat<'_>> {
    let (shape, strides) = (t.shape(), t.strides());
    (shape.len() == 2).then(|| crate::torch_blas::Mat {
        data: t.storage(),
        offset: t.storage_offset(),
        sizes: [shape[0], shape[1]],
        strides: [strides[0], strides[1]],
    })
}
/// PyTorch's `sum_to`: sum leading axes and the size-1 axes the operand was broadcast along.
fn sum_to(full: Vec<f32>, shape: &[usize], target: &[usize]) -> Result<TensorData, String> {
    if shape == target {
        return TensorData::new(full, target.to_vec());
    }
    let lead = shape.len() - target.len();
    let reduced: Vec<bool> = (0..shape.len())
        .map(|i| i < lead || (target[i - lead] == 1 && shape[i] != 1))
        .collect();
    TensorData::new(
        crate::torch_sum::sum(&full, shape, &reduced),
        target.to_vec(),
    )
}
/// Largest (or smallest) value per output slot; NaN wins, as in PyTorch's amax/amin.
fn extreme_forward(
    a: &TensorData,
    axes: &[usize],
    keepdims: bool,
    max: bool,
) -> Result<(TensorData, Vec<usize>), String> {
    let (sums, mapping) = ops::reduce_axes(a, axes, keepdims, false)?;
    let start = if max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let mut out = vec![start; sums.numel()];
    for (&x, &o) in a.values().iter().zip(&mapping) {
        let current = out[o];
        let better = if max { x > current } else { x < current };
        if x.is_nan() || (!current.is_nan() && better) {
            out[o] = x;
        }
    }
    Ok((TensorData::new(out, sums.shape().to_vec())?, mapping))
}
fn derivative(recipe: &Recipe, g: &TensorData) -> Result<Vec<(TensorHandle, TensorData)>, String> {
    match recipe {
        Recipe::Custom(_) => Err("custom VJP requires resumable backward".into()),
        Recipe::Compiled(saved) => {
            let mut workspace = lock(&saved.workspace)?;
            saved.plan.backward_unlocked(Some(g), &mut workspace)?;
            saved
                .parents
                .iter()
                .enumerate()
                .filter_map(|(i, parent)| {
                    workspace.take_input_gradient(i).map(|gradient| {
                        TensorData::new(gradient, parent.value().shape().to_vec())
                            .map(|g| (parent.clone(), g))
                    })
                })
                .collect()
        }
        Recipe::LayerNorm(x, w, b, eps) => {
            let (m, n) = crate::normalization::dimensions(
                x.value(),
                w.as_ref().map(TensorHandle::value),
                b.as_ref().map(TensorHandle::value),
                *eps,
            )?;
            let mut result = Vec::new();
            for (parent, kind) in [
                (Some(x), crate::normalization::Gradient::Input),
                (w.as_ref(), crate::normalization::Gradient::Weight),
                (b.as_ref(), crate::normalization::Gradient::Bias),
            ] {
                if let Some(parent) = parent.filter(|p| p.requires_grad()) {
                    let mut out = Vec::new();
                    crate::normalization::gradient_into(
                        Some(x.value().values()),
                        w.as_ref().map(|w| w.value().values()),
                        g.values(),
                        m,
                        n,
                        *eps,
                        kind,
                        &mut out,
                    )?;
                    result.push((
                        parent.clone(),
                        TensorData::new(out, parent.value().shape().to_vec())?,
                    ));
                }
            }
            Ok(result)
        }
        Recipe::Linear(x, w, bias) => {
            // PyTorch's addmm/mm backward with mat2 = weight.t() (a column-major view).
            let (xv, wv) = (x.value(), w.value());
            let (xm, wm, gm) = match (mat(xv), mat(wv), mat(g)) {
                (Some(xm), Some(wm), Some(gm)) => (xm, wm, gm),
                _ => return Err("linear backward expects rank-2 operands".into()),
            };
            let mut parts = Vec::new();
            if x.requires_grad() {
                let dx = crate::torch_blas::mm_mat1_backward(gm, wm.t(), xm)?;
                parts.push((x.clone(), TensorData::new(dx, xv.shape().to_vec())?));
            }
            if w.requires_grad() {
                // Gradient of weight.t(), then through t() back to weight.
                let dwt = crate::torch_blas::mm_mat2_backward(gm, xm, wm.t())?;
                let dw = crate::torch_blas::Mat::rows(&dwt, wm.sizes[1], wm.sizes[0]).t();
                let dw: Vec<f32> = (0..dw.sizes[0])
                    .flat_map(|i| (0..dw.sizes[1]).map(move |j| (i, j)))
                    .map(|(i, j)| dw.data[i * dw.strides[0] + j * dw.strides[1]])
                    .collect();
                parts.push((w.clone(), TensorData::new(dw, wv.shape().to_vec())?));
            }
            if let Some(bias) = bias.as_ref().filter(|bias| bias.requires_grad()) {
                let n = gm.sizes[1];
                parts.push((bias.clone(), sum_to(g.values().to_vec(), g.shape(), &[n])?));
            }
            Ok(parts)
        }
        Recipe::Mse(a, b) => {
            let mut parts = Vec::with_capacity(2);
            let seed = g.values()[0];

            if a.requires_grad() {
                parts.push((
                    a.clone(),
                    crate::loss::mse_loss_grad(a.value(), b.value(), seed, false)?,
                ));
            }

            if b.requires_grad() {
                parts.push((
                    b.clone(),
                    crate::loss::mse_loss_grad(a.value(), b.value(), seed, true)?,
                ));
            }

            Ok(parts)
        }
        Recipe::Pointwise(parts) => parts
            .iter()
            .filter(|(input, _)| input.requires_grad())
            .map(|(input, slope)| {
                let values = g
                    .values()
                    .iter()
                    .zip(slope.iter())
                    .map(|(&g, &s)| g * s)
                    .collect();
                Ok((
                    input.clone(),
                    TensorData::new(values, input.value().shape().to_vec())?,
                ))
            })
            .collect(),
        Recipe::ScatterAdd(base, src, positions) => {
            let mut parts = Vec::with_capacity(2);
            if base.requires_grad() {
                parts.push((base.clone(), g.clone()));
            }
            if src.requires_grad() {
                let values = positions.iter().map(|&p| g.values()[p]).collect();
                parts.push((
                    src.clone(),
                    TensorData::new(values, src.value().shape().to_vec())?,
                ));
            }
            Ok(parts)
        }
        Recipe::MaskedFill(a, mask) => Ok(vec![(a.clone(), ops::masked_fill_grad(g, mask)?)]),
        Recipe::Embedding(w, indices) => Ok(vec![(
            w.clone(),
            ops::embedding_grad(w.value().shape(), indices, g)?,
        )]),
        Recipe::CrossEntropy(a, labels) => Ok(vec![(
            a.clone(),
            crate::loss::cross_entropy_grad(a.value(), labels, g.values()[0])?,
        )]),
        // Same arithmetic, in the same order, as the original per-op formulas.
        Recipe::Binary(op, a, b) => match op {
            BinaryOp::Add => broadcast_vjp(a, b, g, |_, _, s| s, |_, _, s| s),
            BinaryOp::Sub => broadcast_vjp(a, b, g, |_, _, s| s, |_, _, s| -s),
            BinaryOp::Mul => broadcast_vjp(a, b, g, |_, r, s| s * r, |l, _, s| s * l),
            // PyTorch: grad / other and -grad * ((self / other) / other).
            BinaryOp::Div => broadcast_vjp(a, b, g, |_, r, s| s / r, |l, r, s| -s * ((l / r) / r)),
        },
        Recipe::BinaryMath(op, a, b) => broadcast_vjp(
            a,
            b,
            g,
            |l, r, s| op.grad_left(l, r, s),
            |l, r, s| op.grad_right(l, r, s),
        ),
        Recipe::Cat(inputs, axis) => {
            let mut offset = 0;
            let mut parts = Vec::new();
            for input in inputs {
                let width = input.value().shape()[*axis];
                if input.requires_grad() {
                    let specs: Vec<_> = g
                        .shape()
                        .iter()
                        .enumerate()
                        .map(|(i, &len)| SliceSpec {
                            start: if i == *axis { offset } else { 0 },
                            step: 1,
                            len: if i == *axis { width } else { len },
                            squeeze: false,
                        })
                        .collect();
                    parts.push((input.clone(), g.slice(&specs)?));
                }
                offset += width;
            }
            Ok(parts)
        }
        Recipe::Bmm(a, b) => {
            let mut parts = Vec::new();
            if a.requires_grad() {
                parts.push((
                    a.clone(),
                    crate::dispatch::bmm(g, &b.value().permute(&[0, 2, 1])?)?,
                ));
            }
            if b.requires_grad() {
                parts.push((
                    b.clone(),
                    crate::dispatch::bmm(&a.value().permute(&[0, 2, 1])?, g)?,
                ));
            }
            Ok(parts)
        }
        Recipe::Mv(a, v) => {
            // derivatives.yaml: self: grad.ger(vec), vec: self.t().mv(grad).
            let (am, vv, gv) = (a.value(), v.value(), g.values());
            let mut parts = Vec::new();
            if a.requires_grad() {
                let x = vv.values();
                let outer: Vec<f32> = gv
                    .iter()
                    .flat_map(|&gi| x.iter().map(move |&xj| gi * xj))
                    .collect();
                parts.push((a.clone(), TensorData::new(outer, am.shape().to_vec())?));
            }
            if v.requires_grad() {
                let (m, gvec) = (
                    mat(am).ok_or("mv backward expects a matrix")?,
                    vector(g).ok_or("mv backward expects a vector")?,
                );
                parts.push((
                    v.clone(),
                    TensorData::new(crate::torch_blas::mv(m.t(), gvec)?, vv.shape().to_vec())?,
                ));
            }
            Ok(parts)
        }
        Recipe::Dot(a, b) => {
            // derivatives.yaml: self: grad * other, other: grad * self.
            let seed = g.values()[0];
            let scale = |t: &TensorData| -> Result<TensorData, String> {
                TensorData::new(
                    t.values().iter().map(|&x| seed * x).collect(),
                    t.shape().to_vec(),
                )
            };
            let mut parts = Vec::new();
            if a.requires_grad() {
                parts.push((a.clone(), scale(b.value())?));
            }
            if b.requires_grad() {
                parts.push((b.clone(), scale(a.value())?));
            }
            Ok(parts)
        }
        Recipe::Matmul(a, b) => {
            let (av, bv) = (a.value(), b.value());
            match (mat(av), mat(bv), mat(g)) {
                (Some(am), Some(bm), Some(gm)) => Ok(vec![
                    (
                        a.clone(),
                        TensorData::new(
                            crate::torch_blas::mm_mat1_backward(gm, bm, am)?,
                            av.shape().to_vec(),
                        )?,
                    ),
                    (
                        b.clone(),
                        TensorData::new(
                            crate::torch_blas::mm_mat2_backward(gm, am, bm)?,
                            bv.shape().to_vec(),
                        )?,
                    ),
                ]),
                _ => Ok(vec![
                    (
                        a.clone(),
                        ops::matmul(g, &ops::unary(bv, &UnaryOp::Transpose)?)?,
                    ),
                    (
                        b.clone(),
                        ops::matmul(&ops::unary(av, &UnaryOp::Transpose)?, g)?,
                    ),
                ]),
            }
        }
        Recipe::Reduce(a, axes, keepdims, mean) => {
            let (_, mapping) = ops::reduce_axes(a.value(), axes, *keepdims, *mean)?;
            let divisor = if *mean && !mapping.is_empty() {
                crate::dispatch::ReductionSpec::new(a.value().shape(), Some(axes), *keepdims)?
                    .divisor()?
            } else {
                1
            };
            let values = mapping
                .iter()
                .map(|&i| {
                    let v = g.values()[i];
                    if *mean { v / divisor as f32 } else { v }
                })
                .collect();
            Ok(vec![(
                a.clone(),
                TensorData::new(values, a.value().shape().to_vec())?,
            )])
        }
        Recipe::Extreme(a, axes, keepdims, max) => {
            // PyTorch: (grad / count of elements equal to the result) * (element == result).
            let (out, mapping) = extreme_forward(a.value(), axes, *keepdims, *max)?;
            let (x, best) = (a.value().values(), out.values());
            let mut count = vec![0.0f32; best.len()];
            for (&v, &o) in x.iter().zip(&mapping) {
                count[o] += f32::from(u8::from(v == best[o]));
            }
            let values = x
                .iter()
                .zip(&mapping)
                .map(|(&v, &o)| (g.values()[o] / count[o]) * f32::from(u8::from(v == best[o])))
                .collect();
            Ok(vec![(
                a.clone(),
                TensorData::new(values, a.value().shape().to_vec())?,
            )])
        }
        Recipe::Slice(a, specs) => {
            let indices = a.value().slice_indices(specs)?;
            let mut values = vec![0.; a.value().numel()];
            for (&i, &gradient) in indices.iter().zip(g.values()) {
                values[i] += gradient;
            }
            Ok(vec![(
                a.clone(),
                TensorData::new(values, a.value().shape().to_vec())?,
            )])
        }
        Recipe::Unary(op, a) => {
            let dx = match op {
                UnaryOp::Math(crate::dispatch::UnaryMath::Gelu) => TensorData::new(
                    crate::dispatch::gelu_derivative_array(a.value().values())
                        .into_iter()
                        .zip(g.values())
                        .map(|(slope, &seed)| seed * slope)
                        .collect(),
                    a.value().shape().to_vec(),
                )?,
                UnaryOp::Math(kind) if crate::sleef::unary_backward(*kind, &[], &[]).is_some() => {
                    TensorData::new(
                        crate::sleef::unary_backward(*kind, a.value().values(), g.values())
                            .expect("checked above"),
                        a.value().shape().to_vec(),
                    )?
                }
                UnaryOp::Math(kind) => {
                    use crate::dispatch::{UnaryMath, erf_slope, sign, sinc_slope};
                    let values = a
                        .value()
                        .values()
                        .iter()
                        .zip(g.values())
                        .map(|(&x, &seed)| {
                            // Formulas are shared semantics, independent of kernel registration.
                            match kind {
                                UnaryMath::Sigmoid => {
                                    let y = crate::dispatch::sigmoid(x);
                                    seed * y * (1.0 - y)
                                }
                                UnaryMath::Tanh => {
                                    let y = x.tanh();
                                    seed * (1.0 - y) * (1.0 + y)
                                }
                                UnaryMath::Neg => -seed,
                                UnaryMath::Exp => seed * x.exp(),
                                UnaryMath::Log => seed / x,
                                UnaryMath::Sqrt => seed / (2.0 * x.sqrt()),
                                UnaryMath::Sin => seed * x.cos(),
                                UnaryMath::Cos => -seed * x.sin(),
                                UnaryMath::Gelu => seed * crate::dispatch::gelu_derivative(x),
                                UnaryMath::GeluTanh => {
                                    seed * crate::dispatch::gelu_tanh_derivative(x)
                                }
                                UnaryMath::Abs => seed * sign(x),
                                UnaryMath::Sign => 0.0,
                                UnaryMath::Floor => 0.0,
                                UnaryMath::Ceil => 0.0,
                                UnaryMath::Trunc => 0.0,
                                UnaryMath::Round => 0.0,
                                UnaryMath::Frac => seed,
                                UnaryMath::Reciprocal => {
                                    let y = 1.0 / x;
                                    -seed * y * y
                                }
                                UnaryMath::Rsqrt => {
                                    let y = 1.0 / x.sqrt();
                                    -0.5 * seed * y * y * y
                                }
                                UnaryMath::Square => 2.0 * x * seed,
                                UnaryMath::Exp2 => seed * x.exp2() * std::f32::consts::LN_2,
                                UnaryMath::Expm1 => seed * (x.exp_m1() + 1.0),
                                UnaryMath::Log1p => seed / (1.0 + x),
                                UnaryMath::Log2 => seed / (x * std::f32::consts::LN_2),
                                UnaryMath::Log10 => seed / (x * std::f32::consts::LN_10),
                                UnaryMath::Tan => {
                                    let y = x.tan();
                                    seed * (1.0 + y * y)
                                }
                                UnaryMath::Asin => seed / (1.0 - x * x).sqrt(),
                                UnaryMath::Acos => -seed / (1.0 - x * x).sqrt(),
                                UnaryMath::Atan => seed / (1.0 + x * x),
                                UnaryMath::Sinh => seed * x.cosh(),
                                UnaryMath::Cosh => seed * x.sinh(),
                                UnaryMath::Asinh => seed / (x * x + 1.0).sqrt(),
                                UnaryMath::Acosh => seed / (x * x - 1.0).sqrt(),
                                UnaryMath::Atanh => seed / (1.0 - x * x),
                                UnaryMath::Deg2Rad => seed * (std::f32::consts::PI / 180.0),
                                UnaryMath::Rad2Deg => seed * (180.0 / std::f32::consts::PI),
                                UnaryMath::Erf => seed * erf_slope(x),
                                UnaryMath::Erfc => -seed * erf_slope(x),
                                UnaryMath::Logit => seed / (x * (1.0 - x)),
                                UnaryMath::Sinc => seed * sinc_slope(f64::from(x)) as f32,
                            }
                        })
                        .collect();
                    TensorData::new(values, a.value().shape().to_vec())?
                }
                UnaryOp::Transpose => ops::unary(g, &UnaryOp::Transpose)?,
                UnaryOp::Permute(axes) => {
                    let mut inverse = vec![0; axes.len()];
                    for (i, &axis) in axes.iter().enumerate() {
                        inverse[axis] = i;
                    }
                    ops::permute(g, &inverse)?
                }
                UnaryOp::Reshape(_) => g.reshape(a.value().shape().to_vec())?,
                UnaryOp::Clone => g.clone(),
                UnaryOp::Softmax(axis) => {
                    ops::softmax_grad(&ops::softmax(a.value(), *axis, false)?, g, *axis, false)?
                }
                UnaryOp::LogSoftmax(axis) => {
                    ops::softmax_grad(&ops::softmax(a.value(), *axis, true)?, g, *axis, true)?
                }
                UnaryOp::Relu => TensorData::new(
                    a.value()
                        .values()
                        .iter()
                        .zip(g.values())
                        .map(|(&x, &s)| if x <= 0.0 { 0.0 } else { s })
                        .collect(),
                    a.value().shape().to_vec(),
                )?,
                UnaryOp::Sum | UnaryOp::Mean => {
                    let mut s = g.values()[0];
                    if matches!(op, UnaryOp::Mean) {
                        s /= a.value().values().len() as f32;
                    }
                    TensorData::new(
                        vec![s; a.value().values().len()],
                        a.value().shape().to_vec(),
                    )?
                }
            };
            Ok(vec![(a.clone(), dx)])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scalar(x: f32) -> TensorHandle {
        TensorHandle::new(TensorData::new(vec![x], vec![]).unwrap(), true)
    }
    #[test]
    fn consumed_graph_releases_parents() {
        let x = scalar(2.0);
        let weak = Arc::downgrade(&x.0);
        let y = x.binary(&x, BinaryOp::Mul).unwrap();
        drop(x);
        y.backward(None, false).unwrap();
        assert!(weak.upgrade().is_none());
        assert_eq!(y.value().values(), &[4.0]);
    }
    #[test]
    fn dropped_graph_releases_parents() {
        let x = scalar(2.0);
        let weak = Arc::downgrade(&x.0);
        let y = x.binary(&x, BinaryOp::Mul).unwrap();
        drop(x);
        drop(y);
        assert!(weak.upgrade().is_none());
    }
    #[test]
    fn retained_backward_and_reset_serialize() {
        let x = scalar(2.0);
        let y = x.binary(&x, BinaryOp::Mul).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let reset_x = x.clone();
        let sync = barrier.clone();
        let thread = std::thread::spawn(move || {
            sync.wait();
            reset_x.zero_grad().unwrap();
        });
        barrier.wait();
        y.backward(None, true).unwrap();
        thread.join().unwrap();
        let grad = x.grad().unwrap();
        assert!(grad.is_none() || grad.unwrap().values() == [4.0]);
    }
    #[test]
    fn deep_unused_graph_drops_without_recursive_stack_growth() {
        let x = scalar(1.0);
        let weak = Arc::downgrade(&x.0);
        let mut y = x.clone();
        for _ in 0..20000 {
            y = y.unary(UnaryOp::Relu).unwrap();
        }
        drop(x);
        drop(y);
        assert!(weak.upgrade().is_none());
    }
    #[test]
    fn late_vjp_error_preserves_leaf_gradients_and_retryable_graph() {
        let source = scalar(2.0);
        let sibling = scalar(3.0);
        let other_sibling = scalar(4.0);
        sibling.backward(None, true).unwrap();
        other_sibling.backward(None, true).unwrap();
        // Inject a late derivative failure through an invalid saved reduction.
        // No public callback API or production-only test hook is introduced.
        let bad = TensorHandle::result(
            source.value().clone(),
            Recipe::Reduce(source.clone(), vec![99], false, false),
        );
        let result = sibling
            .binary(&bad, BinaryOp::Add)
            .unwrap()
            .binary(&other_sibling, BinaryOp::Add)
            .unwrap();
        assert!(result.backward(None, false).is_err());
        assert_eq!(sibling.grad().unwrap().unwrap().values(), &[1.]);
        assert_eq!(other_sibling.grad().unwrap().unwrap().values(), &[1.]);
        assert!(source.grad().unwrap().is_none());
        assert!(matches!(
            lock(&result.0.state).unwrap().backward,
            BackwardState::Ready(_)
        ));
        assert!(matches!(
            lock(&bad.0.state).unwrap().backward,
            BackwardState::Ready(_)
        ));
        {
            let _engine = engine_gate().unwrap();
            lock(&bad.0.state).unwrap().backward =
                BackwardState::Ready(Recipe::Reduce(source.clone(), vec![], false, false));
        }
        result.backward(None, false).unwrap();
        assert_eq!(sibling.grad().unwrap().unwrap().values(), &[2.]);
        assert_eq!(other_sibling.grad().unwrap().unwrap().values(), &[2.]);
        assert_eq!(source.grad().unwrap().unwrap().values(), &[1.]);
        assert!(result.backward(None, false).is_err());
    }

    #[test]
    fn wrong_shaped_vjp_cannot_publish_or_broadcast_a_leaf_gradient() {
        let source = TensorHandle::new(TensorData::new(vec![1.; 6], vec![2, 3]).unwrap(), true);
        let bad = TensorHandle::result(
            TensorData::new(vec![1.; 6], vec![2, 3]).unwrap(),
            Recipe::Unary(UnaryOp::Transpose, source.clone()),
        );
        // A valid transpose output would be [3, 2]. This saved shape injects
        // a [3, 2] VJP for a [2, 3] parent, with an identical element count.
        let seed = TensorData::new(vec![1.; 6], vec![2, 3]).unwrap();
        assert!(bad.backward(Some(&seed), false).is_err());
        assert!(source.grad().unwrap().is_none());
    }
    #[test]
    fn resumable_custom_vjp_commits_and_native_driver_refuses_callback() {
        let x = scalar(2.);
        let y = TensorHandle::custom(
            TensorData::new(vec![4.], vec![]).unwrap(),
            vec![x.clone()],
            std::slice::from_ref(&x),
            17,
        )
        .unwrap();
        let mut run = BackwardSession::new(&y, None, false).unwrap();
        match run.advance().unwrap() {
            BackwardStep::Custom {
                callback_id,
                gradient,
            } => {
                assert_eq!(callback_id, 17);
                assert_eq!(gradient.values(), &[1.]);
            }
            _ => panic!("missing custom VJP"),
        }
        assert!(x.grad().unwrap().is_none());
        assert!(
            y.backward(None, false)
                .unwrap_err()
                .contains("Tensor.backward")
        );
        run.resume(vec![Some(TensorData::new(vec![4.], vec![]).unwrap())])
            .unwrap();
        assert!(matches!(run.advance().unwrap(), BackwardStep::Done));
        assert_eq!(x.grad().unwrap().unwrap().values(), &[4.]);
        assert!(y.backward(None, false).is_err());
    }

    #[test]
    fn invalid_custom_gradient_and_abort_allow_graph_retry() {
        let x = scalar(2.);
        let y = TensorHandle::custom(x.value().clone(), vec![x.clone()], &[], 18).unwrap();
        let mut run = BackwardSession::new(&y, None, false).unwrap();
        run.advance().unwrap();
        assert!(
            run.resume(vec![Some(TensorData::new(vec![1.], vec![1]).unwrap())])
                .is_err()
        );
        assert!(x.grad().unwrap().is_none());
        let mut retry = BackwardSession::new(&y, None, false).unwrap();
        retry.advance().unwrap();
        retry.abort().unwrap();
        let mut final_run = BackwardSession::new(&y, None, false).unwrap();
        final_run.advance().unwrap();
        final_run.resume(vec![None]).unwrap();
        assert!(matches!(final_run.advance().unwrap(), BackwardStep::Done));
        assert!(x.grad().unwrap().is_none());
    }

    #[test]
    fn custom_saved_versions_rechecked_after_callback_boundary() {
        let x = scalar(2.);
        let frozen = TensorHandle::new(TensorData::new(vec![3.], vec![]).unwrap(), false);
        let y = TensorHandle::custom(
            x.value().clone(),
            vec![x.clone()],
            std::slice::from_ref(&frozen),
            19,
        )
        .unwrap();
        let mut run = BackwardSession::new(&y, None, false).unwrap();
        run.advance().unwrap();
        replace_batch(&[frozen], vec![TensorData::new(vec![7.], vec![]).unwrap()]).unwrap();
        assert!(
            run.resume(vec![Some(TensorData::new(vec![1.], vec![]).unwrap())])
                .unwrap_err()
                .contains("modified")
        );
        assert!(x.grad().unwrap().is_none());
    }
    #[test]
    fn concurrent_custom_sessions_cannot_commit_twice() {
        let x = scalar(2.);
        let y = TensorHandle::custom(x.value().clone(), vec![x.clone()], &[], 20).unwrap();
        let mut first = BackwardSession::new(&y, None, false).unwrap();
        let mut second = BackwardSession::new(&y, None, false).unwrap();
        first.advance().unwrap();
        second.advance().unwrap();
        first.resume(vec![Some(x.value().clone())]).unwrap();
        assert!(matches!(first.advance().unwrap(), BackwardStep::Done));
        assert!(
            second
                .resume(vec![Some(x.value().clone())])
                .unwrap_err()
                .contains("consumed")
        );
        assert_eq!(x.grad().unwrap().unwrap().values(), &[2.]);
    }

    #[test]
    fn saved_snapshot_does_not_retain_original_graph_node() {
        let x = scalar(2.);
        let snapshot = x.saved_snapshot().unwrap();
        assert_eq!(Arc::strong_count(&x.0), 1);
        assert!(snapshot.is_leaf());
        assert!(!snapshot.requires_grad());
        assert!(!snapshot.has_custom());
        assert_eq!(snapshot.value().values(), &[2.]);
    }
    #[test]
    fn weak_identity_does_not_retain_consumed_ancestors() {
        let x = scalar(2.);
        let identity = x.identity();
        let y = x.binary(&x, BinaryOp::Mul).unwrap();
        drop(x);
        assert!(identity.alive());
        y.backward(None, false).unwrap();
        assert!(!identity.alive());
    }

    #[test]
    fn hook_sessions_yield_once_before_vjp_and_leaf_publication() {
        let x = scalar(2.);
        let y = x.binary(&x, BinaryOp::Mul).unwrap();
        let ids = [x.id(), y.id()].into_iter().collect();
        let mut run = BackwardSession::with_hooks(&y, None, false, ids).unwrap();
        assert!(
            matches!(run.advance().unwrap(), BackwardStep::Hook { node_id, .. } if node_id == y.id())
        );
        assert!(
            run.check_hook(&TensorData::new(vec![1.], vec![1]).unwrap())
                .is_err()
        );
        assert!(x.grad().unwrap().is_none());
        run.resume_hook(TensorData::new(vec![2.], vec![]).unwrap())
            .unwrap();
        match run.advance().unwrap() {
            BackwardStep::Hook { node_id, gradient } => {
                assert_eq!(node_id, x.id());
                assert_eq!(gradient.values(), &[8.]);
            }
            _ => panic!("missing leaf hook"),
        }
        run.resume_hook(TensorData::new(vec![9.], vec![]).unwrap())
            .unwrap();
        assert!(matches!(run.advance().unwrap(), BackwardStep::Done));
        assert_eq!(x.grad().unwrap().unwrap().values(), &[9.]);
    }
    #[test]
    fn moved_compiled_gradients_accumulate_duplicate_parents_and_retain_snapshots() {
        let x = scalar(2.);
        let mut graph = crate::graph::Graph::new(2);
        graph.binary(0, 1, BinaryOp::Mul).unwrap();
        let plan = Arc::new(
            graph
                .compile_backward(&[vec![], vec![]], &[true, true])
                .unwrap(),
        );
        let mut workspace = plan.create_workspace().unwrap();
        let inputs = vec![x.clone(), x.clone()];
        let value = plan.forward(&inputs, &mut workspace).unwrap();
        let output = TensorHandle::compiled(value, plan, workspace, inputs);
        output.backward(None, true).unwrap();
        let first = x.grad().unwrap().unwrap();
        assert_eq!(first.values(), &[4.]);
        output.backward(None, false).unwrap();
        assert_eq!(x.grad().unwrap().unwrap().values(), &[8.]);
        assert_eq!(first.values(), &[4.]);
    }
}

#[cfg(test)]
mod leaf_gradient_payload_study {
    use super::*;

    #[test]
    fn retained_snapshot_and_cumulative_leaf_grad_payload() {
        for elements in [64, 12_448, 525_568] {
            let source = TensorHandle::new(
                TensorData::new(vec![2.; elements], vec![elements]).unwrap(),
                true,
            );
            let mut graph = crate::graph::Graph::new(2);
            let product = graph.binary(0, 1, BinaryOp::Mul).unwrap();
            graph.reduce(product, false).unwrap();
            let plan = Arc::new(
                graph
                    .compile_backward(&[vec![elements], vec![elements]], &[true, true])
                    .unwrap(),
            );
            let mut workspace = plan.create_workspace().unwrap();
            let inputs = vec![source.clone(), source.clone()];
            let value = plan.forward(&inputs, &mut workspace).unwrap();
            let output = TensorHandle::compiled(value, plan, workspace, inputs);

            reset_accumulate_payload_peak();
            output.backward(None, true).unwrap();
            let first = source.grad().unwrap().unwrap();
            assert_eq!(first.values(), &vec![4.; elements]);
            let first_bytes = first.storage_capacity_bytes();
            let first_identity = first.storage_identity();
            let accumulate_peak_bytes = accumulate_payload_peak();
            assert_eq!(accumulate_peak_bytes, first_bytes * 3);
            assert_eq!(accumulate_merge_count(), 1);

            output.backward(None, false).unwrap();
            let current = source.grad().unwrap().unwrap();
            assert_eq!(current.values(), &vec![8.; elements]);
            assert_eq!(first.values(), &vec![4.; elements]);
            assert_ne!(first_identity, current.storage_identity());
            let retained_and_current_bytes = first_bytes + current.storage_capacity_bytes();

            println!(
                "leaf_gradient_payload {{\"elements\":{elements},\"one_gradient_storage_bytes\":{first_bytes},\"old_snapshot_and_current_gradient_bytes\":{retained_and_current_bytes},\"duplicate_parent_merge_live_payload_bytes\":{accumulate_peak_bytes}}}"
            );
            drop(first);
            assert_eq!(current.storage_capacity_bytes(), first_bytes);
        }
    }

    #[test]
    fn wide_branch_merge_payload_deduplicates_shared_seed_storage() {
        for elements in [64, 12_448, 525_568] {
            let storage_tracker = crate::tensor::TensorStorageTracker::start();
            let source = TensorHandle::new(
                TensorData::new(vec![2.; elements], vec![elements]).unwrap(),
                true,
            );
            let mut total: Option<TensorHandle> = None;
            for _ in 0..4 {
                let product = source.binary(&source, BinaryOp::Mul).unwrap();
                total = Some(match total {
                    Some(previous) => previous.binary(&product, BinaryOp::Add).unwrap(),
                    None => product,
                });
            }
            let output = total.unwrap().reduce_axes(vec![0], false, false).unwrap();

            let forward_live_bytes = storage_tracker.live_capacity_bytes();
            storage_tracker.reset_peak();
            reset_accumulate_payload_peak();
            output.backward(None, false).unwrap();
            let backward_peak_bytes = storage_tracker.peak_capacity_bytes();
            let gradient = source.grad().unwrap().unwrap();
            assert_eq!(gradient.values(), &vec![16.; elements]);
            assert_eq!(accumulate_merge_count(), 7);
            let peak = accumulate_payload_peak();
            assert_eq!(peak, gradient.storage_capacity_bytes() * 4);
            println!(
                "wide_branch_merge_payload {{\"elements\":{elements},\"merge_count\":{},\"unique_live_capacity_bytes\":{peak},\"forward_live_tensor_bytes\":{forward_live_bytes},\"backward_live_tensor_peak_bytes\":{backward_peak_bytes}}}",
                accumulate_merge_count(),
            );
        }
    }
}

#[cfg(test)]
mod backward_family_payload_study {
    use super::*;
    use crate::tensor::TensorStorageTracker;

    #[test]
    fn matmul_backward_tensor_storage_peak() {
        let tracker = TensorStorageTracker::start();
        let left = TensorHandle::new(
            TensorData::new(vec![1.; 32 * 128], vec![32, 128]).unwrap(),
            true,
        );
        let right = TensorHandle::new(
            TensorData::new(vec![0.5; 128 * 64], vec![128, 64]).unwrap(),
            true,
        );
        let output = left
            .matmul(&right)
            .unwrap()
            .reduce_axes(vec![0, 1], false, false)
            .unwrap();

        let forward_live_bytes = tracker.live_capacity_bytes();
        tracker.reset_peak();
        output.backward(None, false).unwrap();
        let backward_peak_bytes = tracker.peak_capacity_bytes();
        let left_gradient = left.grad().unwrap().unwrap();
        let right_gradient = right.grad().unwrap().unwrap();
        assert!(left_gradient.values().iter().all(|&value| value == 32.));
        assert!(right_gradient.values().iter().all(|&value| value == 32.));
        println!(
            "matmul_backward_payload {{\"shape\":[32,128,64],\"forward_live_tensor_bytes\":{forward_live_bytes},\"backward_live_tensor_peak_bytes\":{backward_peak_bytes}}}"
        );
    }

    #[test]
    fn layer_norm_backward_tensor_storage_peak() {
        let tracker = TensorStorageTracker::start();
        let input = TensorHandle::new(
            TensorData::new(
                (0..32 * 128).map(|index| (index % 128) as f32).collect(),
                vec![32, 128],
            )
            .unwrap(),
            true,
        );
        let weight = TensorHandle::new(TensorData::new(vec![1.; 128], vec![128]).unwrap(), true);
        let bias = TensorHandle::new(TensorData::new(vec![0.; 128], vec![128]).unwrap(), true);
        let output = input
            .layer_norm(Some(&weight), Some(&bias), 1e-5)
            .unwrap()
            .reduce_axes(vec![0, 1], false, false)
            .unwrap();

        let forward_live_bytes = tracker.live_capacity_bytes();
        tracker.reset_peak();
        output.backward(None, false).unwrap();
        let backward_peak_bytes = tracker.peak_capacity_bytes();
        for tensor in [&input, &weight, &bias] {
            let gradient = tensor.grad().unwrap().unwrap();
            assert!(gradient.values().iter().all(|value| value.is_finite()));
        }
        assert!(
            bias.grad()
                .unwrap()
                .unwrap()
                .values()
                .iter()
                .all(|&value| value == 32.)
        );
        println!(
            "layer_norm_backward_payload {{\"shape\":[32,128],\"forward_live_tensor_bytes\":{forward_live_bytes},\"backward_live_tensor_peak_bytes\":{backward_peak_bytes}}}"
        );
    }

    #[test]
    fn broadcast_backward_tensor_storage_peak() {
        let tracker = TensorStorageTracker::start();
        let input = TensorHandle::new(
            TensorData::new(vec![1.; 32 * 128], vec![32, 128]).unwrap(),
            true,
        );
        let bias = TensorHandle::new(TensorData::new(vec![0.5; 128], vec![128]).unwrap(), true);
        let output = input
            .binary(&bias, BinaryOp::Add)
            .unwrap()
            .reduce_axes(vec![0, 1], false, false)
            .unwrap();

        let forward_live_bytes = tracker.live_capacity_bytes();
        tracker.reset_peak();
        output.backward(None, false).unwrap();
        let backward_peak_bytes = tracker.peak_capacity_bytes();
        assert!(
            input
                .grad()
                .unwrap()
                .unwrap()
                .values()
                .iter()
                .all(|&value| value == 1.)
        );
        assert!(
            bias.grad()
                .unwrap()
                .unwrap()
                .values()
                .iter()
                .all(|&value| value == 32.)
        );
        println!(
            "broadcast_backward_payload {{\"shape\":[32,128],\"bias_shape\":[128],\"forward_live_tensor_bytes\":{forward_live_bytes},\"backward_live_tensor_peak_bytes\":{backward_peak_bytes}}}"
        );
    }

    #[test]
    fn strided_backward_tensor_storage_peak() {
        let tracker = TensorStorageTracker::start();
        let source = TensorHandle::new(
            TensorData::new(
                (0..32 * 256).map(|index| index as f32).collect(),
                vec![32, 256],
            )
            .unwrap(),
            true,
        );
        let view = source
            .slice(vec![
                SliceSpec {
                    start: 0,
                    step: 1,
                    len: 32,
                    squeeze: false,
                },
                SliceSpec {
                    start: 0,
                    step: 2,
                    len: 128,
                    squeeze: false,
                },
            ])
            .unwrap();
        let output = view.reduce_axes(vec![0, 1], false, false).unwrap();

        let forward_live_bytes = tracker.live_capacity_bytes();
        tracker.reset_peak();
        output.backward(None, false).unwrap();
        let backward_peak_bytes = tracker.peak_capacity_bytes();
        let gradient = source.grad().unwrap().unwrap();
        for row in 0..32 {
            for column in 0..256 {
                let expected = if column % 2 == 0 { 1. } else { 0. };
                assert_eq!(gradient.values()[row * 256 + column], expected);
            }
        }
        println!(
            "strided_backward_payload {{\"source_shape\":[32,256],\"view_shape\":[32,128],\"step\":2,\"forward_live_tensor_bytes\":{forward_live_bytes},\"backward_live_tensor_peak_bytes\":{backward_peak_bytes}}}"
        );
    }

    #[test]
    fn hook_callback_session_tensor_storage_peak() {
        let elements = 32 * 128;
        let shape = vec![32, 128];
        let tracker = TensorStorageTracker::start();
        let source = TensorHandle::new(
            TensorData::new(vec![2.; elements], shape.clone()).unwrap(),
            true,
        );
        let output = source.binary(&source, BinaryOp::Mul).unwrap();
        let hooks = [source.id(), output.id()].into_iter().collect();
        let seed = TensorData::new(vec![1.; elements], shape.clone()).unwrap();
        let mut session = BackwardSession::with_hooks(&output, Some(&seed), false, hooks).unwrap();
        let before_callbacks_bytes = tracker.live_capacity_bytes();
        tracker.reset_peak();

        let first_upstream = match session.advance().unwrap() {
            BackwardStep::Hook { node_id, gradient } => {
                assert_eq!(node_id, output.id());
                gradient
            }
            _ => panic!("missing output hook pause"),
        };
        assert!(first_upstream.values().iter().all(|&value| value == 1.));
        let replacement = TensorData::new(vec![2.; elements], shape.clone()).unwrap();
        session.resume_hook(replacement).unwrap();

        let second_upstream = match session.advance().unwrap() {
            BackwardStep::Hook { node_id, gradient } => {
                assert_eq!(node_id, source.id());
                gradient
            }
            _ => panic!("missing leaf hook pause"),
        };
        assert!(second_upstream.values().iter().all(|&value| value == 8.));
        session
            .resume_hook(TensorData::new(vec![9.; elements], shape).unwrap())
            .unwrap();
        assert!(matches!(session.advance().unwrap(), BackwardStep::Done));
        let peak = tracker.peak_capacity_bytes();
        assert!(
            source
                .grad()
                .unwrap()
                .unwrap()
                .values()
                .iter()
                .all(|&value| value == 9.)
        );
        println!(
            "hook_callback_payload {{\"shape\":[32,128],\"before_callbacks_live_tensor_bytes\":{before_callbacks_bytes},\"backward_live_tensor_peak_bytes\":{peak}}}"
        );
    }

    #[test]
    fn custom_vjp_callback_session_tensor_storage_peak() {
        let elements = 32 * 128;
        let shape = vec![32, 128];
        let tracker = TensorStorageTracker::start();
        let source = TensorHandle::new(
            TensorData::new(vec![2.; elements], shape.clone()).unwrap(),
            true,
        );
        let output =
            TensorHandle::custom(source.value().clone(), vec![source.clone()], &[], 92).unwrap();
        let seed = TensorData::new(vec![1.; elements], shape.clone()).unwrap();
        let mut session = BackwardSession::new(&output, Some(&seed), false).unwrap();
        let before_callback_bytes = tracker.live_capacity_bytes();
        tracker.reset_peak();

        let upstream = match session.advance().unwrap() {
            BackwardStep::Custom {
                callback_id,
                gradient,
            } => {
                assert_eq!(callback_id, 92);
                gradient
            }
            _ => panic!("missing custom VJP pause"),
        };
        assert!(upstream.values().iter().all(|&value| value == 1.));
        session
            .resume(vec![Some(
                TensorData::new(vec![2.; elements], shape).unwrap(),
            )])
            .unwrap();
        assert!(matches!(session.advance().unwrap(), BackwardStep::Done));
        let peak = tracker.peak_capacity_bytes();
        assert!(
            source
                .grad()
                .unwrap()
                .unwrap()
                .values()
                .iter()
                .all(|&value| value == 2.)
        );
        println!(
            "custom_vjp_callback_payload {{\"shape\":[32,128],\"before_callback_live_tensor_bytes\":{before_callback_bytes},\"backward_live_tensor_peak_bytes\":{peak}}}"
        );
    }
}
