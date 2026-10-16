//! Resumable backward: no Python objects or guards cross a suspension boundary.
use super::*;

pub(crate) enum BackwardStep {
    Done,
    Hook {
        node_id: usize,
        gradient: TensorData,
    },
    Custom {
        callback_id: u64,
        gradient: TensorData,
    },
}
pub(crate) struct BackwardSession {
    order: Vec<TensorHandle>,
    recipes: HashMap<usize, Recipe>,
    grads: HashMap<usize, TensorData>,
    staged: Vec<(TensorHandle, TensorData)>,
    cursor: usize,
    pending: Option<Arc<CustomVjp>>,
    hook_ids: HashSet<usize>,
    hook_pending: Option<TensorHandle>,
    ready: Option<(TensorHandle, TensorData)>,
    retain_graph: bool,
    closed: bool,
}
impl BackwardSession {
    pub(crate) fn new(
        root: &TensorHandle,
        seed: Option<&TensorData>,
        retain_graph: bool,
    ) -> Result<Self, String> {
        Self::with_hooks(root, seed, retain_graph, HashSet::new())
    }
    pub(crate) fn with_hooks(
        root: &TensorHandle,
        seed: Option<&TensorData>,
        retain_graph: bool,
        hook_ids: HashSet<usize>,
    ) -> Result<Self, String> {
        let _engine = engine_gate()?;
        if !root.requires_grad() {
            return Err("tensor does not require gradients".into());
        }
        let seed = if let Some(seed) = seed {
            if seed.shape() != root.value().shape() {
                return Err("gradient shape must exactly match output".into());
            }
            seed.clone()
        } else {
            if root.value().values().len() != 1 {
                return Err("non-single-element output requires an explicit gradient".into());
            }
            TensorData::new(vec![1.0], root.value().shape().to_vec())?
        };
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut recipes = HashMap::new();
        let mut stack = vec![(root.clone(), false)];
        while let Some((node, exit)) = stack.pop() {
            if exit {
                order.push(node);
                continue;
            }
            if !visited.insert(node.id()) {
                continue;
            }
            node.check_current()?;
            if node.0.has_hooks.load(Ordering::SeqCst) && !hook_ids.contains(&node.id()) {
                return Err("hook graphs require Tensor.backward Python driver".into());
            }
            let state = lock(&node.0.state)?.backward.clone();
            if matches!(state, BackwardState::Consumed) {
                return Err(
                    "backward graph already consumed; use retain_graph=True on the earlier call"
                        .into(),
                );
            }
            stack.push((node.clone(), true));
            if let BackwardState::Ready(recipe) = state {
                for p in recipe.parents().into_iter().rev() {
                    p.check_current()?;
                    if p.requires_grad() {
                        stack.push((p, false));
                    }
                }
                recipe.check_saved()?;
                recipes.insert(node.id(), recipe);
            }
        }
        let mut grads = HashMap::new();
        grads.insert(root.id(), seed);
        let cursor = order.len();
        Ok(Self {
            order,
            recipes,
            grads,
            staged: Vec::new(),
            cursor,
            pending: None,
            hook_ids,
            hook_pending: None,
            ready: None,
            retain_graph,
            closed: false,
        })
    }
    fn validate(&self) -> Result<(), String> {
        if self.closed {
            return Err("backward session is closed".into());
        }
        for node in &self.order {
            node.check_current()?;
            if !node.is_leaf() && !matches!(lock(&node.0.state)?.backward, BackwardState::Ready(_))
            {
                return Err(
                    "backward graph already consumed; use retain_graph=True on the earlier call"
                        .into(),
                );
            }
        }
        for recipe in self.recipes.values() {
            recipe.check_saved()?;
            for parent in recipe.parents() {
                parent.check_current()?;
            }
        }
        Ok(())
    }
    fn contribute(&mut self, parent: &TensorHandle, gradient: TensorData) -> Result<(), String> {
        if gradient.shape() != parent.value().shape() {
            return Err("backward contribution shape must exactly match parent".into());
        }
        if parent.requires_grad() {
            accumulate(&mut self.grads, parent.id(), gradient)?;
        }
        Ok(())
    }
    pub(crate) fn saved(&self) -> Result<Vec<TensorHandle>, String> {
        self.pending
            .as_ref()
            .map(|p| p.saved.clone())
            .ok_or("no custom VJP pending".into())
    }
    pub(crate) fn advance(&mut self) -> Result<BackwardStep, String> {
        let result = {
            let _engine = engine_gate()?;
            self.advance_unlocked()
        };
        if result.is_err() {
            self.abort()?;
        }
        result
    }
    fn advance_unlocked(&mut self) -> Result<BackwardStep, String> {
        self.validate()?;
        if self.pending.is_some() || self.hook_pending.is_some() {
            return Err("resume custom gradients before advancing".into());
        }
        while self.cursor != 0 || self.ready.is_some() {
            let (node, gradient) = if let Some(ready) = self.ready.take() {
                ready
            } else {
                self.cursor -= 1;
                let node = self.order[self.cursor].clone();
                let Some(gradient) = self.grads.remove(&node.id()) else {
                    continue;
                };
                if self.hook_ids.contains(&node.id()) {
                    let node_id = node.id();
                    self.hook_pending = Some(node);
                    return Ok(BackwardStep::Hook { node_id, gradient });
                }
                (node, gradient)
            };
            if let Some(recipe) = self.recipes.get(&node.id()).cloned() {
                if let Recipe::Custom(custom) = recipe {
                    let callback_id = custom.id;
                    self.pending = Some(custom);
                    return Ok(BackwardStep::Custom {
                        callback_id,
                        gradient,
                    });
                }
                for (parent, contribution) in derivative(&recipe, &gradient)? {
                    self.contribute(&parent, contribution)?;
                }
            } else {
                self.staged.push((node, gradient));
            }
        }
        // Prepare all fallible locks and additions before publishing any leaf.
        // Staged handles keep the nodes alive for the guards borrowed below.
        let mut prepared = Vec::with_capacity(self.staged.len());
        for (node, gradient) in &self.staged {
            let state = lock(&node.0.state)?;
            let gradient = if let Some(old) = &state.grad {
                ops::binary(old, gradient, BinaryOp::Add)?
            } else {
                gradient.clone()
            };
            prepared.push((state, gradient));
        }
        let mut consumed = Vec::new();
        if !self.retain_graph {
            for node in &self.order {
                if !node.is_leaf() {
                    consumed.push(lock(&node.0.state)?);
                }
            }
        }
        // Commit contains no fallible calls. ENGINE still excludes observers.
        for (mut state, gradient) in prepared {
            state.grad = Some(gradient);
        }
        for mut state in consumed {
            state.backward = BackwardState::Consumed;
        }
        self.closed = true;
        Ok(BackwardStep::Done)
    }
    pub(crate) fn resume(&mut self, gradients: Vec<Option<TensorData>>) -> Result<(), String> {
        let result = {
            let _engine = engine_gate()?;
            self.resume_unlocked(gradients)
        };
        if result.is_err() {
            self.abort()?;
        }
        result
    }
    fn resume_unlocked(&mut self, gradients: Vec<Option<TensorData>>) -> Result<(), String> {
        self.validate()?;
        let pending = self.pending.take().ok_or("no custom VJP pending")?;
        if gradients.len() != pending.parents.len() {
            return Err("custom backward gradient count mismatch".into());
        }
        for (parent, gradient) in pending.parents.iter().zip(gradients) {
            if let Some(gradient) = gradient {
                self.contribute(parent, gradient)?;
            }
        }
        Ok(())
    }
    pub(crate) fn check_hook(&self, gradient: &TensorData) -> Result<(), String> {
        let _engine = engine_gate()?;
        self.validate()?;
        let node = self.hook_pending.as_ref().ok_or("no hook pending")?;
        if gradient.shape() != node.value().shape() {
            return Err("hook gradient shape must exactly match tensor".into());
        }
        Ok(())
    }
    pub(crate) fn resume_hook(&mut self, gradient: TensorData) -> Result<(), String> {
        self.check_hook(&gradient)?;
        let node = self.hook_pending.take().ok_or("no hook pending")?;
        self.ready = Some((node, gradient));
        Ok(())
    }
    pub(crate) fn abort(&mut self) -> Result<(), String> {
        self.closed = true;
        self.pending = None;
        self.hook_pending = None;
        self.ready = None;
        self.grads.clear();
        self.staged.clear();
        self.recipes.clear();
        self.order.clear();
        Ok(())
    }
}
