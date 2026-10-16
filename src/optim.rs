use crate::{
    autograd::{self, TensorHandle},
    tensor::TensorData,
};
use std::collections::HashSet;
#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Sgd,
    Adam,
    /// Adam with decoupled weight decay (Loshchilov & Hutter), as torch.optim.AdamW.
    AdamW,
}
#[derive(Clone)]
pub struct Config {
    pub kind: Kind,
    pub lr: f64,
    pub momentum: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}
impl Config {
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("lr", self.lr),
            ("momentum", self.momentum),
            ("weight_decay", self.weight_decay),
            ("eps", self.eps),
        ] {
            if !v.is_finite() || v < 0.0 || v > f32::MAX as f64 || (v != 0.0 && v as f32 == 0.0) {
                return Err(format!(
                    "{name} must be finite, nonnegative and representable in float32"
                ));
            }
        }
        if self.eps <= 0.0 || self.eps as f32 == 0.0 {
            return Err("eps must be positive in float32".into());
        }
        for v in [self.beta1, self.beta2] {
            if !v.is_finite() || !(0.0..1.0).contains(&v) || v as f32 >= 1.0 {
                return Err("betas must be in [0,1) and below one in float32".into());
            }
        }
        Ok(())
    }
}
#[derive(Clone, Default)]
pub struct State {
    pub step: u64,
    pub first: Vec<f32>,
    pub second: Vec<f32>,
}
pub(crate) struct PreparedUpdate {
    parameters: Vec<TensorHandle>,
    states: Vec<State>,
}
impl PreparedUpdate {
    pub(crate) fn parameters(&self) -> &[TensorHandle] {
        &self.parameters
    }
}
pub struct Optimizer {
    config: Config,
    shapes: Vec<Vec<usize>>,
    ids: Vec<usize>,
    states: Vec<State>,
}
impl Optimizer {
    pub fn new(config: Config, params: &[TensorHandle]) -> Result<Self, String> {
        config.validate()?;
        let _engine = autograd::engine_gate()?;
        if params.is_empty() {
            return Err("optimizer requires at least one parameter".into());
        }
        let mut ids = HashSet::new();
        for p in params {
            p.check_current()?;
            if !p.is_leaf() || !ids.insert(p.logical_id()) {
                return Err("optimizer requires unique leaf parameters".into());
            }
        }
        Ok(Self {
            config,
            shapes: params.iter().map(|p| p.value().shape().to_vec()).collect(),
            ids: params.iter().map(|p| p.logical_id()).collect(),
            states: vec![State::default(); params.len()],
        })
    }
    /// Change the learning rate between steps; state buffers are kept.
    pub fn set_lr(&mut self, lr: f64) -> Result<(), String> {
        self.set_hyper("lr", lr)
    }
    /// Change one hyperparameter between steps (as an lr scheduler edits param_groups).
    pub fn set_hyper(&mut self, name: &str, value: f64) -> Result<(), String> {
        let mut config = self.config.clone();
        match name {
            "lr" => config.lr = value,
            "momentum" => config.momentum = value,
            "beta1" => config.beta1 = value,
            "beta2" => config.beta2 = value,
            "eps" => config.eps = value,
            "weight_decay" => config.weight_decay = value,
            _ => return Err(format!("unknown optimizer hyperparameter {name}")),
        }
        config.validate()?;
        self.config = config;
        Ok(())
    }
    pub fn snapshot(&self) -> Vec<State> {
        self.states.clone()
    }
    pub fn load(&mut self, states: Vec<State>) -> Result<(), String> {
        if states.len() != self.shapes.len() {
            return Err("optimizer state count mismatch".into());
        }
        for (s, shape) in states.iter().zip(&self.shapes) {
            let n = shape.iter().product();
            let first =
                if s.step == 0 || (self.config.kind == Kind::Sgd && self.config.momentum == 0.0) {
                    0
                } else {
                    n
                };
            let second = if s.step > 0 && self.config.kind != Kind::Sgd {
                n
            } else {
                0
            };
            if s.first.len() != first || s.second.len() != second {
                return Err("optimizer state buffer size mismatch".into());
            }
        }
        self.states = states;
        Ok(())
    }
    pub fn step(&mut self, params: &[TensorHandle]) -> Result<Vec<TensorHandle>, String> {
        let _engine = autograd::engine_gate()?;
        let gradients = params
            .iter()
            .map(TensorHandle::grad_unlocked)
            .collect::<Result<Vec<_>, _>>()?;
        let borrowed: Vec<Option<&[f32]>> = gradients
            .iter()
            .map(|g| g.as_ref().map(TensorData::values))
            .collect();
        self.step_with_gradients_unlocked(params, &borrowed)
    }
    /// Apply explicit VJP gradients without mutating eager leaf gradient storage.
    #[allow(dead_code)] // Internal compiled-gradient entry point; eager uses stored gradients.
    pub fn step_with_gradients(
        &mut self,
        params: &[TensorHandle],
        gradients: &[Option<&[f32]>],
    ) -> Result<Vec<TensorHandle>, String> {
        let _engine = autograd::engine_gate()?;
        self.step_with_gradients_unlocked(params, gradients)
    }
    pub(crate) fn step_with_gradients_unlocked(
        &mut self,
        params: &[TensorHandle],
        gradients: &[Option<&[f32]>],
    ) -> Result<Vec<TensorHandle>, String> {
        let prepared = self.prepare_with_gradients_unlocked(params, gradients)?;
        Ok(self.commit_prepared_unlocked(prepared))
    }
    pub(crate) fn prepare_with_gradients_unlocked(
        &self,
        params: &[TensorHandle],
        gradients: &[Option<&[f32]>],
    ) -> Result<PreparedUpdate, String> {
        if gradients.len() != params.len() {
            return Err("optimizer gradient count mismatch".into());
        }
        if params.len() != self.shapes.len() {
            return Err("optimizer parameter count mismatch".into());
        }
        for ((p, shape), id) in params.iter().zip(&self.shapes).zip(&self.ids) {
            p.check_current()?;
            if !p.is_leaf() || p.logical_id() != *id || p.value().shape() != shape {
                return Err("optimizer parameter identity or shape changed".into());
            }
        }
        for (p, gradient) in params.iter().zip(gradients) {
            if gradient.is_some_and(|g| g.len() != p.value().values().len()) {
                return Err("optimizer gradient size mismatch".into());
            }
        }
        let mut next_states = self.states.clone();
        let mut values = Vec::with_capacity(params.len());
        for (i, p) in params.iter().enumerate() {
            let Some(grad) = gradients[i] else {
                values.push(None);
                continue;
            };
            p.validate_replace(p.value())?;
            let state = &mut next_states[i];
            state.step = state.step.checked_add(1).ok_or("optimizer step overflow")?;
            let g: std::borrow::Cow<'_, [f32]> =
                if self.config.weight_decay == 0.0 || self.config.kind == Kind::AdamW {
                    std::borrow::Cow::Borrowed(grad)
                } else {
                    std::borrow::Cow::Owned(
                        // grad.add(param, alpha=weight_decay): fmadd(param, alpha, grad).
                        grad.iter()
                            .zip(p.value().values())
                            .map(|(&g, &p)| p.mul_add(self.config.weight_decay as f32, g))
                            .collect(),
                    )
                };
            let mut data = p.value().values().to_vec();
            if self.config.kind == Kind::AdamW && self.config.weight_decay != 0.0 {
                // PyTorch: param.mul_(1 - lr * weight_decay), the factor rounded to float32.
                let keep = (1.0 - self.config.lr * self.config.weight_decay) as f32;
                data.iter_mut().for_each(|p| *p *= keep);
            }
            match self.config.kind {
                Kind::Sgd => {
                    let update: &[f32] = if self.config.momentum > 0.0 {
                        // torch creates the buffer the first step momentum is in use.
                        if state.first.len() != g.len() {
                            state.first = g.to_vec();
                        } else {
                            for (m, &g) in state.first.iter_mut().zip(g.iter()) {
                                *m = (self.config.momentum as f32) * *m + g;
                            }
                        }
                        &state.first
                    } else {
                        &g
                    };
                    // param.add_(update, alpha=-lr): fmadd(update, -lr, param).
                    let alpha = -self.config.lr as f32;
                    for (p, &g) in data.iter_mut().zip(update) {
                        *p = g.mul_add(alpha, *p);
                    }
                }
                Kind::Adam | Kind::AdamW => {
                    if state.step == 1 {
                        state.first = vec![0.0; g.len()];
                        state.second = vec![0.0; g.len()];
                    }
                    // torch.optim's _single_tensor_adam (the CPU default), op by op.
                    let pow = crate::sleef::libm_pow;
                    let step = state.step as f64;
                    let step_size = self.config.lr / (1.0 - pow(self.config.beta1, step));
                    let bc2_sqrt = pow(1.0 - pow(self.config.beta2, step), 0.5) as f32;
                    let weight = (1.0 - self.config.beta1) as f32;
                    let (beta2, value) =
                        (self.config.beta2 as f32, (1.0 - self.config.beta2) as f32);
                    let (eps, neg_step) = (self.config.eps as f32, -step_size as f32);
                    // exp_avg.lerp_(grad, 1 - beta1)
                    for (m, &g) in state.first.iter_mut().zip(g.iter()) {
                        *m = if weight.abs() < 0.5 {
                            weight.mul_add(g - *m, *m)
                        } else {
                            (weight - 1.0).mul_add(g - *m, g)
                        };
                    }
                    // exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2): vector
                    // groups of 8 round each step; each thread chunk's scalar tail is fused.
                    for (b, e) in crate::sleef::parallel_chunks(g.len(), 32768) {
                        let whole = b + (e - b) / 8 * 8;
                        for i in b..e {
                            let t = value * g[i];
                            let v = &mut state.second[i];
                            *v = if i < whole {
                                *v * beta2 + t * g[i]
                            } else {
                                t.mul_add(g[i], *v * beta2)
                            };
                        }
                    }
                    // param.addcdiv_(exp_avg, exp_avg_sq.sqrt() / bc2_sqrt + eps, value=-step_size)
                    for ((p, &m), &v) in data.iter_mut().zip(&state.first).zip(&state.second) {
                        *p += neg_step * m / (v.sqrt() / bc2_sqrt + eps);
                    }
                }
            }
            values.push(Some(TensorData::new(data, p.value().shape().to_vec())?));
        }
        let replacements = params
            .iter()
            .zip(values)
            .map(|(p, v)| match v {
                Some(v) => p.prepare_replacement_unlocked(v),
                None => Ok(p.clone()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PreparedUpdate {
            parameters: replacements,
            states: next_states,
        })
    }
    pub(crate) fn commit_prepared_unlocked(
        &mut self,
        prepared: PreparedUpdate,
    ) -> Vec<TensorHandle> {
        for p in &prepared.parameters {
            p.commit_replacement();
        }
        self.states = prepared.states;
        prepared.parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parameter(v: f32) -> TensorHandle {
        TensorHandle::new(TensorData::new(vec![v], vec![1]).unwrap(), true)
    }
    fn config() -> Config {
        Config {
            kind: Kind::Sgd,
            lr: 0.1,
            momentum: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
    #[test]
    fn borrowed_gradient_update_preserves_leaf_gradients_and_failure_atomicity() {
        let a = parameter(1.0);
        let b = parameter(2.0);
        let seed = TensorData::new(vec![7.0], vec![1]).unwrap();
        a.backward(Some(&seed), false).unwrap();
        let mut optimizer = Optimizer::new(config(), &[a.clone(), b.clone()]).unwrap();
        assert!(
            optimizer
                .step_with_gradients(
                    &[a.clone(), b.clone()],
                    &[Some(&[2.0][..]), Some(&[1.0, 2.0][..])]
                )
                .is_err()
        );
        assert_eq!(optimizer.snapshot()[0].step, 0);
        assert_eq!(a.grad().unwrap().unwrap().values(), &[7.0]);
        let result = optimizer
            .step_with_gradients(&[a, b], &[Some(&[2.0]), None])
            .unwrap();
        assert_eq!(result[0].value().values(), &[0.8]);
        assert_eq!(result[1].value().values(), &[2.0]);
        assert_eq!(result[0].grad().unwrap().unwrap().values(), &[7.0]);
        assert_eq!(optimizer.snapshot()[0].step, 1);
        assert_eq!(optimizer.snapshot()[1].step, 0);
    }
    #[test]
    fn discarded_prepared_update_preserves_versions_and_state() {
        let parameter = parameter(1.0);
        let mut optimizer = Optimizer::new(config(), std::slice::from_ref(&parameter)).unwrap();
        let _gate = autograd::engine_gate().unwrap();
        let prepared = optimizer
            .prepare_with_gradients_unlocked(std::slice::from_ref(&parameter), &[Some(&[2.0])])
            .unwrap();
        assert_eq!(prepared.parameters()[0].value().values(), &[0.8]);
        drop(prepared); // Models a Python allocation failure before commit.
        parameter.check_current().unwrap();
        assert_eq!(parameter.value().values(), &[1.0]);
        assert_eq!(optimizer.snapshot()[0].step, 0);
        let prepared = optimizer
            .prepare_with_gradients_unlocked(std::slice::from_ref(&parameter), &[Some(&[2.0])])
            .unwrap();
        let updated = optimizer.commit_prepared_unlocked(prepared);
        assert!(parameter.check_current().is_err());
        updated[0].check_current().unwrap();
        assert_eq!(optimizer.snapshot()[0].step, 1);
    }
    #[test]
    fn failed_later_parameter_leaves_batch_and_state_unchanged() {
        let a = parameter(1.0);
        let b = parameter(2.0);
        let seed = TensorData::new(vec![1.0], vec![1]).unwrap();
        a.backward(Some(&seed), false).unwrap();
        b.backward(Some(&seed), false).unwrap();
        let mut engine = Optimizer::new(config(), &[a.clone(), b.clone()]).unwrap();
        engine
            .load(vec![
                State::default(),
                State {
                    step: u64::MAX,
                    first: vec![],
                    second: vec![],
                },
            ])
            .unwrap();
        assert!(engine.step(&[a.clone(), b.clone()]).is_err());
        a.check_current().unwrap();
        b.check_current().unwrap();
        assert_eq!(engine.snapshot()[0].step, 0);
        assert_eq!(engine.snapshot()[1].step, u64::MAX);
        assert_eq!(a.value().values(), &[1.0]);
    }
    #[test]
    fn stale_handle_cannot_overwrite_updated_parameter() {
        let a = parameter(1.0);
        a.backward(None, false).unwrap();
        let mut engine = Optimizer::new(config(), std::slice::from_ref(&a)).unwrap();
        let updated = engine.step(std::slice::from_ref(&a)).unwrap();
        assert!(engine.step(std::slice::from_ref(&a)).is_err());
        updated[0].check_current().unwrap();
        assert_eq!(engine.snapshot()[0].step, 1);
    }
}
