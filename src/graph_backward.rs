//! Shape-specialized backward schedules. Plans are immutable; workspaces own state.
use super::{ExecutionPlan, Graph, Instruction, workspace::ExecutionWorkspace};
use crate::{
    autograd::{self, TensorHandle},
    cpu_kernels,
    tensor::{DType, Device, TensorData, TensorSpec},
};
use std::sync::Arc;

pub struct BackwardPlan {
    forward: ExecutionPlan,
    needs_grad: Vec<bool>,
    order: Vec<usize>,
    identity: Arc<()>,
}

pub struct TrainingWorkspace {
    forward: ExecutionWorkspace,
    gradients: Vec<Vec<f32>>,
    scratch: Vec<f32>,
    saved: Vec<TensorHandle>,
    input_refs: Vec<usize>,
    identity: Arc<()>,
    ready: bool,
    gradients_ready: bool,
    needs_grad: Vec<bool>,
}

impl Graph {
    pub fn compile_backward(
        &self,
        shapes: &[Vec<usize>],
        input_grad: &[bool],
    ) -> Result<BackwardPlan, String> {
        let specs = shapes
            .iter()
            .map(|s| TensorSpec::new(s.clone(), Device::Cpu, DType::Float32))
            .collect::<Result<Vec<_>, _>>()?;
        self.compile_backward_specs(&specs, input_grad)
    }
    pub fn compile_backward_specs(
        &self,
        specs: &[TensorSpec],
        input_grad: &[bool],
    ) -> Result<BackwardPlan, String> {
        if input_grad.len() != self.input_count {
            return Err("backward input gradient count mismatch".into());
        }
        let forward = self.compile_specs(specs)?;
        let mut needs_grad = input_grad.to_vec();
        for instruction in forward.instructions.iter().skip(self.input_count) {
            needs_grad.push(instruction.parents().iter().any(|&i| needs_grad[i]));
        }
        let last = forward.instructions.len() - 1;
        let mut reachable = vec![false; forward.instructions.len()];
        reachable[last] = true;
        let mut order = Vec::new();
        for i in (0..forward.instructions.len()).rev() {
            if reachable[i] && needs_grad[i] {
                order.push(i);
                for p in forward.instructions[i].parents() {
                    reachable[p] = true;
                }
            }
        }
        for (needed, reachable) in needs_grad.iter_mut().zip(reachable) {
            *needed &= reachable;
        }
        Ok(BackwardPlan {
            forward,
            needs_grad,
            order,
            identity: Arc::new(()),
        })
    }
}

impl BackwardPlan {
    pub fn create_workspace(&self) -> Result<TrainingWorkspace, String> {
        self.allocate_workspace(false)
    }
    pub fn create_planned_workspace(&self) -> Result<TrainingWorkspace, String> {
        self.allocate_workspace(true)
    }
    fn allocate_workspace(&self, planned: bool) -> Result<TrainingWorkspace, String> {
        let mut gradients = Vec::with_capacity(self.needs_grad.len());
        let mut maximum = 0;
        for (spec, &needed) in self.forward.specs.iter().zip(&self.needs_grad) {
            let count = crate::tensor::TensorLayout::count(spec.shape())?;
            let count = if needed { count } else { 0 };
            maximum = maximum.max(count);
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(count)
                .map_err(|_| "gradient allocation failed")?;
            buffer.resize(count, 0.);
            gradients.push(buffer);
        }
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(maximum)
            .map_err(|_| "gradient scratch allocation failed")?;
        let mut pinned = Vec::new();
        for &i in &self.order {
            match &self.forward.instructions[i] {
                Instruction::LayerNorm { input, weight, .. } => {
                    if self.needs_grad[*input] || weight.is_some_and(|w| self.needs_grad[w]) {
                        pinned.push(*input);
                    }
                    if self.needs_grad[*input] {
                        pinned.extend(*weight);
                    }
                }
                Instruction::Binary {
                    left, right, op, ..
                } => {
                    if matches!(op, crate::ops::BinaryOp::Mul | crate::ops::BinaryOp::Div) {
                        if self.needs_grad[*left] {
                            pinned.push(*right);
                        }
                        if self.needs_grad[*right] {
                            pinned.push(*left);
                            if matches!(op, crate::ops::BinaryOp::Div) {
                                pinned.push(*right);
                            }
                        }
                    }
                }
                Instruction::Reduce { .. } => {}
                Instruction::Matmul { left, right, .. } => {
                    if self.needs_grad[*left] {
                        pinned.push(*right);
                    }
                    if self.needs_grad[*right] {
                        pinned.push(*left);
                    }
                }
                Instruction::Linear { input, weight, .. } => {
                    if self.needs_grad[*input] {
                        pinned.push(*weight);
                    }
                    if self.needs_grad[*weight] {
                        pinned.push(*input);
                    }
                }
                Instruction::Relu { input } => pinned.push(*input),
                Instruction::Mse { prediction, target } => {
                    pinned.push(*prediction);
                    pinned.push(*target);
                }
                Instruction::CrossEntropy { prediction, target } => {
                    pinned.push(*prediction);
                    pinned.push(*target);
                }
                Instruction::Input(_) => {}
            }
        }
        let forward = if planned {
            self.forward.allocate_workspace(Some(&pinned))?
        } else {
            self.forward.create_workspace()?
        };
        Ok(TrainingWorkspace {
            forward,
            gradients,
            scratch,
            saved: Vec::with_capacity(self.forward.input_count),
            input_refs: (0..self.forward.input_count).collect(),
            identity: self.identity.clone(),
            ready: false,
            gradients_ready: false,
            needs_grad: self.needs_grad.clone(),
        })
    }
    /// Execute a complete step and publish new parameter handles only after
    /// the existing optimizer transaction commits successfully.
    pub fn train_step(
        &self,
        inputs: &mut [TensorHandle],
        parameter_slots: &[usize],
        optimizer: &mut crate::optim::Optimizer,
        ws: &mut TrainingWorkspace,
    ) -> Result<TensorData, String> {
        let _gate = autograd::engine_gate()?;
        self.train_step_unlocked(inputs, parameter_slots, optimizer, ws)
    }

    pub(crate) fn train_step_unlocked(
        &self,
        inputs: &mut [TensorHandle],
        parameter_slots: &[usize],
        optimizer: &mut crate::optim::Optimizer,
        ws: &mut TrainingWorkspace,
    ) -> Result<TensorData, String> {
        let (output, prepared) =
            self.prepare_train_step_unlocked(inputs, parameter_slots, optimizer, ws)?;
        let updated = optimizer.commit_prepared_unlocked(prepared);
        for (&slot, value) in parameter_slots.iter().zip(updated) {
            inputs[slot] = value;
        }
        Ok(output)
    }
    pub(crate) fn prepare_train_step_unlocked(
        &self,
        inputs: &[TensorHandle],
        parameter_slots: &[usize],
        optimizer: &crate::optim::Optimizer,
        ws: &mut TrainingWorkspace,
    ) -> Result<(TensorData, crate::optim::PreparedUpdate), String> {
        self.validate_workspace(ws)?;
        if inputs.len() != self.forward.input_count {
            return Err("compiled graph input count mismatch".into());
        }
        let mut seen = vec![false; self.forward.input_count];
        for &slot in parameter_slots {
            if slot >= self.forward.input_count
                || seen[slot]
                || !inputs[slot].is_leaf()
                || !inputs[slot].requires_grad()
            {
                return Err("training requires unique trainable input parameter slots".into());
            }
            if inputs
                .iter()
                .enumerate()
                .any(|(i, input)| i != slot && input.logical_id() == inputs[slot].logical_id())
            {
                return Err("training parameter aliases must share one graph input slot".into());
            }
            seen[slot] = true;
        }
        let output = self.forward_unlocked(inputs, ws)?;
        self.backward_unlocked(None, ws)?;
        let parameters: Vec<_> = parameter_slots.iter().map(|&i| inputs[i].clone()).collect();
        let gradients: Vec<_> = parameter_slots
            .iter()
            .map(|&i| ws.input_gradient(i))
            .collect();
        let prepared = optimizer.prepare_with_gradients_unlocked(&parameters, &gradients);
        ws.ready = false;
        ws.gradients_ready = false;
        // Detached fused loss cannot retain a backward reader. Release handles,
        // retaining only buffers for the next serialized step.
        ws.saved.clear();
        Ok((output, prepared?))
    }
    pub(crate) fn create_inference_workspace(&self) -> Result<ExecutionWorkspace, String> {
        self.forward.create_planned_workspace()
    }
    pub(crate) fn inference_unlocked(
        &self,
        inputs: &[TensorHandle],
        ws: &mut ExecutionWorkspace,
    ) -> Result<TensorData, String> {
        // no_grad reads immutable values and produces an independent snapshot.
        let borrowed: Vec<_> = inputs.iter().map(TensorHandle::value).collect();
        self.forward.execute_borrowed(&borrowed, ws)
    }
    fn validate_workspace(&self, ws: &TrainingWorkspace) -> Result<(), String> {
        if !Arc::ptr_eq(&self.identity, &ws.identity) {
            return Err("backward workspace belongs to another plan".into());
        }
        Ok(())
    }
    /// Snapshot current immutable handles, including frozen operands, for version checks.
    pub fn forward(
        &self,
        inputs: &[TensorHandle],
        ws: &mut TrainingWorkspace,
    ) -> Result<TensorData, String> {
        let _gate = autograd::engine_gate()?;
        self.forward_unlocked(inputs, ws)
    }
    pub(crate) fn forward_unlocked(
        &self,
        inputs: &[TensorHandle],
        ws: &mut TrainingWorkspace,
    ) -> Result<TensorData, String> {
        self.validate_workspace(ws)?;
        ws.ready = false;
        ws.gradients_ready = false;
        if inputs.len() != self.forward.input_count {
            return Err("compiled graph input count mismatch".into());
        }
        // Match eager value reads. Saved versions are validated before backward,
        // where stale provenance could otherwise affect gradients or updates.
        let borrowed: Vec<&TensorData> = inputs.iter().map(TensorHandle::value).collect();
        let output = self.forward.execute_borrowed(&borrowed, &mut ws.forward)?;
        ws.saved.clear();
        ws.saved.extend_from_slice(inputs);
        ws.ready = true;
        Ok(output)
    }
    /// Reset and accumulate gradients for the most recent successful forward.
    /// No eager gradients or optimizer state are changed by this operation.
    pub fn backward(
        &self,
        seed: Option<&TensorData>,
        ws: &mut TrainingWorkspace,
    ) -> Result<(), String> {
        let _gate = autograd::engine_gate()?;
        self.backward_unlocked(seed, ws)
    }
    pub(crate) fn backward_unlocked(
        &self,
        seed: Option<&TensorData>,
        ws: &mut TrainingWorkspace,
    ) -> Result<(), String> {
        self.validate_workspace(ws)?;
        ws.gradients_ready = false;
        if !ws.ready {
            return Err("compiled backward requires a successful forward".into());
        }
        for input in &ws.saved {
            input.check_current()?;
        }
        let last = self.forward.instructions.len() - 1;
        if !self.needs_grad[last] {
            return Err("compiled output does not require gradients".into());
        }
        if let Some(seed) = seed {
            if seed.shape() != self.forward.specs[last].shape() {
                return Err("gradient shape must exactly match output".into());
            }
            self.forward.specs[last].check(seed, "gradient")?;
        } else if crate::tensor::TensorLayout::count(self.forward.specs[last].shape())? != 1 {
            return Err("non-single-element output requires an explicit gradient".into());
        }
        // Private compiled VJPs transfer input adjoints into immutable tensors.
        // Retained/retried backward must restore every drained destination first.
        for (i, gradient) in ws
            .gradients
            .iter_mut()
            .take(self.forward.input_count)
            .enumerate()
        {
            if self.needs_grad[i] {
                let count = crate::tensor::TensorLayout::count(self.forward.specs[i].shape())?;
                if gradient.len() != count {
                    gradient
                        .try_reserve_exact(count.saturating_sub(gradient.len()))
                        .map_err(|_| "gradient refill allocation failed")?;
                    gradient.resize(count, 0.);
                }
            }
        }
        for gradient in &mut ws.gradients {
            gradient.fill(0.);
        }
        if let Some(seed) = seed {
            ws.gradients[last].copy_from_slice(seed.values());
        } else {
            ws.gradients[last][0] = 1.;
        }
        let count = self.forward.input_count;
        let value = |i: usize| -> &[f32] {
            if i < count {
                ws.saved[i].value().values()
            } else {
                ws.forward.value(i)
            }
        };
        for &i in &self.order {
            let (previous, current) = ws.gradients.split_at_mut(i);
            let g = &current[0];
            match &self.forward.instructions[i] {
                Instruction::Input(_) => {}
                Instruction::LayerNorm {
                    input,
                    weight,
                    bias,
                    m,
                    n,
                    eps,
                } => {
                    for (parent, kind) in [
                        (Some(*input), crate::normalization::Gradient::Input),
                        (*weight, crate::normalization::Gradient::Weight),
                        (*bias, crate::normalization::Gradient::Bias),
                    ] {
                        if let Some(parent) = parent.filter(|&p| self.needs_grad[p]) {
                            let x = if matches!(kind, crate::normalization::Gradient::Bias) {
                                None
                            } else {
                                Some(value(*input))
                            };
                            let w = if matches!(kind, crate::normalization::Gradient::Input) {
                                weight.map(value)
                            } else {
                                None
                            };
                            crate::normalization::gradient_into(
                                x,
                                w,
                                g,
                                *m,
                                *n,
                                *eps,
                                kind,
                                &mut ws.scratch,
                            )?;
                            for (dst, &part) in previous[parent].iter_mut().zip(&ws.scratch) {
                                *dst += part;
                            }
                        }
                    }
                }

                Instruction::Reduce { input, mean } => {
                    if self.needs_grad[*input] {
                        let factor = if *mean {
                            g[0] / previous[*input].len() as f32
                        } else {
                            g[0]
                        };
                        for dst in &mut previous[*input] {
                            *dst += factor;
                        }
                    }
                }
                Instruction::Matmul {
                    left,
                    right,
                    m,
                    k,
                    n,
                } => {
                    if self.needs_grad[*left] {
                        cpu_kernels::gemm_accumulate(
                            g,
                            value(*right),
                            *m,
                            *n,
                            *k,
                            false,
                            true,
                            &mut previous[*left],
                        )?;
                    }
                    if self.needs_grad[*right] {
                        cpu_kernels::gemm_accumulate(
                            value(*left),
                            g,
                            *k,
                            *m,
                            *n,
                            true,
                            false,
                            &mut previous[*right],
                        )?;
                    }
                }
                Instruction::Binary {
                    left,
                    right,
                    op,
                    broadcast,
                } => {
                    for left_side in [true, false] {
                        let parent = if left_side { *left } else { *right };
                        if !self.needs_grad[parent] {
                            continue;
                        }
                        match op {
                            crate::ops::BinaryOp::Sub => {
                                broadcast.sub_gradient(left_side, g, &mut previous[parent])
                            }
                            crate::ops::BinaryOp::Div => broadcast.div_gradient(
                                left_side,
                                g,
                                &mut previous[parent],
                                value(*left),
                                value(*right),
                            ),
                            crate::ops::BinaryOp::Add => {
                                broadcast.add_gradient(left_side, g, &mut previous[parent])
                            }
                            crate::ops::BinaryOp::Mul => broadcast.mul_gradient(
                                left_side,
                                g,
                                &mut previous[parent],
                                value(if left_side { *right } else { *left }),
                            ),
                        }
                    }
                }
                Instruction::Relu { input } => {
                    if self.needs_grad[*input] {
                        for ((dst, &x), &grad) in
                            previous[*input].iter_mut().zip(value(*input)).zip(g)
                        {
                            *dst += if x <= 0. { 0. } else { grad };
                        }
                    }
                }
                Instruction::CrossEntropy { prediction, target } => {
                    if self.needs_grad[*prediction] {
                        let classes = self.forward.specs[*prediction].shape()[1];
                        let n = self.forward.specs[*prediction].shape()[0];
                        let labels = value(*target);
                        for ((row, &label), dst) in value(*prediction)
                            .chunks(classes)
                            .zip(labels.iter())
                            .zip(previous[*prediction].chunks_mut(classes))
                        {
                            let label = crate::dispatch::decode_class_index(label, classes)?;
                            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
                            for (j, (&x, grad)) in row.iter().zip(dst.iter_mut()).enumerate() {
                                *grad += ((x - max).exp() / sum
                                    - if j == label { 1.0 } else { 0.0 })
                                    * (g[0] / n as f32);
                            }
                        }
                    }
                }
                Instruction::Mse { prediction, target } => {
                    let a = value(*prediction);
                    let b = value(*target);
                    let scale = 2. * g[0] / a.len() as f32;
                    for (parent, sign) in [(*prediction, 1.), (*target, -1.)] {
                        if self.needs_grad[parent] {
                            for ((dst, &x), &y) in previous[parent].iter_mut().zip(a).zip(b) {
                                *dst += (x - y) * (sign * scale);
                            }
                        }
                    }
                }
                Instruction::Linear {
                    input,
                    weight,
                    bias,
                    m,
                    k,
                    n,
                } => {
                    if self.needs_grad[*input] {
                        cpu_kernels::gemm_accumulate(
                            g,
                            value(*weight),
                            *m,
                            *n,
                            *k,
                            false,
                            false,
                            &mut previous[*input],
                        )?;
                    }
                    if self.needs_grad[*weight] {
                        cpu_kernels::gemm_accumulate(
                            g,
                            value(*input),
                            *n,
                            *m,
                            *k,
                            true,
                            false,
                            &mut previous[*weight],
                        )?;
                    }
                    if let Some(bias) = bias.filter(|&b| self.needs_grad[b]) {
                        for row in g.chunks((*n).max(1)) {
                            for (dst, &part) in previous[bias].iter_mut().zip(row) {
                                *dst += part;
                            }
                        }
                    }
                }
            }
        }
        ws.gradients_ready = true;
        Ok(())
    }
}
impl TrainingWorkspace {
    /// Allocated float32 payload capacity, excluding handles and metadata.
    pub fn allocated_bytes(&self) -> usize {
        self.forward.allocated_bytes()
            + self
                .gradients
                .iter()
                .map(|g| g.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>()
            + self.scratch.capacity() * std::mem::size_of::<f32>()
    }
    pub fn input_gradient(&self, index: usize) -> Option<&[f32]> {
        if !self.gradients_ready || index >= self.input_refs.len() || !self.needs_grad[index] {
            return None;
        }
        Some(&self.gradients[index])
    }
    /// Move a private compiled-VJP input buffer once after successful backward.
    /// Public executor and fused optimizer consumers keep borrowed gradients.
    pub(crate) fn take_input_gradient(&mut self, index: usize) -> Option<Vec<f32>> {
        if !self.gradients_ready || index >= self.input_refs.len() || !self.needs_grad[index] {
            return None;
        }
        Some(std::mem::take(&mut self.gradients[index]))
    }
    #[cfg(test)]
    pub(super) fn gradient_pointers(&self) -> Vec<*const f32> {
        self.gradients.iter().map(|g| g.as_ptr()).collect()
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::{hint::black_box, time::Instant};
    #[test]
    #[ignore = "manual compiled backward benchmark"]
    fn compiled_backward_benchmark() {
        for (batch, k, hidden, n) in [(1, 64, 128, 32), (32, 64, 128, 32), (8, 17, 23, 7)] {
            let mut graph = Graph::new(6);
            let h = graph.linear(0, 1, Some(2)).unwrap();
            let a = graph.relu(h).unwrap();
            let p = graph.linear(a, 3, Some(4)).unwrap();
            graph.mse(p, 5).unwrap();
            let shapes = [
                vec![batch, k],
                vec![hidden, k],
                vec![hidden],
                vec![n, hidden],
                vec![n],
                vec![batch, n],
            ];
            let flags = [false, true, true, true, true, false];
            let inputs: Vec<_> = shapes
                .iter()
                .zip(flags)
                .enumerate()
                .map(|(seed, (s, grad))| {
                    TensorHandle::new(
                        TensorData::new(
                            (0..s.iter().product())
                                .map(|i| ((i + seed) % 19) as f32 / 100.)
                                .collect(),
                            s.clone(),
                        )
                        .unwrap(),
                        grad,
                    )
                })
                .collect();
            let compile_start = Instant::now();
            let plan = graph.compile_backward(&shapes, &flags).unwrap();
            let mut ws = plan.create_workspace().unwrap();
            let compile_us = compile_start.elapsed().as_secs_f64() * 1e6;
            let eager = inputs[0]
                .linear(&inputs[1], Some(&inputs[2]))
                .unwrap()
                .unary(crate::ops::UnaryOp::Relu)
                .unwrap()
                .linear(&inputs[3], Some(&inputs[4]))
                .unwrap()
                .mse_loss(&inputs[5])
                .unwrap();
            eager.backward(None, false).unwrap();
            plan.forward(&inputs, &mut ws).unwrap();
            plan.backward(None, &mut ws).unwrap();
            for (i, input) in inputs.iter().enumerate().take(5).skip(1) {
                let expected = input.grad().unwrap().unwrap();
                for (&got, &want) in ws.input_gradient(i).unwrap().iter().zip(expected.values()) {
                    assert!((got - want).abs() <= 1e-6 + 1e-4 * want.abs());
                }
            }
            for _ in 0..200 {
                plan.forward(&inputs, &mut ws).unwrap();
                plan.backward(None, &mut ws).unwrap();
            }
            let mut backward = Vec::new();
            let mut combined = Vec::new();
            for repetition in 0..5 {
                // Rotate order to reduce a fixed hot-cache ordering advantage.
                for combined_first in [repetition % 2 == 0, repetition % 2 != 0] {
                    let start = Instant::now();
                    for _ in 0..2000 {
                        if combined_first {
                            black_box(plan.forward(black_box(&inputs), &mut ws).unwrap());
                        }
                        plan.backward(None, black_box(&mut ws)).unwrap();
                        black_box(ws.input_gradient(1));
                    }
                    let us = start.elapsed().as_secs_f64() * 1e6 / 2000.;
                    if combined_first {
                        combined.push(us);
                    } else {
                        backward.push(us);
                    }
                }
            }
            backward.sort_by(f64::total_cmp);
            combined.sort_by(f64::total_cmp);
            println!(
                "compiled_backward batch={batch} shape={k}/{hidden}/{n} compile_workspace_us={compile_us:.3} backward_median_us={:.3} forward_backward_median_us={:.3} backward_blocks={backward:?} combined_blocks={combined:?}",
                backward[2], combined[2]
            );
        }
    }
    #[test]
    #[ignore = "manual complete native training benchmark"]
    fn compiled_training_benchmark() {
        use crate::optim::{Config, Kind, Optimizer};
        for (batch, k, hidden, n) in [(1, 64, 128, 32), (32, 64, 128, 32), (8, 17, 23, 7)] {
            for kind in [Kind::Sgd, Kind::Adam] {
                let mut graph = Graph::new(6);
                let h = graph.linear(0, 1, Some(2)).unwrap();
                let a = graph.relu(h).unwrap();
                let p = graph.linear(a, 3, Some(4)).unwrap();
                graph.mse(p, 5).unwrap();
                let shapes = [
                    vec![batch, k],
                    vec![hidden, k],
                    vec![hidden],
                    vec![n, hidden],
                    vec![n],
                    vec![batch, n],
                ];
                let flags = [false, true, true, true, true, false];
                let fixture = || {
                    shapes
                        .iter()
                        .zip(flags)
                        .enumerate()
                        .map(|(seed, (shape, grad))| {
                            TensorHandle::new(
                                TensorData::new(
                                    (0..shape.iter().product())
                                        .map(|i| ((i + seed) % 19) as f32 / 100.)
                                        .collect(),
                                    shape.clone(),
                                )
                                .unwrap(),
                                grad,
                            )
                        })
                        .collect::<Vec<_>>()
                };
                let mut compiled = fixture();
                let mut eager = fixture();
                let config = Config {
                    kind,
                    lr: 0.0001,
                    momentum: 0.,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.,
                };
                let mut compiled_opt = Optimizer::new(config.clone(), &compiled[1..5]).unwrap();
                let mut eager_opt = Optimizer::new(config, &eager[1..5]).unwrap();
                let plan = graph.compile_backward(&shapes, &flags).unwrap();
                let mut ws = plan.create_workspace().unwrap();
                let eager_step = |inputs: &mut Vec<TensorHandle>, opt: &mut Optimizer| {
                    for p in &inputs[1..5] {
                        p.zero_grad().unwrap();
                    }
                    let loss = inputs[0]
                        .linear(&inputs[1], Some(&inputs[2]))
                        .unwrap()
                        .unary(crate::ops::UnaryOp::Relu)
                        .unwrap()
                        .linear(&inputs[3], Some(&inputs[4]))
                        .unwrap()
                        .mse_loss(&inputs[5])
                        .unwrap();
                    loss.backward(None, false).unwrap();
                    let updated = opt.step(&inputs[1..5]).unwrap();
                    inputs[1..5].clone_from_slice(&updated);
                    black_box(loss);
                };
                for _ in 0..100 {
                    plan.train_step(&mut compiled, &[1, 2, 3, 4], &mut compiled_opt, &mut ws)
                        .unwrap();
                    eager_step(&mut eager, &mut eager_opt);
                }
                for (a, b) in compiled[1..5].iter().zip(&eager[1..5]) {
                    for (&a, &b) in a.value().values().iter().zip(b.value().values()) {
                        assert!((a - b).abs() <= 1e-6 + 1e-4 * b.abs());
                    }
                }
                let mut times = Vec::new();
                let mut eager_times = Vec::new();
                for rep in 0..5 {
                    for compiled_first in [rep % 2 == 0, rep % 2 != 0] {
                        let start = Instant::now();
                        for _ in 0..300 {
                            if compiled_first {
                                black_box(
                                    plan.train_step(
                                        &mut compiled,
                                        &[1, 2, 3, 4],
                                        &mut compiled_opt,
                                        &mut ws,
                                    )
                                    .unwrap(),
                                );
                            } else {
                                eager_step(&mut eager, &mut eager_opt);
                            }
                        }
                        let us = start.elapsed().as_secs_f64() * 1e6 / 300.;
                        if compiled_first {
                            times.push(us);
                        } else {
                            eager_times.push(us);
                        }
                    }
                }
                for (a, b) in compiled[1..5].iter().zip(&eager[1..5]) {
                    for (&a, &b) in a.value().values().iter().zip(b.value().values()) {
                        assert!((a - b).abs() <= 1e-6 + 1e-4 * b.abs());
                    }
                }
                times.sort_by(f64::total_cmp);
                eager_times.sort_by(f64::total_cmp);
                let kind = if kind == Kind::Sgd { "sgd" } else { "adam" };
                println!(
                    "native_training optimizer={kind} batch={batch} shape={k}/{hidden}/{n} compiled_median_us={:.3} eager_median_us={:.3} compiled_blocks={times:?} eager_blocks={eager_times:?}",
                    times[2], eager_times[2]
                );
            }
        }
    }
    #[test]
    #[ignore = "manual workspace memory and latency benchmark"]
    fn planned_memory_benchmark() {
        let mut graph = Graph::new(1);
        let mut value = 0;
        for _ in 0..8 {
            value = graph.relu(value).unwrap();
        }
        let plan = graph.compile(&[vec![32, 128]]).unwrap();
        let input = TensorData::new(vec![0.5; 4096], vec![32, 128]).unwrap();
        let mut dedicated = plan.create_workspace().unwrap();
        let mut pooled = plan.create_planned_workspace().unwrap();
        let inputs = [input];
        assert_eq!(
            plan.execute_with_workspace(&inputs, &mut dedicated)
                .unwrap()
                .values(),
            plan.execute_with_workspace(&inputs, &mut pooled)
                .unwrap()
                .values()
        );
        println!(
            "memory inference_relu8 dedicated_bytes={} pooled_bytes={} dedicated_buffers={} pooled_buffers={}",
            dedicated.allocated_bytes(),
            pooled.allocated_bytes(),
            dedicated.buffer_count(),
            pooled.buffer_count()
        );
        for _ in 0..200 {
            plan.execute_with_workspace(&inputs, &mut dedicated)
                .unwrap();
            plan.execute_with_workspace(&inputs, &mut pooled).unwrap();
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        for rep in 0..5 {
            for first in [rep % 2 == 0, rep % 2 != 0] {
                let ws = if first { &mut dedicated } else { &mut pooled };
                let start = Instant::now();
                for _ in 0..2000 {
                    black_box(plan.execute_with_workspace(black_box(&inputs), ws).unwrap());
                }
                let us = start.elapsed().as_secs_f64() * 1e6 / 2000.;
                if first {
                    a.push(us);
                } else {
                    b.push(us);
                }
            }
        }
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        println!(
            "memory inference_relu8 dedicated_median_us={:.3} pooled_median_us={:.3}",
            a[2], b[2]
        );
        let mut graph = Graph::new(6);
        let h = graph.linear(0, 1, Some(2)).unwrap();
        let a = graph.relu(h).unwrap();
        let p = graph.linear(a, 3, Some(4)).unwrap();
        graph.mse(p, 5).unwrap();
        let shapes = [
            vec![32, 64],
            vec![128, 64],
            vec![128],
            vec![32, 128],
            vec![32],
            vec![32, 32],
        ];
        let flags = [false, true, true, true, true, false];
        let inputs: Vec<_> = shapes
            .iter()
            .zip(flags)
            .map(|(s, g)| {
                TensorHandle::new(
                    TensorData::new(vec![0.1; s.iter().product()], s.clone()).unwrap(),
                    g,
                )
            })
            .collect();
        let plan = graph.compile_backward(&shapes, &flags).unwrap();
        let mut dedicated = plan.create_workspace().unwrap();
        let mut pooled = plan.create_planned_workspace().unwrap();
        println!(
            "memory training_mlp dedicated_bytes={} pooled_bytes={}",
            dedicated.allocated_bytes(),
            pooled.allocated_bytes()
        );
        for ws in [&mut dedicated, &mut pooled] {
            plan.forward(&inputs, ws).unwrap();
            plan.backward(None, ws).unwrap();
        }
        for i in 1..5 {
            assert_eq!(dedicated.input_gradient(i), pooled.input_gradient(i));
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        for rep in 0..5 {
            for first in [rep % 2 == 0, rep % 2 != 0] {
                let ws = if first { &mut dedicated } else { &mut pooled };
                let start = Instant::now();
                for _ in 0..2000 {
                    black_box(plan.forward(&inputs, ws).unwrap());
                    plan.backward(None, ws).unwrap();
                    black_box(ws.input_gradient(1));
                }
                let us = start.elapsed().as_secs_f64() * 1e6 / 2000.;
                if first {
                    a.push(us);
                } else {
                    b.push(us);
                }
            }
        }
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        println!(
            "memory training_mlp dedicated_median_us={:.3} pooled_median_us={:.3}",
            a[2], b[2]
        );
    }
    #[test]
    #[ignore = "manual residual and gating graph benchmark"]
    fn binary_graph_benchmark() {
        use crate::ops::BinaryOp;
        for batch in [1, 32] {
            let mut graph = Graph::new(4);
            let gated = graph.binary(0, 1, BinaryOp::Mul).unwrap();
            let shifted = graph.binary(gated, 2, BinaryOp::Add).unwrap();
            let activated = graph.relu(shifted).unwrap();
            graph.mse(activated, 3).unwrap();
            let shapes = [vec![batch, 128], vec![128], vec![], vec![batch, 128]];
            let flags = [true, true, true, false];
            let fixture = || {
                shapes
                    .iter()
                    .zip(flags)
                    .enumerate()
                    .map(|(seed, (s, g))| {
                        TensorHandle::new(
                            TensorData::new(
                                (0..s.iter().product())
                                    .map(|i| ((i + seed) % 19) as f32 / 20. - 0.3)
                                    .collect(),
                                s.clone(),
                            )
                            .unwrap(),
                            g,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let compiled = fixture();
            let eager = fixture();
            let plan = graph.compile_backward(&shapes, &flags).unwrap();
            let mut ws = plan.create_planned_workspace().unwrap();
            let eager_step = || {
                for p in &eager[..3] {
                    p.zero_grad().unwrap();
                }
                let loss = eager[0]
                    .binary(&eager[1], BinaryOp::Mul)
                    .unwrap()
                    .binary(&eager[2], BinaryOp::Add)
                    .unwrap()
                    .unary(crate::ops::UnaryOp::Relu)
                    .unwrap()
                    .mse_loss(&eager[3])
                    .unwrap();
                loss.backward(None, false).unwrap();
                black_box(loss);
            };
            plan.forward(&compiled, &mut ws).unwrap();
            plan.backward(None, &mut ws).unwrap();
            eager_step();
            for (i, input) in eager.iter().enumerate().take(3) {
                let want = input.grad().unwrap().unwrap();
                for (&a, &b) in ws.input_gradient(i).unwrap().iter().zip(want.values()) {
                    assert!((a - b).abs() < 1e-6);
                }
            }
            for _ in 0..200 {
                plan.forward(&compiled, &mut ws).unwrap();
                plan.backward(None, &mut ws).unwrap();
                eager_step();
            }
            let mut a = Vec::new();
            let mut b = Vec::new();
            for rep in 0..5 {
                for first in [rep % 2 == 0, rep % 2 != 0] {
                    let start = Instant::now();
                    for _ in 0..1000 {
                        if first {
                            black_box(plan.forward(&compiled, &mut ws).unwrap());
                            plan.backward(None, &mut ws).unwrap();
                            black_box(ws.input_gradient(1));
                        } else {
                            eager_step();
                        }
                    }
                    let us = start.elapsed().as_secs_f64() * 1e6 / 1000.;
                    if first {
                        a.push(us);
                    } else {
                        b.push(us);
                    }
                }
            }
            a.sort_by(f64::total_cmp);
            b.sort_by(f64::total_cmp);
            println!(
                "binary_graph batch={batch} compiled_median_us={:.3} eager_median_us={:.3} workspace_payload_bytes={} cached_broadcast_map_bytes={} descriptor_bytes={}",
                a[2],
                b[2],
                ws.allocated_bytes(),
                plan.forward.broadcast_map_bytes(),
                plan.forward.broadcast_descriptor_bytes()
            );
        }
    }
}

#[cfg(test)]
mod training_tests {
    use super::*;
    use crate::{
        ops::UnaryOp,
        optim::{Config, Kind, Optimizer},
    };
    fn h(values: Vec<f32>, shape: Vec<usize>, grad: bool) -> TensorHandle {
        TensorHandle::new(TensorData::new(values, shape).unwrap(), grad)
    }
    fn config(kind: Kind, momentum: f64) -> Config {
        Config {
            kind,
            lr: 0.01,
            momentum,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
    #[test]
    fn complete_training_matches_eager_sgd_momentum_and_adam() {
        for (kind, momentum) in [(Kind::Sgd, 0.), (Kind::Sgd, 0.8), (Kind::Adam, 0.)] {
            let mut graph = Graph::new(4);
            let linear = graph.linear(0, 1, Some(2)).unwrap();
            let relu = graph.relu(linear).unwrap();
            graph.mse(relu, 3).unwrap();
            let shapes = [vec![2, 2], vec![1, 2], vec![1], vec![2, 1]];
            let plan = graph
                .compile_backward(&shapes, &[false, true, true, false])
                .unwrap();
            let mut ws = plan.create_planned_workspace().unwrap();
            let fixture = || {
                vec![
                    h(vec![1., 2., -2., 1.], shapes[0].clone(), false),
                    h(vec![0.3, 0.8], shapes[1].clone(), true),
                    h(vec![0.4], shapes[2].clone(), true),
                    h(vec![2., 0.5], shapes[3].clone(), false),
                ]
            };
            let mut compiled = fixture();
            let mut eager = fixture();
            let mut compiled_opt = Optimizer::new(config(kind, momentum), &compiled[1..3]).unwrap();
            let mut eager_opt = Optimizer::new(config(kind, momentum), &eager[1..3]).unwrap();
            for iteration in 0..12 {
                if iteration == 6 {
                    compiled[0] = h(vec![2., 1., -1., 3.], shapes[0].clone(), false);
                    eager[0] = h(vec![2., 1., -1., 3.], shapes[0].clone(), false);
                }
                for p in &eager[1..3] {
                    p.zero_grad().unwrap();
                }
                let expected = eager[0]
                    .linear(&eager[1], Some(&eager[2]))
                    .unwrap()
                    .unary(UnaryOp::Relu)
                    .unwrap()
                    .mse_loss(&eager[3])
                    .unwrap();
                expected.backward(None, false).unwrap();
                let updated = eager_opt.step(&eager[1..3]).unwrap();
                eager[1..3].clone_from_slice(&updated);
                let loss = plan
                    .train_step(&mut compiled, &[1, 2], &mut compiled_opt, &mut ws)
                    .unwrap();
                assert!((loss.values()[0] - expected.value().values()[0]).abs() < 1e-6);
                for (a, b) in compiled[1..3].iter().zip(&eager[1..3]) {
                    for (&a, &b) in a.value().values().iter().zip(b.value().values()) {
                        assert!((a - b).abs() < 1e-6);
                    }
                }
                for (a, b) in compiled_opt.snapshot().iter().zip(eager_opt.snapshot()) {
                    assert_eq!(a.step, b.step);
                    assert_eq!(a.first, b.first);
                    assert_eq!(a.second, b.second);
                }
            }
        }
    }
    #[test]
    fn failed_training_optimizer_transaction_preserves_all_parameters() {
        let mut graph = Graph::new(4);
        let p = graph.linear(0, 1, Some(2)).unwrap();
        graph.mse(p, 3).unwrap();
        let shapes = [vec![1, 1], vec![1, 1], vec![1], vec![1, 1]];
        let plan = graph
            .compile_backward(&shapes, &[false, true, true, false])
            .unwrap();
        let mut ws = plan.create_workspace().unwrap();
        let mut inputs = vec![
            h(vec![1.], shapes[0].clone(), false),
            h(vec![2.], shapes[1].clone(), true),
            h(vec![3.], shapes[2].clone(), true),
            h(vec![0.], shapes[3].clone(), false),
        ];
        let originals = inputs.clone();
        let mut optimizer = Optimizer::new(config(Kind::Sgd, 0.), &inputs[1..3]).unwrap();
        optimizer
            .load(vec![
                crate::optim::State::default(),
                crate::optim::State {
                    step: u64::MAX,
                    first: vec![],
                    second: vec![],
                },
            ])
            .unwrap();
        assert!(
            plan.train_step(&mut inputs, &[1, 2], &mut optimizer, &mut ws)
                .is_err()
        );
        assert_eq!(optimizer.snapshot()[0].step, 0);
        assert_eq!(inputs[1].value().values(), &[2.]);
        assert_eq!(inputs[2].value().values(), &[3.]);
        for p in originals {
            p.check_current().unwrap();
        }
        assert!(
            plan.train_step(&mut inputs, &[1, 1], &mut optimizer, &mut ws)
                .is_err()
        );
    }
    #[test]
    fn disconnected_training_parameter_skips_decay_and_state() {
        let mut graph = Graph::new(3);
        graph.mse(0, 1).unwrap();
        let plan = graph
            .compile_backward(&[vec![1], vec![1], vec![1]], &[true, false, true])
            .unwrap();
        let mut ws = plan.create_workspace().unwrap();
        let unused = h(vec![3.], vec![1], true);
        let mut inputs = vec![
            h(vec![1.], vec![1], true),
            h(vec![0.], vec![1], false),
            unused.clone(),
        ];
        let mut optimizer =
            Optimizer::new(config(Kind::Adam, 0.), std::slice::from_ref(&unused)).unwrap();
        plan.train_step(&mut inputs, &[2], &mut optimizer, &mut ws)
            .unwrap();
        assert_eq!(inputs[2].value().values(), &[3.]);
        assert_eq!(optimizer.snapshot()[0].step, 0);
        unused.check_current().unwrap();
    }
    #[test]
    fn training_rejects_parameter_aliases_in_distinct_input_slots() {
        let mut graph = Graph::new(3);
        let prediction = graph.linear(0, 1, None).unwrap();
        graph.mse(prediction, 2).unwrap();
        let plan = graph
            .compile_backward(&[vec![1, 1], vec![1, 1], vec![1, 1]], &[true, true, false])
            .unwrap();
        let mut ws = plan.create_workspace().unwrap();
        let p = h(vec![2.], vec![1, 1], true);
        let mut inputs = vec![p.clone(), p.clone(), h(vec![0.], vec![1, 1], false)];
        let mut optimizer =
            Optimizer::new(config(Kind::Sgd, 0.), std::slice::from_ref(&p)).unwrap();
        assert!(
            plan.train_step(&mut inputs, &[1], &mut optimizer, &mut ws)
                .is_err()
        );
        assert_eq!(optimizer.snapshot()[0].step, 0);
        p.check_current().unwrap();
    }
    #[test]
    fn one_slot_used_as_input_and_weight_accumulates_both_derivatives() {
        let mut graph = Graph::new(1);
        graph.linear(0, 0, None).unwrap();
        let plan = graph.compile_backward(&[vec![2, 2]], &[true]).unwrap();
        let mut ws = plan.create_workspace().unwrap();
        let x = h(vec![1., 2., 3., 4.], vec![2, 2], true);
        let seed = TensorData::new(vec![1.; 4], vec![2, 2]).unwrap();
        x.linear(&x, None)
            .unwrap()
            .backward(Some(&seed), false)
            .unwrap();
        plan.forward(std::slice::from_ref(&x), &mut ws).unwrap();
        plan.backward(Some(&seed), &mut ws).unwrap();
        assert_eq!(
            ws.input_gradient(0).unwrap(),
            x.grad().unwrap().unwrap().values()
        );
        assert_eq!(ws.input_gradient(0).unwrap(), &[8., 12., 8., 12.]);
    }
}

#[cfg(test)]
mod payload_accounting_study {
    use super::*;
    use std::collections::HashSet;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Payload {
        forward: usize,
        adjoints: usize,
        scratch: usize,
    }

    impl Payload {
        fn read(workspace: &TrainingWorkspace) -> Self {
            let result = Self {
                forward: workspace.forward.allocated_bytes(),
                adjoints: workspace
                    .gradients
                    .iter()
                    .map(|g| g.capacity() * size_of::<f32>())
                    .sum(),
                scratch: workspace.scratch.capacity() * size_of::<f32>(),
            };
            assert_eq!(result.total(), workspace.allocated_bytes());
            result
        }

        fn total(self) -> usize {
            self.forward + self.adjoints + self.scratch
        }

        fn json(self) -> String {
            format!(
                "{{\"forward\":{},\"adjoints\":{},\"scratch\":{},\"total\":{}}}",
                self.forward,
                self.adjoints,
                self.scratch,
                self.total()
            )
        }
    }

    #[test]
    fn workspace_payload_lifecycle_study() {
        for (k, hidden, n) in [(64, 128, 32), (256, 1024, 256)] {
            for batch in [1, 32] {
                let mut graph = Graph::new(6);
                let h = graph.linear(0, 1, Some(2)).unwrap();
                let a = graph.relu(h).unwrap();
                let p = graph.linear(a, 3, Some(4)).unwrap();
                graph.mse(p, 5).unwrap();
                let shapes = [
                    vec![batch, k],
                    vec![hidden, k],
                    vec![hidden],
                    vec![n, hidden],
                    vec![n],
                    vec![batch, n],
                ];
                let flags = [false, true, true, true, true, false];
                let inputs: Vec<_> = shapes
                    .iter()
                    .enumerate()
                    .map(|(i, shape)| {
                        let value = if i == 0 {
                            0.1
                        } else if i == 5 {
                            0.05
                        } else {
                            0.01
                        };
                        TensorHandle::new(
                            TensorData::new(vec![value; shape.iter().product()], shape.clone())
                                .unwrap(),
                            flags[i],
                        )
                    })
                    .collect();
                let plan = graph.compile_backward(&shapes, &flags).unwrap();
                let mut dedicated = plan.create_workspace().unwrap();
                let mut primary = plan.create_planned_workspace().unwrap();
                let dedicated_initial = Payload::read(&dedicated);
                let planned_initial = Payload::read(&primary);
                assert_eq!(dedicated_initial, planned_initial);
                let expected = plan.forward(&inputs, &mut dedicated).unwrap();
                let first = plan.forward(&inputs, &mut primary).unwrap();
                assert_eq!(first.values(), expected.values());

                // A second live forward uses distinct activation/adjoint buffers.
                let mut alternate = inputs.clone();
                alternate[0] = TensorHandle::new(
                    TensorData::new(vec![0.4; batch * k], vec![batch, k]).unwrap(),
                    false,
                );
                let mut secondary = plan.create_planned_workspace().unwrap();
                let second = plan.forward(&alternate, &mut secondary).unwrap();
                let first_snapshot = first.values().to_vec();
                assert_ne!(first.values(), second.values());
                assert_eq!(first.values(), first_snapshot);
                let mut storage_ids = HashSet::new();
                let mut borrowed_tensor_bytes = 0usize;
                for input in inputs.iter().chain(&alternate) {
                    if storage_ids.insert(input.value().storage_identity()) {
                        borrowed_tensor_bytes += input.value().storage_capacity_bytes();
                    }
                }
                let mut output_tensor_bytes = 0usize;
                for output in [&first, &second] {
                    assert!(storage_ids.insert(output.storage_identity()));
                    output_tensor_bytes += output.storage_capacity_bytes();
                }
                for (index, (a, b)) in primary
                    .gradient_pointers()
                    .iter()
                    .zip(secondary.gradient_pointers())
                    .enumerate()
                {
                    if primary.gradients[index].capacity() > 0 {
                        assert_ne!(*a, b);
                    }
                }
                plan.backward(None, &mut dedicated).unwrap();
                plan.backward(None, &mut primary).unwrap();
                for i in 1..5 {
                    assert_eq!(primary.input_gradient(i), dedicated.input_gradient(i));
                }
                drop(dedicated);
                let before = Payload::read(&primary);
                let second_before = Payload::read(&secondary);
                let two_live = before.total() + second_before.total();
                let output_logical_bytes = (first.numel() + second.numel()) * size_of::<f32>();
                let snapshots: Vec<_> = (1..5)
                    .map(|i| {
                        TensorData::new(primary.take_input_gradient(i).unwrap(), shapes[i].clone())
                            .unwrap()
                    })
                    .collect();
                let moved = snapshots
                    .iter()
                    .map(TensorData::storage_capacity_bytes)
                    .sum::<usize>();
                for snapshot in &snapshots {
                    assert!(storage_ids.insert(snapshot.storage_identity()));
                }
                let drained = Payload::read(&primary);
                assert_eq!(before.total() - drained.total(), moved);
                assert_eq!(drained.total() + second_before.total() + moved, two_live);

                // Invalid seed shape fails before refilling or modifying buffers.
                let invalid = TensorData::new(vec![1.], vec![1]).unwrap();
                assert!(plan.backward(Some(&invalid), &mut primary).is_err());
                assert_eq!(Payload::read(&primary).total(), drained.total());
                plan.backward(None, &mut primary).unwrap();
                let retried = Payload::read(&primary);
                for (i, snapshot) in (1..5).zip(&snapshots) {
                    assert_eq!(primary.input_gradient(i).unwrap(), snapshot.values());
                }
                plan.backward(None, &mut secondary).unwrap();
                assert_ne!(primary.input_gradient(1), secondary.input_gradient(1));
                for (i, snapshot) in (1..5).zip(&snapshots) {
                    assert_eq!(primary.input_gradient(i).unwrap(), snapshot.values());
                }
                let combined_retry = retried.total() + Payload::read(&secondary).total() + moved;
                let combined_retry_plus_tensors =
                    combined_retry + borrowed_tensor_bytes + output_tensor_bytes;
                println!(
                    "workspace_payload {{\"batch\":{batch},\"features\":[{k},{hidden},{n}],\"dedicated_initial\":{},\"planned_initial\":{},\"after_backward\":{},\"two_live_workspaces\":{two_live},\"returned_outputs_workspace_excluded_logical_bytes\":{output_logical_bytes},\"after_transfer\":{},\"moved_gradient_capacity\":{moved},\"after_retry\":{},\"borrowed_input_parameter_tensor_capacity\":{borrowed_tensor_bytes},\"returned_output_tensor_capacity\":{output_tensor_bytes},\"transferred_vjp_gradient_tensor_capacity\":{moved},\"tracked_payload_after_retry_excluding_control_temporaries\":{combined_retry_plus_tensors}}}",
                    dedicated_initial.json(),
                    planned_initial.json(),
                    before.json(),
                    drained.json(),
                    retried.json()
                );
            }
        }
    }
}
