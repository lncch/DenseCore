use super::{ExecutionPlan, Instruction};
use crate::{cpu_kernels, tensor::TensorData};
use std::sync::Arc;

pub struct ExecutionWorkspace {
    buffers: Vec<Vec<f32>>,
    slots: Vec<Option<usize>>,
    lengths: Vec<usize>,
    capacities: Vec<usize>,
    identity: Arc<()>,
}
impl ExecutionWorkspace {
    pub fn allocated_bytes(&self) -> usize {
        self.buffers
            .iter()
            .map(|b| b.capacity() * std::mem::size_of::<f32>())
            .sum()
    }
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }
    pub(super) fn value(&self, node: usize) -> &[f32] {
        &self.buffers[self.slots[node].expect("non-input workspace node")][..self.lengths[node]]
    }
}
impl ExecutionPlan {
    pub fn create_workspace(&self) -> Result<ExecutionWorkspace, String> {
        self.allocate_workspace(None)
    }
    pub fn create_planned_workspace(&self) -> Result<ExecutionWorkspace, String> {
        self.allocate_workspace(Some(&[]))
    }
    /// Pinned values remain live through all subsequent backward calls.
    pub(super) fn allocate_workspace(
        &self,
        pinned: Option<&[usize]>,
    ) -> Result<ExecutionWorkspace, String> {
        let count = self.instructions.len();
        let lengths = self
            .specs
            .iter()
            .map(|s| crate::tensor::TensorLayout::count(s.shape()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ends: Vec<usize> = (0..count).collect();
        for (i, op) in self.instructions.iter().enumerate() {
            let parents = op.parents();
            for p in parents {
                ends[p] = ends[p].max(i);
            }
        }
        ends[count - 1] = count;
        if let Some(pinned) = pinned {
            for &i in pinned {
                if i >= count {
                    return Err("invalid pinned activation".into());
                }
                ends[i] = count;
            }
        }
        let mut slots = vec![None; count];
        let mut capacities = Vec::<usize>::new();
        let mut occupied_until = Vec::<usize>::new();
        for i in self.input_count..count {
            let reusable = if pinned.is_some() {
                capacities
                    .iter()
                    .enumerate()
                    .filter(|&(slot, &capacity)| occupied_until[slot] < i && capacity >= lengths[i])
                    .min_by_key(|&(_, &capacity)| capacity)
                    .map(|(slot, _)| slot)
            } else {
                None
            };
            let slot = if let Some(slot) = reusable {
                slot
            } else {
                capacities.push(lengths[i]);
                occupied_until.push(0);
                capacities.len() - 1
            };
            slots[i] = Some(slot);
            occupied_until[slot] = ends[i];
        }
        let mut buffers = Vec::with_capacity(capacities.len());
        for &capacity in &capacities {
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(capacity)
                .map_err(|_| "workspace allocation failed")?;
            buffer.resize(capacity, 0.);
            buffers.push(buffer);
        }
        Ok(ExecutionWorkspace {
            buffers,
            slots,
            lengths,
            capacities,
            identity: self.identity.clone(),
        })
    }
    pub fn execute_with_workspace(
        &self,
        inputs: &[TensorData],
        workspace: &mut ExecutionWorkspace,
    ) -> Result<TensorData, String> {
        let refs: Vec<&TensorData> = inputs.iter().collect();
        self.execute_borrowed(&refs, workspace)
    }
    pub(super) fn execute_borrowed(
        &self,
        inputs: &[&TensorData],
        workspace: &mut ExecutionWorkspace,
    ) -> Result<TensorData, String> {
        if !Arc::ptr_eq(&self.identity, &workspace.identity) {
            return Err("workspace does not match execution plan".into());
        }
        self.validate_inputs(inputs)?;
        for (b, &capacity) in workspace.buffers.iter().zip(&workspace.capacities) {
            if b.capacity() < capacity {
                return Err("workspace buffer capacity mismatch".into());
            }
        }
        for (position, instruction) in self.instructions.iter().enumerate().skip(self.input_count) {
            let slot = workspace.slots[position].ok_or("invalid execution plan workspace")?;
            let (before, current_after) = workspace.buffers.split_at_mut(slot);
            let (output, after) = current_after
                .split_first_mut()
                .ok_or("invalid workspace slot")?;
            output.resize(workspace.lengths[position], 0.);
            let get = |index: usize| -> &[f32] {
                if index < self.input_count {
                    inputs[index].values()
                } else {
                    let source = workspace.slots[index].expect("source workspace slot");
                    assert_ne!(source, slot, "live workspace source/output alias");
                    let buffer = if source < slot {
                        &before[source]
                    } else {
                        &after[source - slot - 1]
                    };
                    &buffer[..workspace.lengths[index]]
                }
            };
            match instruction {
                Instruction::Input(_) => {}
                Instruction::LayerNorm {
                    input,
                    weight,
                    bias,
                    m,
                    n,
                    eps,
                } => crate::normalization::forward_into(
                    get(*input),
                    weight.map(get),
                    bias.map(get),
                    *m,
                    *n,
                    *eps,
                    output,
                )?,
                Instruction::Matmul {
                    left,
                    right,
                    m,
                    k,
                    n,
                } => cpu_kernels::gemm_into(
                    get(*left),
                    get(*right),
                    *m,
                    *k,
                    *n,
                    false,
                    false,
                    None,
                    output,
                )?,
                Instruction::Reduce { input, mean } => {
                    let source = get(*input);
                    output[0] = crate::dispatch::reduce_cpu_values(source, *mean);
                }
                Instruction::Binary {
                    left,
                    right,
                    op,
                    broadcast,
                } => {
                    let a = get(*left);
                    let b = get(*right);
                    broadcast.forward(a, b, output, *op);
                }
                Instruction::Linear {
                    input,
                    weight,
                    bias,
                    m,
                    k,
                    n,
                } => cpu_kernels::gemm_into(
                    get(*input),
                    get(*weight),
                    *m,
                    *k,
                    *n,
                    false,
                    true,
                    bias.map(get),
                    output,
                )?,
                Instruction::Relu { input } => {
                    for (dst, &v) in output.iter_mut().zip(get(*input)) {
                        *dst = if v < 0. { 0. } else { v };
                    }
                }
                Instruction::CrossEntropy { prediction, target } => {
                    output[0] = crate::loss::cross_entropy_tensor_values(
                        get(*prediction),
                        get(*target),
                        self.specs[*prediction].shape(),
                        self.specs[*target].shape(),
                    )?;
                }
                Instruction::Mse { prediction, target } => {
                    output[0] = crate::loss::mse_values(get(*prediction), get(*target))?;
                }
            }
        }
        let last = self.instructions.len() - 1;
        let values = if last < self.input_count {
            inputs[last].values().to_vec()
        } else {
            workspace.value(last).to_vec()
        };
        // compile_specs only lowers CPU float32, matching this owned output.
        TensorData::new(values, self.specs[last].shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    fn tensor(data: Vec<f32>, shape: Vec<usize>) -> TensorData {
        TensorData::new(data, shape).unwrap()
    }

    #[test]
    fn planned_workspace_recycles_only_dead_buffers_and_owns_output() {
        let mut graph = Graph::new(1);
        let mut value = 0;
        for _ in 0..8 {
            value = graph.relu(value).unwrap();
        }
        let plan = graph.compile(&[vec![4]]).unwrap();
        let mut dedicated = plan.create_workspace().unwrap();
        let mut planned = plan.create_planned_workspace().unwrap();
        assert!(planned.allocated_bytes() < dedicated.allocated_bytes());
        assert_eq!(planned.buffer_count(), 2);
        let first = plan
            .execute_with_workspace(&[tensor(vec![-1., 2., 3., -4.], vec![4])], &mut planned)
            .unwrap();
        for data in [vec![4., -3., 2., -1.], vec![-1.; 4]] {
            let inputs = [tensor(data, vec![4])];
            assert_eq!(
                plan.execute_with_workspace(&inputs, &mut planned)
                    .unwrap()
                    .values(),
                plan.execute_with_workspace(&inputs, &mut dedicated)
                    .unwrap()
                    .values()
            );
        }
        assert_eq!(first.values(), &[0., 2., 3., 0.]);
        let other = graph.compile(&[vec![4]]).unwrap();
        assert!(
            other
                .execute_with_workspace(&[tensor(vec![1.; 4], vec![4])], &mut planned)
                .is_err()
        );
    }
    #[test]
    fn workspace_lifetime_payload_study() {
        fn measure(name: &str, graph: &Graph, expected_dedicated: usize, expected_planned: usize) {
            let shape = vec![32, 128];
            let plan = graph.compile(std::slice::from_ref(&shape)).unwrap();
            let mut dedicated = plan.create_workspace().unwrap();
            let mut planned = plan.create_planned_workspace().unwrap();
            assert_eq!(dedicated.buffer_count(), expected_dedicated);
            assert_eq!(planned.buffer_count(), expected_planned);

            let values: Vec<f32> = (0..4096)
                .map(|i| if i % 2 == 0 { 0.25 } else { -0.5 })
                .collect();
            let first_input = tensor(values, shape.clone());
            let first_planned = plan
                .execute_with_workspace(std::slice::from_ref(&first_input), &mut planned)
                .unwrap();
            let first_dedicated = plan
                .execute_with_workspace(std::slice::from_ref(&first_input), &mut dedicated)
                .unwrap();
            assert_eq!(first_planned.values(), first_dedicated.values());

            let second_input = tensor(vec![-0.75; 4096], shape);
            assert_eq!(
                plan.execute_with_workspace(std::slice::from_ref(&second_input), &mut planned)
                    .unwrap()
                    .values(),
                plan.execute_with_workspace(std::slice::from_ref(&second_input), &mut dedicated)
                    .unwrap()
                    .values()
            );
            assert_eq!(first_planned.values(), first_dedicated.values());

            println!(
                "workspace_lifetime_payload {{\"graph\":\"{name}\",\"dedicated_buffers\":{},\"planned_buffers\":{},\"dedicated_capacity_bytes\":{},\"planned_capacity_bytes\":{}}}",
                dedicated.buffer_count(),
                planned.buffer_count(),
                dedicated.allocated_bytes(),
                planned.allocated_bytes(),
            );
            assert!(planned.allocated_bytes() < dedicated.allocated_bytes());
        }

        let mut chain = Graph::new(1);
        let mut value = 0;
        for _ in 0..8 {
            value = chain.relu(value).unwrap();
        }
        measure("relu_chain_8", &chain, 8, 2);

        let mut diamond = Graph::new(1);
        let first = diamond.relu(0).unwrap();
        let second = diamond.relu(first).unwrap();
        let third = diamond.relu(second).unwrap();
        let branch = diamond.relu(0).unwrap();
        let mut joined = diamond
            .binary(third, branch, crate::ops::BinaryOp::Add)
            .unwrap();
        for _ in 0..4 {
            joined = diamond.relu(joined).unwrap();
        }
        measure("relu_diamond_tail_4", &diamond, 9, 3);
    }
    #[test]
    fn pooled_slots_use_exact_lengths_when_large_storage_hosts_smaller_values() {
        let mut graph = Graph::new(2);
        let a = graph.relu(0).unwrap();
        graph.relu(a).unwrap();
        graph.linear(0, 1, None).unwrap();
        let inputs = [
            tensor(vec![1.; 8], vec![2, 4]),
            tensor(vec![2.; 8], vec![2, 4]),
        ];
        let plan = graph.compile(&[vec![2, 4], vec![2, 4]]).unwrap();
        let mut ws = plan.create_planned_workspace().unwrap();
        for _ in 0..3 {
            assert_eq!(
                plan.execute_with_workspace(&inputs, &mut ws)
                    .unwrap()
                    .values(),
                &[8.; 4]
            );
        }
    }
    #[test]
    fn workspace_matches_eager_mlp() {
        let mut graph = Graph::new(6);

        let hidden = graph.linear(0, 1, Some(2)).unwrap();
        let activated = graph.relu(hidden).unwrap();
        let prediction = graph.linear(activated, 3, Some(4)).unwrap();
        graph.mse(prediction, 5).unwrap();

        let inputs = vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            tensor(vec![0.5, -0.5], vec![2]),
            tensor(vec![2.0, 1.0], vec![1, 2]),
            tensor(vec![-1.0], vec![1]),
            tensor(vec![0.0, 1.0], vec![2, 1]),
        ];

        let shapes: Vec<Vec<usize>> = inputs.iter().map(|v| v.shape().to_vec()).collect();

        let plan = graph.compile(&shapes).unwrap();
        let mut workspace = plan.create_workspace().unwrap();

        let eager = graph.execute(&inputs).unwrap();
        let compiled = plan
            .execute_with_workspace(&inputs, &mut workspace)
            .unwrap();

        assert_eq!(eager.shape(), compiled.shape());
        assert_eq!(eager.values(), compiled.values());
    }

    #[test]
    fn workspace_reuses_allocations() {
        let mut graph = Graph::new(2);

        let hidden = graph.relu(0).unwrap();
        graph.mse(hidden, 1).unwrap();

        let plan = graph.compile(&[vec![4], vec![4]]).unwrap();
        let mut workspace = plan.create_workspace().unwrap();

        let pointers_before: Vec<*const f32> = workspace
            .buffers
            .iter()
            .map(|buffer| buffer.as_ptr())
            .collect();

        for values in [
            vec![-1.0, 2.0, -3.0, 4.0],
            vec![5.0, -6.0, 7.0, -8.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ] {
            let inputs = vec![tensor(values, vec![4]), tensor(vec![1.0; 4], vec![4])];

            let eager = graph.execute(&inputs).unwrap();

            let compiled = plan
                .execute_with_workspace(&inputs, &mut workspace)
                .unwrap();

            assert_eq!(eager.values(), compiled.values());
        }

        let pointers_after: Vec<*const f32> = workspace
            .buffers
            .iter()
            .map(|buffer| buffer.as_ptr())
            .collect();

        assert_eq!(pointers_before, pointers_after);
    }

    #[test]
    fn workspace_supports_branching() {
        let mut graph = Graph::new(2);

        let first = graph.relu(0).unwrap();
        let second = graph.relu(0).unwrap();
        graph.mse(first, second).unwrap();

        let plan = graph.compile(&[vec![3], vec![3]]).unwrap();
        let mut workspace = plan.create_workspace().unwrap();

        let inputs = vec![
            tensor(vec![-1.0, 2.0, -3.0], vec![3]),
            tensor(vec![4.0, 5.0, 6.0], vec![3]),
        ];

        let eager = graph.execute(&inputs).unwrap();

        let compiled = plan
            .execute_with_workspace(&inputs, &mut workspace)
            .unwrap();

        assert_eq!(eager.values(), compiled.values());
    }

    #[test]
    fn workspace_rejects_wrong_input_shape() {
        let mut graph = Graph::new(1);
        graph.relu(0).unwrap();

        let plan = graph.compile(&[vec![2, 3]]).unwrap();
        let mut workspace = plan.create_workspace().unwrap();

        let wrong = tensor(vec![1.0; 6], vec![3, 2]);

        assert!(
            plan.execute_with_workspace(&[wrong], &mut workspace)
                .is_err()
        );
    }

    #[test]
    fn workspace_supports_linear_without_bias() {
        let mut graph = Graph::new(2);
        graph.linear(0, 1, None).unwrap();

        let inputs = vec![
            tensor(vec![1.0, 2.0], vec![1, 2]),
            tensor(vec![3.0, 4.0], vec![1, 2]),
        ];

        let plan = graph.compile(&[vec![1, 2], vec![1, 2]]).unwrap();

        let mut workspace = plan.create_workspace().unwrap();

        let eager = graph.execute(&inputs).unwrap();
        let compiled = plan
            .execute_with_workspace(&inputs, &mut workspace)
            .unwrap();

        assert_eq!(eager.values(), compiled.values());
    }
}

#[cfg(test)]
mod backward_tests {
    use super::*;
    use crate::ops::UnaryOp;
    use crate::{autograd::TensorHandle, graph::Graph};
    fn handle(v: Vec<f32>, s: Vec<usize>, grad: bool) -> TensorHandle {
        TensorHandle::new(TensorData::new(v, s).unwrap(), grad)
    }
    #[test]
    fn compiled_backward_accumulates_branches_and_reuses_slots() {
        let mut graph = Graph::new(3);
        let left = graph.linear(0, 1, None).unwrap();
        let right = graph.linear(2, 1, None).unwrap();
        graph.mse(left, right).unwrap();
        let plan = graph
            .compile_backward(&[vec![1, 2], vec![1, 2], vec![1, 2]], &[false, true, false])
            .unwrap();
        let mut ws = plan.create_planned_workspace().unwrap();
        let inputs = vec![
            handle(vec![2., 3.], vec![1, 2], false),
            handle(vec![4., 5.], vec![1, 2], true),
            handle(vec![1., 1.], vec![1, 2], false),
        ];
        plan.forward(&inputs, &mut ws).unwrap();
        plan.backward(None, &mut ws).unwrap();
        assert_eq!(ws.input_gradient(1).unwrap(), &[28., 56.]);
        assert!(ws.input_gradient(0).is_none());
        let pointers = ws.gradient_pointers();
        for _ in 0..3 {
            plan.forward(&inputs, &mut ws).unwrap();
            plan.backward(None, &mut ws).unwrap();
            assert_eq!(ws.input_gradient(1).unwrap(), &[28., 56.]);
        }
        assert_eq!(pointers, ws.gradient_pointers());
    }
    #[test]
    fn compiled_backward_matches_eager_and_rejects_stale_forward() {
        let mut graph = Graph::new(4);
        let linear = graph.linear(0, 1, Some(2)).unwrap();
        let relu = graph.relu(linear).unwrap();
        graph.mse(relu, 3).unwrap();
        let shapes = [vec![2, 2], vec![1, 2], vec![1], vec![2, 1]];
        let plan = graph
            .compile_backward(&shapes, &[true, true, true, true])
            .unwrap();
        let mut ws = plan.create_workspace().unwrap();
        let inputs = vec![
            handle(vec![1., -2., 3., 4.], shapes[0].clone(), true),
            handle(vec![2., 1.], shapes[1].clone(), true),
            handle(vec![1.], shapes[2].clone(), true),
            handle(vec![3., 2.], shapes[3].clone(), true),
        ];
        let eager = inputs[0]
            .linear(&inputs[1], Some(&inputs[2]))
            .unwrap()
            .unary(UnaryOp::Relu)
            .unwrap()
            .mse_loss(&inputs[3])
            .unwrap();
        eager.backward(None, false).unwrap();
        plan.forward(&inputs, &mut ws).unwrap();
        plan.backward(None, &mut ws).unwrap();
        for (i, input) in inputs.iter().enumerate() {
            assert_eq!(
                ws.input_gradient(i).unwrap(),
                input.grad().unwrap().unwrap().values()
            );
        }
        assert!(
            plan.backward(Some(&TensorData::new(vec![1.], vec![1]).unwrap()), &mut ws)
                .is_err()
        );
        crate::autograd::replace_batch(
            &[inputs[1].clone()],
            vec![TensorData::new(vec![1., 1.], vec![1, 2]).unwrap()],
        )
        .unwrap();
        assert!(plan.backward(None, &mut ws).is_err());
    }
    #[test]
    fn planned_training_preserves_all_backward_activation_reads() {
        let mut graph = Graph::new(4);
        let unused = graph.relu(0).unwrap();
        graph.relu(unused).unwrap();
        let p = graph.linear(0, 1, Some(2)).unwrap();
        let a = graph.relu(p).unwrap();
        graph.mse(a, 3).unwrap();
        let shapes = [vec![2, 2], vec![1, 2], vec![1], vec![2, 1]];
        let plan = graph
            .compile_backward(&shapes, &[true, true, true, true])
            .unwrap();
        let mut baseline = plan.create_workspace().unwrap();
        let mut planned = plan.create_planned_workspace().unwrap();
        assert!(planned.allocated_bytes() < baseline.allocated_bytes());
        for offset in [0., 2., -3.] {
            let inputs = vec![
                handle(vec![1. + offset, 2., 3., 4.], shapes[0].clone(), true),
                handle(vec![2., 1.], shapes[1].clone(), true),
                handle(vec![-1.], shapes[2].clone(), true),
                handle(vec![1., 2.], shapes[3].clone(), true),
            ];
            assert_eq!(
                plan.forward(&inputs, &mut planned).unwrap().values(),
                plan.forward(&inputs, &mut baseline).unwrap().values()
            );
            plan.backward(None, &mut planned).unwrap();
            plan.backward(None, &mut baseline).unwrap();
            for i in 0..4 {
                assert_eq!(planned.input_gradient(i), baseline.input_gradient(i));
            }
        }
    }
    #[test]
    fn planned_backward_pins_frozen_intermediate_weights() {
        let mut graph = Graph::new(3);
        let weight = graph.relu(1).unwrap();
        let prediction = graph.linear(0, weight, None).unwrap();
        graph.mse(prediction, 2).unwrap();
        let shapes = [vec![2, 2], vec![1, 2], vec![2, 1]];
        let plan = graph
            .compile_backward(&shapes, &[true, false, false])
            .unwrap();
        let mut ws = plan.create_planned_workspace().unwrap();
        let inputs = vec![
            handle(vec![1., 2., 3., 4.], shapes[0].clone(), true),
            handle(vec![-1., 2.], shapes[1].clone(), false),
            handle(vec![0., 0.], shapes[2].clone(), false),
        ];
        plan.forward(&inputs, &mut ws).unwrap();
        plan.backward(None, &mut ws).unwrap();
        assert_eq!(ws.input_gradient(0).unwrap(), &[0., 8., 0., 16.]);
    }
    #[test]
    fn backward_validates_readiness_plan_seed_and_frozen_versions() {
        let mut graph = Graph::new(2);
        graph.linear(0, 1, None).unwrap();
        let shapes = [vec![1, 2], vec![2, 2]];
        let plan = graph.compile_backward(&shapes, &[true, false]).unwrap();
        let other = graph.compile_backward(&shapes, &[true, false]).unwrap();
        let mut ws = plan.create_workspace().unwrap();
        let inputs = vec![
            handle(vec![2., 3.], shapes[0].clone(), true),
            handle(vec![1., 2., 3., 4.], shapes[1].clone(), false),
        ];
        assert!(plan.backward(None, &mut ws).is_err());
        assert!(other.forward(&inputs, &mut ws).is_err());
        plan.forward(&inputs, &mut ws).unwrap();
        assert!(plan.backward(None, &mut ws).is_err());
        let seed = TensorData::new(vec![2., 3.], vec![1, 2]).unwrap();
        for _ in 0..2 {
            plan.backward(Some(&seed), &mut ws).unwrap();
            assert_eq!(ws.input_gradient(0).unwrap(), &[11., 16.]);
        }
        crate::autograd::replace_batch(
            &[inputs[1].clone()],
            vec![TensorData::new(vec![0.; 4], vec![2, 2]).unwrap()],
        )
        .unwrap();
        assert!(plan.backward(Some(&seed), &mut ws).is_err());
        assert!(ws.input_gradient(0).is_none());
        let fresh = vec![
            handle(vec![1., 2.], shapes[0].clone(), true),
            handle(vec![1.; 4], shapes[1].clone(), false),
        ];
        plan.forward(&fresh, &mut ws).unwrap();
        assert!(plan.forward(&[], &mut ws).is_err());
        assert!(plan.backward(Some(&seed), &mut ws).is_err());
    }
    #[test]
    fn backward_multiple_shapes_and_changed_values_match_eager() {
        for (batch, k, n) in [(1, 3, 2), (4, 2, 5), (2, 0, 3), (0, 2, 3)] {
            let mut graph = Graph::new(3);
            let linear = graph.linear(0, 1, Some(2)).unwrap();
            graph.relu(linear).unwrap();
            let shapes = [vec![batch, k], vec![n, k], vec![n]];
            let plan = graph
                .compile_backward(&shapes, &[true, true, true])
                .unwrap();
            let mut ws = plan.create_planned_workspace().unwrap();
            for offset in [-0.7, 0.3] {
                let inputs: Vec<_> = shapes
                    .iter()
                    .map(|s| {
                        handle(
                            (0..s.iter().product())
                                .map(|i| i as f32 * 0.1 + offset)
                                .collect(),
                            s.clone(),
                            true,
                        )
                    })
                    .collect();
                let seed = TensorData::new(vec![0.5; batch * n], vec![batch, n]).unwrap();
                let eager = inputs[0]
                    .linear(&inputs[1], Some(&inputs[2]))
                    .unwrap()
                    .unary(UnaryOp::Relu)
                    .unwrap();
                eager.backward(Some(&seed), false).unwrap();
                plan.forward(&inputs, &mut ws).unwrap();
                plan.backward(Some(&seed), &mut ws).unwrap();
                for (i, input) in inputs.iter().enumerate() {
                    let want = input.grad().unwrap().unwrap();
                    for (&got, &want) in ws.input_gradient(i).unwrap().iter().zip(want.values()) {
                        assert!((got - want).abs() <= 1e-5 + 1e-5 * want.abs());
                    }
                }
            }
        }
    }
    #[test]
    fn disconnected_nan_branch_is_excluded_and_relu_nan_matches_eager() {
        let mut graph = Graph::new(2);
        graph.relu(0).unwrap();
        graph.relu(1).unwrap();
        let plan = graph
            .compile_backward(&[vec![1], vec![1]], &[true, true])
            .unwrap();
        let mut ws = plan.create_workspace().unwrap();
        let inputs = vec![
            handle(vec![f32::NAN], vec![1], true),
            handle(vec![f32::NAN], vec![1], true),
        ];
        plan.forward(&inputs, &mut ws).unwrap();
        plan.backward(None, &mut ws).unwrap();
        assert!(ws.input_gradient(0).is_none());
        assert_eq!(ws.input_gradient(1).unwrap(), &[1.]);
    }
    #[test]
    fn extracted_input_gradient_moves_allocation_and_refills_identity_plan() {
        let graph = Graph::new(1);
        let plan = graph.compile_backward(&[vec![]], &[true]).unwrap();
        let mut ws = plan.create_workspace().unwrap();
        let inputs = vec![handle(vec![2.], vec![], true)];
        plan.forward(&inputs, &mut ws).unwrap();
        plan.backward(None, &mut ws).unwrap();
        let pointer = ws.input_gradient(0).unwrap().as_ptr();
        let allocated_before = ws.allocated_bytes();
        let extracted = ws.take_input_gradient(0).unwrap();
        assert_eq!(pointer, extracted.as_ptr());
        assert_eq!(
            allocated_before - ws.allocated_bytes(),
            extracted.capacity() * std::mem::size_of::<f32>()
        );
        assert_eq!(extracted, vec![1.]);
        plan.backward(None, &mut ws).unwrap();
        assert_eq!(ws.input_gradient(0).unwrap(), &[1.]);
        assert_ne!(ws.input_gradient(0).unwrap().as_ptr(), extracted.as_ptr());
        assert_eq!(extracted, vec![1.]);
    }
    #[test]
    fn extracted_empty_and_frozen_input_gradients_repeat_correctly() {
        let mut graph = Graph::new(2);
        graph.binary(0, 1, crate::ops::BinaryOp::Add).unwrap();
        let plan = graph
            .compile_backward(&[vec![0], vec![]], &[true, false])
            .unwrap();
        let mut ws = plan.create_workspace().unwrap();
        let inputs = vec![
            handle(vec![], vec![0], true),
            handle(vec![2.], vec![], false),
        ];
        let seed = TensorData::new(vec![], vec![0]).unwrap();
        plan.forward(&inputs, &mut ws).unwrap();
        for _ in 0..2 {
            plan.backward(Some(&seed), &mut ws).unwrap();
            assert_eq!(ws.take_input_gradient(0), Some(vec![]));
            assert!(ws.take_input_gradient(1).is_none());
        }
    }
}
