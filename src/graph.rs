#[path = "graph_backward.rs"]
mod backward;
#[path = "graph_broadcast.rs"]
mod broadcast;
use broadcast::CompiledBroadcast;
#[path = "graph_workspace.rs"]
mod workspace;
pub(crate) use backward::{BackwardPlan, TrainingWorkspace};
pub(crate) use workspace::ExecutionWorkspace;

use std::sync::Arc;

use crate::{
    loss,
    ops::{self, BinaryOp, UnaryOp},
    tensor::{DType, Device, TensorData, TensorSpec},
};

#[derive(Clone, Debug)]
pub enum GraphOp {
    LayerNorm {
        input: usize,
        weight: Option<usize>,
        bias: Option<usize>,
        eps: f32,
        features: Option<usize>,
    },
    Matmul {
        left: usize,
        right: usize,
    },
    Reduce {
        input: usize,
        mean: bool,
    },
    Input(usize),
    Binary {
        left: usize,
        right: usize,
        op: BinaryOp,
    },
    Linear {
        input: usize,
        weight: usize,
        bias: Option<usize>,
    },
    Relu {
        input: usize,
    },
    Mse {
        prediction: usize,
        target: usize,
    },
    CrossEntropy {
        prediction: usize,
        target: usize,
    },
}

#[derive(Clone)]
pub enum Instruction {
    LayerNorm {
        input: usize,
        weight: Option<usize>,
        bias: Option<usize>,
        eps: f32,
        m: usize,
        n: usize,
    },
    Matmul {
        left: usize,
        right: usize,
        m: usize,
        k: usize,
        n: usize,
    },
    Reduce {
        input: usize,
        mean: bool,
    },
    Input(usize),
    Binary {
        left: usize,
        right: usize,
        op: BinaryOp,
        broadcast: CompiledBroadcast,
    },
    Linear {
        input: usize,
        weight: usize,
        bias: Option<usize>,
        m: usize,
        k: usize,
        n: usize,
    },
    Relu {
        input: usize,
    },
    Mse {
        prediction: usize,
        target: usize,
    },
    CrossEntropy {
        prediction: usize,
        target: usize,
    },
}

impl Instruction {
    /// Preserve repeated edges; x+x and x*x need both derivative contributions.
    fn parents(&self) -> Vec<usize> {
        match self {
            Self::Input(_) => vec![],
            Self::LayerNorm {
                input,
                weight,
                bias,
                ..
            } => {
                let mut p = vec![*input];
                p.extend(*weight);
                p.extend(*bias);
                p
            }
            Self::Matmul { left, right, .. } => vec![*left, *right],
            Self::Reduce { input, .. } => vec![*input],
            Self::Binary { left, right, .. } => vec![*left, *right],
            Self::Linear {
                input,
                weight,
                bias,
                ..
            } => {
                let mut p = vec![*input, *weight];
                p.extend(*bias);
                p
            }
            Self::Relu { input } => vec![*input],
            Self::Mse { prediction, target } | Self::CrossEntropy { prediction, target } => {
                vec![*prediction, *target]
            }
        }
    }
}

pub struct ExecutionPlan {
    instructions: Vec<Instruction>,
    specs: Vec<TensorSpec>,
    input_count: usize,
    identity: Arc<()>,
}

impl ExecutionPlan {
    pub fn execute(&self, inputs: &[TensorData]) -> Result<TensorData, String> {
        self.validate_inputs(&inputs.iter().collect::<Vec<_>>())?;

        let mut values = Vec::with_capacity(self.instructions.len());

        for instruction in &self.instructions {
            let value = match instruction {
                Instruction::Input(index) => inputs[*index].clone(),
                Instruction::LayerNorm {
                    input,
                    weight,
                    bias,
                    eps,
                    ..
                } => crate::normalization::forward(
                    &values[*input],
                    weight.map(|i| &values[i]),
                    bias.map(|i| &values[i]),
                    *eps,
                )?,
                Instruction::Matmul { left, right, .. } => {
                    ops::matmul(&values[*left], &values[*right])?
                }
                Instruction::Reduce { input, mean } => ops::unary(
                    &values[*input],
                    &if *mean { UnaryOp::Mean } else { UnaryOp::Sum },
                )?,
                Instruction::Binary {
                    left, right, op, ..
                } => ops::binary(&values[*left], &values[*right], *op)?,

                Instruction::Linear {
                    input,
                    weight,
                    bias,
                    ..
                } => ops::linear(
                    &values[*input],
                    &values[*weight],
                    bias.map(|index| &values[index]),
                )?,

                Instruction::Relu { input } => ops::unary(&values[*input], &UnaryOp::Relu)?,

                Instruction::Mse { prediction, target } => {
                    loss::mse_loss(&values[*prediction], &values[*target])?
                }
                Instruction::CrossEntropy { prediction, target } => {
                    let labels =
                        loss::tensor_labels(values[*prediction].shape(), &values[*target])?;
                    loss::cross_entropy(&values[*prediction], &labels)?
                }
            };

            values.push(value);
        }

        values.pop().ok_or_else(|| "empty execution plan".into())
    }

    /// Compact compiled broadcasts retain no output-sized index maps.
    pub fn broadcast_map_bytes(&self) -> usize {
        0
    }
    /// Descriptor storage including enum padding; excludes other plan metadata.
    pub fn broadcast_descriptor_bytes(&self) -> usize {
        self.instructions
            .iter()
            .filter(|op| matches!(op, Instruction::Binary { .. }))
            .count()
            * std::mem::size_of::<CompiledBroadcast>()
    }
    pub fn specs(&self) -> &[TensorSpec] {
        &self.specs
    }
    fn validate_inputs(&self, inputs: &[&TensorData]) -> Result<(), String> {
        if inputs.len() != self.input_count {
            return Err("compiled graph input count mismatch".into());
        }
        for (index, input) in inputs.iter().enumerate() {
            if !self.specs[index].matches(input) {
                self.specs[index].check(input, &format!("compiled graph input {index}"))?;
            }
        }
        Ok(())
    }
}

pub struct Graph {
    nodes: Vec<GraphOp>,
    input_count: usize,
}

impl Graph {
    pub fn new(input_count: usize) -> Self {
        let nodes = (0..input_count).map(GraphOp::Input).collect();

        Self { nodes, input_count }
    }

    fn check_ref(&self, index: usize) -> Result<(), String> {
        if index >= self.nodes.len() {
            return Err(format!("invalid graph node reference: {index}"));
        }
        Ok(())
    }

    pub fn binary(&mut self, left: usize, right: usize, op: BinaryOp) -> Result<usize, String> {
        self.check_ref(left)?;
        self.check_ref(right)?;
        let index = self.nodes.len();
        self.nodes.push(GraphOp::Binary { left, right, op });
        Ok(index)
    }
    pub fn matmul(&mut self, left: usize, right: usize) -> Result<usize, String> {
        self.check_ref(left)?;
        self.check_ref(right)?;
        let index = self.nodes.len();
        self.nodes.push(GraphOp::Matmul { left, right });
        Ok(index)
    }
    pub fn reduce(&mut self, input: usize, mean: bool) -> Result<usize, String> {
        self.check_ref(input)?;
        let index = self.nodes.len();
        self.nodes.push(GraphOp::Reduce { input, mean });
        Ok(index)
    }
    pub fn layer_norm(
        &mut self,
        input: usize,
        weight: Option<usize>,
        bias: Option<usize>,
        eps: f32,
        features: Option<usize>,
    ) -> Result<usize, String> {
        self.check_ref(input)?;
        for i in [weight, bias].into_iter().flatten() {
            self.check_ref(i)?;
        }
        crate::normalization::validate_epsilon(eps)?;
        let index = self.nodes.len();
        self.nodes.push(GraphOp::LayerNorm {
            input,
            weight,
            bias,
            eps,
            features,
        });
        Ok(index)
    }
    pub fn linear(
        &mut self,
        input: usize,
        weight: usize,
        bias: Option<usize>,
    ) -> Result<usize, String> {
        self.check_ref(input)?;
        self.check_ref(weight)?;

        if let Some(bias) = bias {
            self.check_ref(bias)?;
        }

        let index = self.nodes.len();

        self.nodes.push(GraphOp::Linear {
            input,
            weight,
            bias,
        });

        Ok(index)
    }

    pub fn relu(&mut self, input: usize) -> Result<usize, String> {
        self.check_ref(input)?;

        let index = self.nodes.len();
        self.nodes.push(GraphOp::Relu { input });

        Ok(index)
    }

    pub fn mse(&mut self, prediction: usize, target: usize) -> Result<usize, String> {
        self.check_ref(prediction)?;
        self.check_ref(target)?;

        let index = self.nodes.len();

        self.nodes.push(GraphOp::Mse { prediction, target });

        Ok(index)
    }

    pub fn cross_entropy(&mut self, prediction: usize, target: usize) -> Result<usize, String> {
        self.check_ref(prediction)?;
        self.check_ref(target)?;
        let index = self.nodes.len();
        self.nodes
            .push(GraphOp::CrossEntropy { prediction, target });
        Ok(index)
    }

    pub fn compile(&self, input_shapes: &[Vec<usize>]) -> Result<ExecutionPlan, String> {
        let specs = input_shapes
            .iter()
            .map(|shape| TensorSpec::new(shape.clone(), Device::Cpu, DType::Float32))
            .collect::<Result<Vec<_>, _>>()?;
        self.compile_specs(&specs)
    }

    pub fn compile_specs(&self, input_specs: &[TensorSpec]) -> Result<ExecutionPlan, String> {
        // Specialized workspaces currently lower CPU float32 only. Adding a
        // real variant requires an explicit lowering decision, even for an input-only graph.
        for spec in input_specs {
            match (spec.device(), spec.dtype()) {
                (Device::Cpu, DType::Float32) => {}
                (Device::Cpu, _) => {
                    return Err("compiled workspaces support float32 only".into());
                }
            }
        }
        let input_shapes: Vec<_> = input_specs.iter().map(|s| s.shape().to_vec()).collect();
        let shapes = self.infer_shapes(&input_shapes)?;

        if self.nodes.is_empty() {
            return Err("cannot compile an empty graph".into());
        }

        let instructions = self
            .nodes
            .iter()
            .map(|op| {
                let instruction = match op {
                    GraphOp::Input(index) => Instruction::Input(*index),
                    GraphOp::LayerNorm {
                        input,
                        weight,
                        bias,
                        eps,
                        ..
                    } => Instruction::LayerNorm {
                        input: *input,
                        weight: *weight,
                        bias: *bias,
                        eps: *eps,
                        m: shapes[*input][0],
                        n: shapes[*input][1],
                    },
                    GraphOp::Matmul { left, right } => Instruction::Matmul {
                        left: *left,
                        right: *right,
                        m: shapes[*left][0],
                        k: shapes[*left][1],
                        n: shapes[*right][1],
                    },
                    GraphOp::Reduce { input, mean } => Instruction::Reduce {
                        input: *input,
                        mean: *mean,
                    },
                    GraphOp::Binary { left, right, op } => Instruction::Binary {
                        left: *left,
                        right: *right,
                        op: *op,
                        broadcast: CompiledBroadcast::new(&shapes[*left], &shapes[*right])?,
                    },

                    GraphOp::Linear {
                        input,
                        weight,
                        bias,
                    } => Instruction::Linear {
                        input: *input,
                        weight: *weight,
                        bias: *bias,
                        m: shapes[*input][0],
                        k: shapes[*input][1],
                        n: shapes[*weight][0],
                    },

                    GraphOp::Relu { input } => Instruction::Relu { input: *input },

                    GraphOp::Mse { prediction, target } => Instruction::Mse {
                        prediction: *prediction,
                        target: *target,
                    },
                    GraphOp::CrossEntropy { prediction, target } => Instruction::CrossEntropy {
                        prediction: *prediction,
                        target: *target,
                    },
                };
                Ok(instruction)
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut specs: Vec<TensorSpec> = Vec::with_capacity(instructions.len());
        for (instruction, shape) in instructions.iter().zip(shapes) {
            let spec = if let Instruction::Input(index) = instruction {
                input_specs[*index].clone()
            } else {
                let parents = instruction.parents();
                let first = &specs[parents[0]];
                // Compiled CE labels currently use detached float32 class-index metadata.
                if parents.iter().any(|&i| {
                    specs[i].device() != first.device() || specs[i].dtype() != first.dtype()
                }) {
                    return Err("compiled operands must share device and dtype".into());
                }
                TensorSpec::new(shape, first.device(), first.dtype())?
            };
            specs.push(spec);
        }
        Ok(ExecutionPlan {
            instructions,
            specs,
            input_count: self.input_count,
            identity: Arc::new(()),
        })
    }

    pub fn infer_shapes(&self, input_shapes: &[Vec<usize>]) -> Result<Vec<Vec<usize>>, String> {
        if input_shapes.len() != self.input_count {
            return Err(format!(
                "expected {} input shapes, received {}",
                self.input_count,
                input_shapes.len()
            ));
        }

        let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(self.nodes.len());

        for op in &self.nodes {
            let shape = match op {
                GraphOp::Input(index) => input_shapes[*index].clone(),
                GraphOp::LayerNorm {
                    input,
                    weight,
                    bias,
                    eps,
                    features,
                } => crate::dispatch::infer_layer_norm_shape(
                    &shapes[*input],
                    weight.map(|i| shapes[i].as_slice()),
                    bias.map(|i| shapes[i].as_slice()),
                    *eps,
                    *features,
                )?,
                GraphOp::Reduce { input, .. } => {
                    crate::dispatch::infer_reduction_shape(&shapes[*input], None, false)?
                }
                GraphOp::Matmul { left, right } => {
                    crate::dispatch::infer_matmul_shape(&shapes[*left], &shapes[*right])?
                }
                GraphOp::Binary { left, right, op } => {
                    crate::dispatch::infer_binary_shape(*op, &shapes[*left], &shapes[*right])?
                }

                GraphOp::Linear {
                    input,
                    weight,
                    bias,
                } => crate::dispatch::infer_linear_shape(
                    &shapes[*input],
                    &shapes[*weight],
                    bias.map(|i| shapes[i].as_slice()),
                )?,

                GraphOp::Relu { input } => crate::dispatch::infer_relu_shape(&shapes[*input])?,

                GraphOp::Mse { prediction, target } => {
                    crate::dispatch::infer_mse_shape(&shapes[*prediction], &shapes[*target])?
                }
                GraphOp::CrossEntropy { prediction, target } => {
                    crate::dispatch::infer_cross_entropy_shape(
                        &shapes[*prediction],
                        &shapes[*target],
                    )?
                }
            };

            shapes.push(shape);
        }

        Ok(shapes)
    }

    pub fn execute(&self, inputs: &[TensorData]) -> Result<TensorData, String> {
        if inputs.len() != self.input_count {
            return Err(format!(
                "expected {} inputs, received {}",
                self.input_count,
                inputs.len()
            ));
        }

        if self.nodes.is_empty() {
            return Err("cannot execute an empty graph".into());
        }

        // Initial implementation: retain every intermediate.
        // A later memory planner will replace this allocation strategy.
        let mut values: Vec<TensorData> = Vec::with_capacity(self.nodes.len());

        for op in &self.nodes {
            let result = match op {
                GraphOp::Input(index) => inputs[*index].clone(),
                GraphOp::LayerNorm {
                    input,
                    weight,
                    bias,
                    eps,
                    ..
                } => crate::normalization::forward(
                    &values[*input],
                    weight.map(|i| &values[i]),
                    bias.map(|i| &values[i]),
                    *eps,
                )?,
                GraphOp::Matmul { left, right } => ops::matmul(&values[*left], &values[*right])?,
                GraphOp::Reduce { input, mean } => ops::unary(
                    &values[*input],
                    &if *mean { UnaryOp::Mean } else { UnaryOp::Sum },
                )?,
                GraphOp::Binary { left, right, op } => {
                    ops::binary(&values[*left], &values[*right], *op)?
                }

                GraphOp::Linear {
                    input,
                    weight,
                    bias,
                } => ops::linear(
                    &values[*input],
                    &values[*weight],
                    bias.map(|index| &values[index]),
                )?,

                GraphOp::Relu { input } => ops::unary(&values[*input], &UnaryOp::Relu)?,

                GraphOp::Mse { prediction, target } => {
                    loss::mse_loss(&values[*prediction], &values[*target])?
                }
                GraphOp::CrossEntropy { prediction, target } => {
                    let labels =
                        loss::tensor_labels(values[*prediction].shape(), &values[*target])?;
                    loss::cross_entropy(&values[*prediction], &labels)?
                }
            };

            values.push(result);
        }

        values
            .pop()
            .ok_or_else(|| "graph produced no output".into())
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: Vec<f32>, shape: Vec<usize>) -> TensorData {
        TensorData::new(data, shape).unwrap()
    }

    #[test]
    fn graph_matches_eager_execution() {
        // Inputs:
        // x, w1, b1, w2, b2, target
        let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let w1 = tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let b1 = tensor(vec![0.0, 0.0], vec![2]);
        let w2 = tensor(vec![2.0, 1.0], vec![1, 2]);
        let b2 = tensor(vec![-2.0], vec![1]);
        let target = tensor(vec![0.0, 0.0], vec![2, 1]);

        let inputs = vec![x, w1, b1, w2, b2, target];

        let mut graph = Graph::new(6);

        let hidden = graph.linear(0, 1, Some(2)).unwrap();
        let activated = graph.relu(hidden).unwrap();
        let prediction = graph.linear(activated, 3, Some(4)).unwrap();
        graph.mse(prediction, 5).unwrap();

        let compiled_result = graph.execute(&inputs).unwrap();

        let eager_hidden = ops::linear(&inputs[0], &inputs[1], Some(&inputs[2])).unwrap();
        let eager_activated = ops::unary(&eager_hidden, &UnaryOp::Relu).unwrap();
        let eager_prediction = ops::linear(&eager_activated, &inputs[3], Some(&inputs[4])).unwrap();
        let eager_result = loss::mse_loss(&eager_prediction, &inputs[5]).unwrap();

        assert_eq!(compiled_result.shape(), eager_result.shape());
        assert_eq!(compiled_result.values(), eager_result.values());

        assert_eq!(graph.node_count(), 10);
    }

    #[test]
    fn rejects_invalid_references_and_inputs() {
        let mut graph = Graph::new(2);

        assert!(graph.linear(0, 100, None).is_err());
        assert!(graph.relu(100).is_err());
        assert!(graph.mse(0, 100).is_err());

        assert!(graph.execute(&[]).is_err());
    }

    #[test]
    fn graph_is_reusable() {
        let mut graph = Graph::new(1);
        graph.relu(0).unwrap();

        let positive = tensor(vec![-1.0, 2.0], vec![2]);
        let negative = tensor(vec![3.0, -4.0], vec![2]);

        assert_eq!(graph.execute(&[positive]).unwrap().values(), &[0.0, 2.0]);

        assert_eq!(graph.execute(&[negative]).unwrap().values(), &[3.0, 0.0]);
    }
}

#[cfg(test)]
mod shape_tests {
    use super::*;

    #[test]
    fn infers_mlp_shapes() {
        let mut graph = Graph::new(6);

        let hidden = graph.linear(0, 1, Some(2)).unwrap();
        let activated = graph.relu(hidden).unwrap();
        let output = graph.linear(activated, 3, Some(4)).unwrap();
        let loss = graph.mse(output, 5).unwrap();

        let shapes = graph
            .infer_shapes(&[
                vec![32, 64],
                vec![128, 64],
                vec![128],
                vec![32, 128],
                vec![32],
                vec![32, 32],
            ])
            .unwrap();

        assert_eq!(shapes[hidden], vec![32, 128]);
        assert_eq!(shapes[activated], vec![32, 128]);
        assert_eq!(shapes[output], vec![32, 32]);
        assert_eq!(shapes[loss], Vec::<usize>::new());
    }

    #[test]
    fn rejects_invalid_shapes() {
        let mut graph = Graph::new(3);
        graph.linear(0, 1, Some(2)).unwrap();

        assert!(
            graph
                .infer_shapes(&[vec![32, 64], vec![128, 65], vec![128],])
                .is_err()
        );

        assert!(
            graph
                .infer_shapes(&[vec![32, 64], vec![128, 64], vec![127],])
                .is_err()
        );

        assert!(graph.infer_shapes(&[vec![32, 64],]).is_err());
    }

    #[test]
    fn infers_branching_graph() {
        let mut graph = Graph::new(2);

        let first = graph.relu(0).unwrap();
        let second = graph.relu(0).unwrap();
        let loss = graph.mse(first, second).unwrap();

        let shapes = graph.infer_shapes(&[vec![8, 16], vec![8, 16]]).unwrap();

        assert_eq!(shapes[first], vec![8, 16]);
        assert_eq!(shapes[second], vec![8, 16]);
        assert_eq!(shapes[loss], Vec::<usize>::new());
    }
}

#[cfg(test)]
mod compiled_tests {
    use super::*;

    fn tensor(data: Vec<f32>, shape: Vec<usize>) -> TensorData {
        TensorData::new(data, shape).unwrap()
    }

    #[test]
    fn compiled_matches_eager_mlp() {
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

        let shapes: Vec<Vec<usize>> = inputs.iter().map(|x| x.shape().to_vec()).collect();

        let plan = graph.compile(&shapes).unwrap();

        let eager = graph.execute(&inputs).unwrap();
        let compiled = plan.execute(&inputs).unwrap();

        assert_eq!(eager.shape(), compiled.shape());
        assert_eq!(eager.values(), compiled.values());

        assert_eq!(plan.specs()[hidden].shape(), &[2, 2]);
        assert_eq!(plan.specs()[prediction].shape(), &[2, 1]);
    }

    #[test]
    fn compiled_plan_is_reusable() {
        let mut graph = Graph::new(2);

        let first = graph.relu(0).unwrap();
        let second = graph.relu(0).unwrap();
        graph.mse(first, second).unwrap();

        let plan = graph.compile(&[vec![2], vec![2]]).unwrap();

        for values in [vec![-2.0, 3.0], vec![4.0, -5.0], vec![0.0, 0.0]] {
            let inputs = vec![tensor(values, vec![2]), tensor(vec![7.0, 8.0], vec![2])];

            let eager = graph.execute(&inputs).unwrap();
            let compiled = plan.execute(&inputs).unwrap();

            assert_eq!(eager.values(), compiled.values());
        }
    }

    #[test]
    fn compiled_rejects_shape_mismatch() {
        let mut graph = Graph::new(1);
        graph.relu(0).unwrap();

        let plan = graph.compile(&[vec![2, 3]]).unwrap();

        let incorrect = tensor(vec![1.0; 6], vec![3, 2]);

        assert!(plan.execute(&[incorrect]).is_err());
        assert!(plan.execute(&[]).is_err());
    }

    #[test]
    fn compiled_rejects_invalid_linear_shapes() {
        let mut graph = Graph::new(2);
        graph.linear(0, 1, None).unwrap();

        assert!(graph.compile(&[vec![32, 64], vec![128, 65]]).is_err());
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    fn tensor(shape: Vec<usize>, seed: usize) -> TensorData {
        let count: usize = shape.iter().product();

        let data = (0..count)
            .map(|i| ((i + seed) % 101) as f32 / 101.0)
            .collect();

        TensorData::new(data, shape).unwrap()
    }

    fn benchmark(batch: usize) {
        let mut graph = Graph::new(6);

        let hidden = graph.linear(0, 1, Some(2)).unwrap();
        let activated = graph.relu(hidden).unwrap();
        let prediction = graph.linear(activated, 3, Some(4)).unwrap();
        graph.mse(prediction, 5).unwrap();

        let inputs = vec![
            tensor(vec![batch, 64], 1),
            tensor(vec![128, 64], 2),
            tensor(vec![128], 3),
            tensor(vec![32, 128], 4),
            tensor(vec![32], 5),
            tensor(vec![batch, 32], 6),
        ];

        let shapes: Vec<Vec<usize>> = inputs.iter().map(|x| x.shape().to_vec()).collect();

        let plan = graph.compile(&shapes).unwrap();
        let mut workspace = plan.create_workspace().unwrap();

        // Confirm numerical equivalence before timing.
        let eager = graph.execute(&inputs).unwrap();
        let compiled = plan.execute(&inputs).unwrap();
        let reused = plan
            .execute_with_workspace(&inputs, &mut workspace)
            .unwrap();

        assert_eq!(eager.values(), compiled.values());
        assert_eq!(eager.values(), reused.values());

        const WARMUP: usize = 500;
        const ITERATIONS: usize = 5000;

        for _ in 0..WARMUP {
            black_box(graph.execute(black_box(&inputs)).unwrap());
            black_box(plan.execute(black_box(&inputs)).unwrap());
            black_box(
                plan.execute_with_workspace(black_box(&inputs), &mut workspace)
                    .unwrap(),
            );
        }

        fn time<F: FnMut()>(mut f: F) -> f64 {
            let start = Instant::now();

            for _ in 0..ITERATIONS {
                f();
            }

            start.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64
        }

        let eager_us = time(|| {
            black_box(graph.execute(black_box(&inputs)).unwrap());
        });

        let compiled_us = time(|| {
            black_box(plan.execute(black_box(&inputs)).unwrap());
        });

        let workspace_us = time(|| {
            black_box(
                plan.execute_with_workspace(black_box(&inputs), &mut workspace)
                    .unwrap(),
            );
        });

        println!();
        println!("=== DenseCore Graph Benchmark: batch {batch} ===");
        println!("Eager graph:       {eager_us:.3} µs");
        println!("Compiled:          {compiled_us:.3} µs");
        println!("Compiled workspace:{workspace_us:.3} µs");
        println!("Workspace speedup: {:.2}x", eager_us / workspace_us);
    }

    #[test]
    #[ignore = "manual performance benchmark"]
    fn benchmark_batch1() {
        benchmark(1);
    }

    #[test]
    #[ignore = "manual performance benchmark"]
    fn benchmark_batch32() {
        benchmark(32);
    }
}

#[cfg(test)]
mod binary_tests {
    use super::*;
    use crate::{autograd::TensorHandle, ops::BinaryOp};
    fn handle(shape: Vec<usize>, offset: f32, grad: bool) -> TensorHandle {
        TensorHandle::new(
            TensorData::new(
                (0..shape.iter().product())
                    .map(|i| i as f32 * 0.2 + offset)
                    .collect(),
                shape,
            )
            .unwrap(),
            grad,
        )
    }
    #[test]
    fn compiled_broadcast_metadata_stays_small_for_large_output() {
        let mut graph = Graph::new(2);
        graph.binary(0, 1, BinaryOp::Mul).unwrap();
        let plan = graph.compile(&[vec![128, 256], vec![256]]).unwrap();
        assert!(
            plan.broadcast_map_bytes() < 1024,
            "compiled broadcasting retains output-sized index maps"
        );
    }
    #[test]
    fn binary_graph_broadcast_vjp_matches_eager_in_both_orders() {
        for (a, b) in [
            (vec![2, 3], vec![3]),
            (vec![3], vec![2, 3]),
            (vec![2, 3], vec![]),
            (vec![], vec![2, 3]),
            (vec![2, 3], vec![2, 3]),
            (vec![0, 3], vec![3]),
            (vec![2, 0], vec![0]),
            (vec![2, 1, 3], vec![1, 4, 1]),
            (vec![1, 0, 3], vec![1, 1, 3]),
        ] {
            for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Sub, BinaryOp::Div] {
                let mut graph = Graph::new(2);
                graph.binary(0, 1, op).unwrap();
                let plan = graph
                    .compile_backward(&[a.clone(), b.clone()], &[true, true])
                    .unwrap();
                let mut ws = plan.create_planned_workspace().unwrap();
                for offset in [-0.3, 0.7] {
                    let inputs = [
                        handle(a.clone(), offset, true),
                        handle(b.clone(), 0.5, true),
                    ];
                    let eager = inputs[0].binary(&inputs[1], op).unwrap();
                    let seed = TensorData::new(
                        vec![0.7; eager.value().values().len()],
                        eager.value().shape().to_vec(),
                    )
                    .unwrap();
                    eager.backward(Some(&seed), false).unwrap();
                    assert_eq!(
                        plan.forward(&inputs, &mut ws).unwrap().values(),
                        eager.value().values()
                    );
                    plan.backward(Some(&seed), &mut ws).unwrap();
                    for (i, input) in inputs.iter().enumerate() {
                        let expected = input.grad().unwrap().unwrap();
                        for (&got, &want) in
                            ws.input_gradient(i).unwrap().iter().zip(expected.values())
                        {
                            assert!((got - want).abs() < 1e-6);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn residual_diamond_shared_weights_and_duplicate_binary_edges() {
        for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Sub, BinaryOp::Div] {
            let mut graph = Graph::new(2);
            let first = graph.linear(0, 1, None).unwrap();
            let second = graph.linear(0, 1, None).unwrap();
            let joined = graph.binary(first, second, op).unwrap();
            graph.binary(joined, joined, BinaryOp::Add).unwrap();
            let plan = graph
                .compile_backward(&[vec![2, 2], vec![2, 2]], &[true, true])
                .unwrap();
            let mut ws = plan.create_planned_workspace().unwrap();
            let inputs = [handle(vec![2, 2], 0.2, true), handle(vec![2, 2], 0.3, true)];
            let a = inputs[0].linear(&inputs[1], None).unwrap();
            let b = inputs[0].linear(&inputs[1], None).unwrap();
            let joined = a.binary(&b, op).unwrap();
            let eager = joined.binary(&joined, BinaryOp::Add).unwrap();
            let seed = TensorData::new(vec![1.; 4], vec![2, 2]).unwrap();
            eager.backward(Some(&seed), false).unwrap();
            plan.forward(&inputs, &mut ws).unwrap();
            plan.backward(Some(&seed), &mut ws).unwrap();
            for (i, input) in inputs.iter().enumerate() {
                assert_eq!(
                    ws.input_gradient(i).unwrap(),
                    input.grad().unwrap().unwrap().values()
                );
            }
        }
    }
    #[test]
    fn binary_invalid_broadcast_and_frozen_intermediate_mul_pinning() {
        let mut graph = Graph::new(2);
        let frozen = graph.relu(1).unwrap();
        graph.binary(0, frozen, BinaryOp::Mul).unwrap();
        assert!(graph.compile(&[vec![2, 2], vec![3]]).is_err());
        let plan = graph
            .compile_backward(&[vec![2, 2], vec![2]], &[true, false])
            .unwrap();
        let mut ws = plan.create_planned_workspace().unwrap();
        let inputs = [handle(vec![2, 2], 0.3, true), handle(vec![2], -0.1, false)];
        let seed = TensorData::new(vec![2.; 4], vec![2, 2]).unwrap();
        plan.forward(&inputs, &mut ws).unwrap();
        plan.backward(Some(&seed), &mut ws).unwrap();
        assert_eq!(ws.input_gradient(0).unwrap(), &[0., 0.2, 0., 0.2]);
        assert!(ws.input_gradient(1).is_none());
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use crate::tensor::{DType, Device, TensorSpec};

    #[test]
    fn all_supported_instructions_propagate_checked_metadata() {
        let mut graph = Graph::new(4);
        let linear = graph.linear(0, 1, Some(2)).unwrap();
        let relu = graph.relu(linear).unwrap();
        let norm = graph
            .layer_norm(relu, Some(2), None, 1e-5, Some(2))
            .unwrap();
        let mat = graph.matmul(norm, 1).unwrap();
        let add = graph.binary(mat, 0, BinaryOp::Add).unwrap();
        let mul = graph.binary(add, 0, BinaryOp::Mul).unwrap();
        let sum = graph.reduce(mul, false).unwrap();
        let mean = graph.reduce(mul, true).unwrap();
        let mse = graph.mse(mul, 3).unwrap();
        let ce = graph.cross_entropy(mat, 2).unwrap();
        let shapes = [vec![2, 2], vec![2, 2], vec![2], vec![2, 2]];
        let inputs: Vec<_> = shapes
            .iter()
            .map(|s| TensorSpec::new(s.clone(), Device::Cpu, DType::Float32).unwrap())
            .collect();
        let plan = graph.compile_specs(&inputs).unwrap();
        for spec in plan.specs() {
            assert_eq!(spec.device(), Device::Cpu);
            assert_eq!(spec.dtype(), DType::Float32);
        }
        for index in [linear, relu, norm, mat, add, mul] {
            assert_eq!(plan.specs()[index].shape(), &[2, 2]);
        }
        for index in [sum, mean, mse, ce] {
            assert!(plan.specs()[index].shape().is_empty());
        }
    }

    #[test]
    fn input_only_compilation_rejects_overflowing_shapes() {
        assert!(Graph::new(1).compile(&[vec![usize::MAX, 2]]).is_err());
        assert!(TensorSpec::new(vec![usize::MAX, 2], Device::Cpu, DType::Float32).is_err());
        let plan = Graph::new(1).compile(&[vec![0, usize::MAX, 2]]).unwrap();
        let input = TensorData::new(vec![], vec![0, usize::MAX, 2]).unwrap();
        assert!(plan.specs()[0].matches(&input));
        let mut workspace = plan.create_planned_workspace().unwrap();
        assert_eq!(
            plan.execute_with_workspace(std::slice::from_ref(&input), &mut workspace)
                .unwrap()
                .shape(),
            &[0, usize::MAX, 2]
        );
        assert_eq!(plan.execute(&[input]).unwrap().shape(), &[0, usize::MAX, 2]);
    }

    #[test]
    fn typed_plan_validates_inputs_and_reuses_workspace_after_rejection() {
        let mut graph = Graph::new(1);
        graph.relu(0).unwrap();
        let input = TensorData::new(vec![-2., 3.], vec![1, 2]).unwrap();
        let wrong = TensorData::new(vec![-2., 3.], vec![2, 1]).unwrap();
        let plan = graph
            .compile_specs(&[TensorSpec::from_tensor(&input)])
            .unwrap();
        let mut ws = plan.create_planned_workspace().unwrap();
        assert!(plan.execute(std::slice::from_ref(&wrong)).is_err());
        assert!(plan.execute_with_workspace(&[wrong], &mut ws).is_err());
        let result = plan.execute_with_workspace(&[input], &mut ws).unwrap();
        assert_eq!(result.values(), &[0., 3.]);
        assert!(plan.specs().last().unwrap().matches(&result));
    }
    #[test]
    fn division_planned_workspace_preserves_both_operands_for_denominator_vjp() {
        for mask in [[true, false], [false, true], [true, true]] {
            let mut graph = Graph::new(2);
            let numerator = graph.binary(0, 0, BinaryOp::Add).unwrap();
            let denominator = graph.binary(1, 1, BinaryOp::Add).unwrap();
            let quotient = graph.binary(numerator, denominator, BinaryOp::Div).unwrap();
            graph.binary(quotient, quotient, BinaryOp::Add).unwrap();
            let plan = graph
                .compile_backward(&[vec![2, 3], vec![2, 1]], &mask)
                .unwrap();
            let mut ws = plan.create_planned_workspace().unwrap();
            let inputs = [
                crate::autograd::TensorHandle::new(
                    TensorData::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap(),
                    mask[0],
                ),
                crate::autograd::TensorHandle::new(
                    TensorData::new(vec![2., 3.], vec![2, 1]).unwrap(),
                    mask[1],
                ),
            ];
            let a = inputs[0].binary(&inputs[0], BinaryOp::Add).unwrap();
            let b = inputs[1].binary(&inputs[1], BinaryOp::Add).unwrap();
            let q = a.binary(&b, BinaryOp::Div).unwrap();
            let eager = q.binary(&q, BinaryOp::Add).unwrap();
            let seed = TensorData::new(vec![1.; 6], vec![2, 3]).unwrap();
            eager.backward(Some(&seed), false).unwrap();
            for _ in 0..2 {
                plan.forward(&inputs, &mut ws).unwrap();
                plan.backward(Some(&seed), &mut ws).unwrap();
                for (index, input) in inputs.iter().enumerate() {
                    if mask[index] {
                        let expected = input.grad().unwrap().unwrap();
                        for (&got, &want) in ws
                            .input_gradient(index)
                            .unwrap()
                            .iter()
                            .zip(expected.values())
                        {
                            assert!((got - want).abs() < 1e-6);
                        }
                    } else {
                        assert!(ws.input_gradient(index).is_none());
                    }
                }
            }
        }
    }
}
