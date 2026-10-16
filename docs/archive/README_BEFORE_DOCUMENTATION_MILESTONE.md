# Historical README before documentation milestone

Archived from commit b2e5c10 on 2026-10-14. This snapshot contains superseded statements. Use the root README for current instructions. Stage reports and experiment records remain at their original paths.

# Densecore dense training engine

MIT Rust/Python float32 CPU framework for dense-network training and inference, with eager autograd and optional guarded graph compilation. Familiar supported Python API: registered modules, parameters, autograd, mean MSE/CrossEntropy, native SGD/Adam and copied state dictionaries. Package name is provisional; no PyPI/GitHub publication yet.

Version 0.10 local candidate adds opt-in fused training. Current validation details: docs/NATIVE_TRAINING_ENGINEERING_REPORT.md. Version 0.9 platform evidence below remains historical.

## Install a local wheel

```sh
python -m pip install /absolute/path/to/densecore-0.10.0-cp313-cp313-macosx_11_0_arm64.whl
```

NumPy and safetensors are runtime Python dependencies. Prebuilt wheels need no Rust compiler or PyTorch. Version 0.9 was tested with CPython 3.13 on macOS ARM64 and Linux ARM64. Linux x86-64 validation is recorded in docs/LOCAL_RELEASE.md. Other Python versions are covered by prepared CI, which has not run remotely.

## Run the pretrained real-data classifier

From this repository after installing the wheel:

```sh
python examples/predict_letter.py 2 8 3 5 1 8 13 0 6 6 10 8 0 8 0 8
```

Expected class: `T`. The included safetensors checkpoint needs neither PyTorch nor a dataset download. Inputs are the 16 UCI Letter Recognition features, not images or arbitrary text. Dataset and derived artifacts require the attribution in docs/DATASET_ATTRIBUTION.md.

Heldout accuracy: 95.85%, versus matched PyTorch training at 96.09%. Smallest measured training case was about 3× faster with about 6.6× lower peak process RSS; largest case was slower. See [full results](benchmarks/LETTER_RESULTS.md) for all cases, protocol and limits.

## Train

```python
from densecore import tensor, nn, optim, manual_seed
manual_seed(7)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
x = tensor([[-2, -1], [1, 2]])
labels = [0, 1]  # Integer label metadata
for _ in range(100):
    optimizer.zero_grad()
    loss = loss_fn(model(x), labels)
    loss.backward()
    optimizer.step()
print(model(x).numpy().argmax(axis=1))
```

`examples/train_dense.py` runs deterministic regression and classification checks with predeclared heldout rules. These are workflow examples, not speed benchmarks.

## Supported API

`Tensor`/`tensor`: float32 CPU construction, NumPy copies, shape/dtype/device, requires_grad/is_leaf/grad, backward and leaf zero_grad. Gradients and NumPy outputs are detached copies. Input arrays cannot mutate native storage. Addition/multiplication support equal shapes, rank-0 scalars or row bias `(features,)` with `(batch,features)`. NumPy ndarray arithmetic is rejected; convert it explicitly. Real NumPy scalars work.

Rank-2 matmul/transpose, copying reshape, ReLU and whole-tensor sum/mean record first-order gradients. No inferred reshape -1, axis reductions or broader broadcasting. Empty sum is zero, empty mean NaN, and zero-width matmul produces zeros. Higher-order gradients are unsupported.

`nn.Module` supports custom forward, automatic Parameter/Module attribute registration, named_parameters/parameters, train/eval, zero_grad, state_dict and strict load_state_dict. Shared parameters deduplicate in parameters; all names appear in state_dict. Loading validates every name/shape and shared-alias value before mutation, preserves Parameter objects and invalidates saved graphs. No buffer registration, hooks or automatic registration of modules in ordinary lists/dicts. Sequential registers child names 0,1,...; Linear accepts 2D batches and initializes weight/bias within +/-1/sqrt(in_features). manual_seed controls this initialization.

MSELoss/functional.mse_loss require equal-shape Tensor inputs and mean reduction. CrossEntropyLoss/functional.cross_entropy accept `(batch,classes)` logits with integer Python/NumPy labels `(batch,)`. Only mean reduction, class-index targets and unweighted loss are supported. Floating labels, ignore_index, label smoothing and other reductions are rejected.

SGD supports required lr, momentum and coupled weight_decay. Adam supports lr, betas, eps and coupled weight_decay, with per-parameter bias-corrected moments. No parameter groups, closure, Nesterov/dampening, AMSGrad, AdamW or differentiable step. Unsupported kwargs fail. step preserves gradients; grad=None skips updates/state, while zero gradients can still apply momentum/decay. Hyperparameters must be finite and fit float32. Training/model updates are intended for one Python thread.

## Graph lifetime and resume

Leaf gradients accumulate until zero_grad resets None; non-leaf grad remains None. backward needs an exact-shape detached Tensor seed unless output contains one element. Default backward consumes saved recipes. Values remain readable; repeated backward through a consumed path fails. Use retain_graph=True before repeated calls.

Updating/loading a parameter invalidates saved snapshots. Backward checks every saved operand, including frozen transformed weights, before changing leaf gradients. Failed seed/version/consumption preflight leaves existing gradients unchanged.

Model state_dict contains detached Tensor copies. Optimizer state_dict uses this package's own format: version1, kind, hyperparameters and ordered state records with step and copied momentum/moment Tensors. load_state_dict validates/stages state before replacement; it is not a PyTorch optimizer state-file importer. Save/load model and optimizer state together to resume the same trajectory. File checkpoints use safetensors, as shown below.

## Inference mode

```python
from densecore import no_grad, cpu_backend
model.eval()
with no_grad():
    predictions = model(x).numpy()
print(cpu_backend())  # accelerate on macOS, reference on Linux
```

no_grad/enable_grad are thread-local synchronous context managers and decorators. Nesting, reuse and exception restoration are supported. Async/generator decorators are rejected; do not span await/task boundaries with these contexts. Explicit requires_grad=True factories remain requiring gradients. Tensor.detach returns an independent copy. Inference results inside no_grad remain stable after parameter updates.

## Optional compilation

```python
import densecore
model = densecore.compile(model)  # Returns the same module; state names stay unchanged.
# Or model.compile(), which enables compilation in place and returns None.
loss = loss_fn(model(x), labels)
loss.backward()
optimizer.step()
```

Compilation supports exact built-in Linear, ReLU and nested Sequential modules. Other modules, Tensor subclasses, replaced Python methods and unsupported call signatures run eagerly. Native runtime, shape and version errors propagate. Module.eval() does not disable gradients; use no_grad for inference buffer reuse.

Shapes and requested gradients specialize lazily, with four native plans cached per captured structure. Child/parameter replacements rebuild the capture; current parameter values reflect optimizer steps and state loads. Every live training output owns its saved activations, so overlapping and retained graphs remain valid. Native compiled backward contributes gradients to ordinary eager autograd, including upstream input graphs and shared parameters.

This initial Python API adds capture guards and allocates a training workspace per live forward. It is slower than Densecore eager on the measured small MLP. Compilation is optional while these costs are reduced; it is not a blanket speed optimization. Complete native training/workspace reuse also exists internally. See docs/AUTONOMOUS_PROGRESS.md for measured paths and limitations.

## Weights and checkpoints

```python
from densecore import checkpoints
checkpoints.save_weights("weights.safetensors", model)
checkpoints.load_weights("weights.safetensors", model)
checkpoints.save_checkpoint("training.safetensors", model, optimizer)
checkpoints.load_checkpoint("training.safetensors", model, optimizer)
```

Weights use exact parameter names, shapes and float32 dtype. Architecture is supplied separately. External safetensors weights are accepted when they match the chosen model. Shared aliases must contain matching values. Checkpoint files include named weights, optimizer buffers and versioned JSON metadata in one file. Optimizer must cover all registered parameters in registration order. Malformed metadata/name/shape/state checks occur before mutation. Successful saves atomically replace the destination; failed serialization preserves it.

Convert a trusted float32 PyTorch tensor state_dict with the optional helper:

```python
checkpoints.convert_torch_weights("source.pt", "weights.safetensors", model=model)
```

Conversion lazily imports PyTorch and uses CPU weights-only loading. A wrapper containing state_dict is accepted. An optional key_map must cover every source name and map to unique destination names. Target model validation is optional; weights from unsupported architectures do not become executable by converting their file format. Non-float32/sparse values are rejected. Arbitrary pickled Module objects and PyTorch optimizer-state conversion are unsupported.

## Build and validation

```sh
python -m pip install -r requirements-dev.txt
maturin develop --release
cargo test --release
cargo clippy --release --all-targets -- -D warnings
python -m pytest tests
python examples/train_dense.py
maturin build --release --out dist
```

PyO3 is pinned and Cargo.lock records native dependencies. PyTorch2.13 is a development oracle only. Oracle tests run with no skips during validation; non-oracle tests run on the installed wheel without PyTorch. See STAGE3.md for evidence.

Matrix multiplication uses macOS system Accelerate and a portable reference kernel on Linux. Broadcasting still allocates index maps and NumPy transfers use Python lists. The Stage5 benchmark measures about5.2x lower peak process RSS and11x faster imports than an optimized eager PyTorch control for the declared small dense workload, while hot inference and batch32 training are slower. See benchmarks/STAGE5_RESULTS.md for all cases and limits. These Stage5 results describe version0.5. Version0.7 adds compiled forward/backward and optional buffer pooling; current public benchmarks are in benchmarks/COMPILED_RESULTS.md. Optimized Linux kernels and GPU backends remain roadmap work.

## Layer normalization and compiled extensions

`nn.LayerNorm(features)` and `functional.layer_norm` support rank-2 inputs, optional affine parameters and first-order gradients. Tensor storage is float32; centered statistics and input-gradient calculations use float64 internally to reduce cancellation. Ordinary float32 parity and large-offset float64-reference checks use separate explicit tolerances. This is numerical equivalence within tolerances, not bitwise PyTorch equivalence.

Native graph execution supports rank-2 matmul and whole-tensor sum/mean. Public guarded module capture includes Linear, ReLU, LayerNorm and Sequential. Models outside this capture subset fall back to eager execution. Compiled execution remains opt-in: it did not beat eager Densecore on the real classifier.

Local release artifacts and validation: [release checklist](docs/LOCAL_RELEASE.md). MIT covers Densecore source; bundled dependency licenses and dataset attribution remain separate.

## Opt-in fused compiled training

```python
import densecore as dc
model = dc.compile(dc.nn.Sequential(dc.nn.Linear(64, 128), dc.nn.ReLU(), dc.nn.Linear(128, 32)))
optimizer = dc.optim.Adam(model.parameters(), lr=0.001)
x = dc.tensor([[0.1] * 64])
target = dc.tensor([[0.0] * 32])
loss = dc.compiled_train_step(model, x, target, optimizer)
```

This call performs forward, mean MSE, backward and one optimizer update. Loss is detached. Do not call `loss.backward()` or `optimizer.step()` again for that batch. Native optimizer state and Python Parameter identity survive successive steps. Existing eager `.grad` values remain unchanged; fused gradients are private to the step. Ordinary zero_grad/forward/backward/step behavior is unchanged.

Requires a compiled built-in Linear/ReLU/Sequential model, exact trainable Parameters, detached float32 CPU inputs/targets and built-in SGD/Adam with the same parameter identities and order. Shared parameters use one deduplicated graph slot. Unsupported capture, LayerNorm, frozen parameters, differentiable inputs/targets and disabled gradient mode raise rather than fall back. Native parameter versions still reject stale graphs.

Concurrent fused calls serialize through handle publication. Concurrent ordinary model or optimizer mutation remains unsupported. Native publication occurs under the held GIL; free-threaded Python is not supported. Fatal allocation failures or process interruption cannot guarantee recoverability. Benchmark protocol and measured scope: benchmarks/NATIVE_TRAINING_RESULTS.md.
