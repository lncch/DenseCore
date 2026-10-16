# DenseCore

DenseCore is an MIT-licensed deep-learning library: a Rust engine with a PyTorch-shaped Python API, for CPU training and inference of small models.

**Its results match PyTorch bit for bit.** Point `import torch` at DenseCore, run Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) unchanged, and from `torch.manual_seed(1337)` you get PyTorch's exact initial weights, loss, gradients and weights at every training step, with or without dropout. Not "within 1e-5": the same bits.

```sh
PYTHONPATH=python python scripts/demo_bit_exact.py      # needs PyTorch 2.14.1 to compare against
...
step 49: loss 2.6270534992218018 vs 2.6270534992218018 lossdiff=0 norm 0 grads_off=0 params_off=0
all 50 steps bit-identical
```

How it gets there, and where it stops: [the write-up](docs/WRITEUP_DRAFT.md).

| Measured on an Apple-silicon Mac, 2026-10-17 | DenseCore | PyTorch 2.14.1 |
|---|---|---|
| Installed size | 2.3 MB (wheel 0.9 MB; plus NumPy, safetensors) | 583 MB |
| `import` time (warm) | ~0.09 s | ~0.8-1.1 s |
| Same bits as PyTorch | nanoGPT training; 420 of 676 OpInfo ops | |

Limits: CPU only, float32 math, exact against PyTorch 2.14.1 on Apple-silicon macOS (PyTorch itself gives different bits on Linux and x86). DenseCore is slower than PyTorch, roughly 1.5x to 3x on the nanoGPT step.

Built by Almoayyad Abuljdail with heavy use of AI coding tools (OpenAI Codex and Anthropic Claude Code).

Version **0.10.0 is a local release candidate**. No public package upload or GitHub release has occurred. Build from this repository or install a matching local wheel.

## Features

- Autograd, `nn` modules (Linear, Embedding, LayerNorm, GELU, Dropout, ModuleList/ModuleDict, ...), attention, SGD/Adam/AdamW with parameter groups, gradient clipping
- PyTorch-named API: `torch.ones/arange/from_numpy/topk/multinomial/save/load`, `x.view/split/to/item`, list and `None` indexing, mask assignment ([full table](docs/COMPATIBILITY.md))
- Ten dtypes (float32, float64, float16, bfloat16, int8/16/32/64, uint8, bool); conversions match PyTorch bit for bit, and math with gradients is float32
- Rust-owned tensors with version-checked gradients, safetensors checkpoints and pickle-free `save`/`load`
- Guarded module compilation and fused training steps for supported models
- Bit-identical training results across thread counts on one machine ([probe](benchmarks/DETERMINISM_RESULTS.md))
- PyTorch's exact numerics on CPU float32: reductions, vector math, BLAS calls, flash attention, optimizers, LR schedulers, the random generator, nn init, dropout and `torch.utils.data` ([roadmap](docs/ROADMAP.md))

## Install DenseCore

Run commands from the repository root. Runtime requirements are Python 3.11 or newer, NumPy `>=2,<3` and safetensors `>=0.8,<0.9`. PyTorch is optional and used for development comparisons and weight conversion.

### Build from source

Install Rust 1.88 or newer and a platform linker before building. macOS builds require Xcode Command Line Tools; Linux builds require C build tools. Validation used Rust 1.99.0; the minimum compiler version comes from `Cargo.toml`.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install "maturin>=1.15,<2" "numpy>=2,<3" "safetensors>=0.8,<0.9"
maturin develop --release
```

### Install a local wheel

A prebuilt wheel requires neither Rust nor PyTorch. For the tested macOS ARM64/Python 3.13 environment, activate a Python 3.13 virtual environment and run:

```sh
python -m pip install dist/densecore-0.10.0-cp313-cp313-macosx_11_0_arm64.whl
```

Select a wheel matching your Python version, operating system and architecture. Local wheels are ignored build outputs; a fresh source checkout does not contain them. Existing `dist/` wheels are historical runtime-validation builds with earlier README metadata. Current audited candidate artifacts use a separate directory and [readiness manifest](docs/RELEASE_READINESS_ARTIFACTS.json). Linux wheel filenames and SHA256 hashes appear in the [0.10 artifact manifest](docs/NATIVE_TRAINING_RELEASE_ARTIFACTS.json).

Version 0.10 installed-package validation:

| Platform | Python | Evidence |
|---|---|---|
| macOS ARM64 | 3.13 | 187 tests in a clean environment without PyTorch |
| Linux ARM64, glibc 2.36 | 3.11.17 | 187 installed-wheel tests, 53 Rust tests and release clippy |
| Linux x86-64, glibc 2.36, local emulation | 3.13.16 | Same checks; installation/correctness evidence, not native x86 speed |

Linux wheels use a `manylinux_2_34` tag. Other platform/Python combinations, including Python 3.12, remain unverified for this release. See the [engineering report](docs/NATIVE_TRAINING_ENGINEERING_REPORT.md) for details.

## Train a classifier

This standalone example trains a two-class model. Labels are integer metadata, not DenseCore tensors.

```python
import densecore as dc

dc.manual_seed(7)
model = dc.nn.Sequential(dc.nn.Linear(2, 8), dc.nn.ReLU(), dc.nn.Linear(8, 2))
optimizer = dc.optim.Adam(model.parameters(), lr=0.01)
x = dc.tensor([[-2, -1], [1, 2]])
labels = [0, 1]
for _ in range(100):
    optimizer.zero_grad()
    loss = dc.functional.cross_entropy(model(x), labels)
    loss.backward()
    optimizer.step()
model.eval()
with dc.no_grad():
    predictions = model(x).numpy().argmax(axis=1).tolist()
assert predictions == labels
print(predictions)
```

Expected output: `[0, 1]`. `eval()` changes module mode; it does not disable gradient recording. Use `no_grad()` for inference. Gradient contexts support nesting and exception restoration. Explicit `requires_grad=True` factories remain requiring gradients inside `no_grad()`; do not span async/task boundaries with gradient contexts.

## Use tensors and gradients

Tensor construction, `numpy()` and state dictionaries copy values. Gradients and detached tensors may share immutable native storage. NumPy arrays cannot mutate native storage.

Unreleased storage work adds zero-copy reshape for contiguous inputs and transpose/permute views. Existing kernels materialize strided inputs into logical row-major values when needed. See [storage semantics and limitations](docs/TENSOR_STORAGE.md) and [compatibility matrix](docs/COMPATIBILITY.md). Registered eager math kernels are described in [unary dispatch](docs/OPERATOR_DISPATCH.md). Numerical gradient validation is available through [`densecore.autograd.gradcheck`](docs/GRADCHECK.md). Package version metadata remains unchanged.

```python
import numpy as np
import densecore as dc

x = dc.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
bias = dc.tensor([1.0, 2.0])
loss = ((x + bias) * 2.0).mean()
loss.backward()
np.testing.assert_array_equal(x.grad.numpy(), np.full((2, 2), 0.5))
print(float(loss.numpy()))
```

Expected loss: `8.0`. Leaf gradients accumulate until `zero_grad()` resets them to `None`; non-leaf `.grad` remains `None`. Default `backward()` consumes saved recipes. Set `retain_graph=True` before repeated backward calls. Outputs whose element count is not one need an exact-shape detached Tensor seed. Failed seed/version/consumption checks preserve existing leaf gradients.

Updating or loading a parameter invalidates older saved graphs. Tensor values remain readable after their backward graph is consumed. Higher-order gradients are unsupported.

## Compile supported modules

`dc.compile(model)` returns the same module and preserves parameter names. `model.compile()` enables compilation in place and returns `None`.

```python
import numpy as np
import densecore as dc

dc.manual_seed(11)
model = dc.compile(dc.nn.Sequential(
    dc.nn.Linear(3, 8), dc.nn.LayerNorm(8), dc.nn.ReLU(), dc.nn.Linear(8, 1)
))
optimizer = dc.optim.SGD(model.parameters(), lr=0.01)
x = dc.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
target = dc.tensor([[0.0], [1.0]])
for _ in range(20):
    optimizer.zero_grad()
    loss = dc.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
assert np.isfinite(loss.numpy()).all()
print("compiled training completed")
```

Public capture supports exact built-in `Linear`, `LayerNorm`, `ReLU` and nested `Sequential` modules. It does not trace arbitrary Python or custom tensor expressions. Unsupported models, patched methods, Tensor subclasses and unsupported call signatures fall back to eager execution. Native shape, version and runtime errors propagate.

Shapes and gradient masks specialize lazily, with up to four cached native plans per captured structure. Structural/parameter replacements and mode changes invalidate signature metadata. Current native handles reflect optimizer steps and state loads. Each live ordinary compiled output owns its saved activations, supporting overlapping or retained graphs and upstream eager autograd.

Compilation is optional. Ordinary compiled execution remains slower than eager DenseCore on the measured small dense workloads. `no_grad()` inference reuses buffers; gradient-enabled ordinary forwards use separate workspaces.

## Fuse a complete MSE training step

`compiled_train_step` is a separate opt-in API. It returns a detached scalar after updating parameters once.

```python
import numpy as np
import densecore as dc

dc.manual_seed(13)
model = dc.compile(dc.nn.Sequential(
    dc.nn.Linear(2, 8), dc.nn.ReLU(), dc.nn.Linear(8, 1)
))
optimizer = dc.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
x = dc.tensor([[1.0, 0.0], [0.0, 1.0]])
target = dc.tensor([[0.0], [1.0]])
for _ in range(50):
    loss = dc.compiled_train_step(model, x, target, optimizer)
assert not loss.requires_grad and np.isfinite(loss.numpy()).all()
print("fused training completed")
```

Do not call `loss.backward()` or `optimizer.step()` again for that batch. Fused gradients remain private; existing eager `.grad` values are neither cleared nor accumulated. Model Parameter objects and native optimizer state survive successive steps.

Fused training requires a compiled built-in `Linear`/`ReLU`/`Sequential` model, exact trainable Parameters, detached float32 CPU inputs/targets, gradients enabled, and built-in SGD/Adam covering parameters in matching identity/order. Shared Parameter objects use one slot. LayerNorm, frozen parameters, custom dispatch, differentiable inputs/targets and unsupported aliases raise rather than fall back.

Native execution validates and stages optimizer updates, allocates Python publication objects before commit, then publishes current handles under the held global interpreter lock (GIL). Fused callers serialize through publication. Concurrent ordinary model/optimizer mutation and free-threaded Python remain unsupported. Fatal allocation failures or process interruption cannot guarantee recovery.

Only detached fused steps reuse a full training workspace. Up to four shape/gradient specializations retain buffers; ordinary live graphs keep independent workspaces.

## Save weights and resume training

Safetensors files store float32 weights with exact names/shapes. Supply architecture separately. Full checkpoints also store versioned optimizer state; the optimizer must cover registered parameters in registration order.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import densecore as dc

dc.manual_seed(17)
model = dc.compile(dc.nn.Linear(2, 1))
optimizer = dc.optim.Adam(model.parameters())
x, target = dc.tensor([[1.0, 2.0]]), dc.tensor([[1.0]])
dc.compiled_train_step(model, x, target, optimizer)
with dc.no_grad():
    expected = model(x).numpy()
with TemporaryDirectory() as directory:
    weights = Path(directory) / "weights.safetensors"
    checkpoint = Path(directory) / "training.safetensors"
    dc.checkpoints.save_weights(weights, model)
    dc.checkpoints.load_weights(weights, model)
    dc.checkpoints.save_checkpoint(checkpoint, model, optimizer)
    dc.compiled_train_step(model, x, target, optimizer)
    dc.checkpoints.load_checkpoint(checkpoint, model, optimizer)
with dc.no_grad():
    np.testing.assert_array_equal(model(x).numpy(), expected)
print("checkpoint restored")
```

State loading validates names, shapes and shared-alias values before mutation, preserving Parameter identity. Checkpoint loading validates model and optimizer state before replacement. Successful file saves atomically replace destinations. Optimizer dictionaries use DenseCore's format, not PyTorch optimizer state files.

### Convert a trusted PyTorch state dictionary

The optional helper imports PyTorch lazily and loads CPU tensors with `weights_only=True`. This standalone example creates its own trusted source file; it requires the development dependencies.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
import densecore as dc

torch.manual_seed(7)
source_model = torch.nn.Linear(2, 1)
model = dc.nn.Linear(2, 1)
with TemporaryDirectory() as directory:
    source = Path(directory) / "source.pt"
    destination = Path(directory) / "weights.safetensors"
    torch.save(source_model.state_dict(), source)
    dc.checkpoints.convert_torch_weights(source, destination, model=model)
    dc.checkpoints.load_weights(destination, model)
with torch.no_grad(), dc.no_grad():
    expected = source_model(torch.tensor([[1.0, 2.0]])).numpy()
    np.testing.assert_allclose(model(dc.tensor([[1.0, 2.0]])).numpy(),
                               expected, rtol=1e-5, atol=1e-6)
print("weights converted")
```

An optional `key_map` maps every source name to a unique destination name. Conversion accepts float32 dense tensor dictionaries, optionally wrapped in `state_dict`. Arbitrary pickled Modules, sparse/non-float32 weights and PyTorch optimizer conversion are unsupported. Converting weights does not add support for their architecture.

## Run the pretrained letter classifier

The repository includes a safetensors checkpoint and preprocessing manifest. It needs neither PyTorch nor a dataset download for inference.

```sh
python examples/predict_letter.py 2 8 3 5 1 8 13 0 6 6 10 8 0 8 0 8
```

Expected class: `T`. Inputs are the 16 UCI Letter Recognition features, not images or arbitrary text. The duplicate-group-safe heldout evaluation recorded **95.8457%** DenseCore accuracy versus **96.0930%** for matched PyTorch training. Test data did not select checkpoints. See [classifier protocol/results](benchmarks/LETTER_RESULTS.md) and [dataset attribution](docs/DATASET_ATTRIBUTION.md). Dataset licensing is separate from the MIT source license.

## Supported Python operations

Floating-point math uses CPU float32 storage. Explicit `dtype="bool"` tensors have native boolean storage for views and logical operations; boolean autograd, reductions, neural-network math and compiler capture are unsupported. LayerNorm uses float64 centered arithmetic internally for statistics and input gradients; equivalence uses explicit tolerances rather than bitwise equality.

| Surface | Supported behavior | Bounds |
|---|---|---|
| Tensor arithmetic | `+`, `*`, real scalars, rank-2 `@`/`matmul` | Equal shapes, rank-0 scalar, or `(features,)` with `(batch, features)` broadcasting only |
| Batched matrices | `dc.bmm(a, b)`, `a.bmm(b)` | Rank-3 operands, equal batches, no broadcasting; eager CPU float32 only |
| Joining tensors | `dc.cat(tensors, dim=0)`, `dc.stack(tensors, dim=0)` | Existing axis for cat, new axis for stack; eager CPU float32 only |
| Comparisons | `==`, `!=`, `<`, `<=`, `>`, `>=`; named `eq`/`ne`/`lt`/`le`/`gt`/`ge` methods/functions | Same-dtype float32 or bool inputs with broadcasting; real bool results; eager only; identity-based hashing |
| Boolean tensors | `tensor(data, dtype="bool")`, `logical_not()` / `~`, logical-and/or/xor (`&`, `|`, `^`), views/layout inspection, copying NumPy export | CPU only; nondifferentiable; no boolean indexing/reductions or dtype promotion |
| Tensor transforms | `reshape()` including `-1`, `transpose()`, `permute()`, `contiguous()`, layout inspection, axis-aware `sum()`/`mean()`, `detach()` | Views support float32/bool; reductions require float32; strided kernels materialize logical values; no writable storage |
| Autograd | `requires_grad`, `is_leaf`, copied `grad`, `backward`, `zero_grad` | First order; implicit seeds only for one-element outputs; exact-shape detached seeds otherwise |
| Modules | Registration, parameter enumeration, `train`/`eval`, state dictionaries; Linear/ReLU/Sequential/LayerNorm | Linear and feature LayerNorm require rank-2 batches; no buffer registration or module forward/backward hooks |
| Losses | Functional and module mean MSE/cross-entropy | Equal-shape MSE; integer class-index metadata for `(batch, classes)` logits |
| Optimizers | SGD: `lr`, momentum, coupled weight decay; Adam: `lr`, betas, eps, coupled weight decay | No parameter groups, closures, Nesterov, AMSGrad or AdamW |
| Gradient mode | `no_grad`, `enable_grad`, `is_grad_enabled` | Thread-local synchronous contexts/decorators; no async/generator decorators |

`grad=None` skips optimizer updates/state; a zero gradient can still apply momentum or weight decay. Optimizer hyperparameters must satisfy finite float32-range validation. Linear initialization uses seeded uniform weights/bias within `±1/sqrt(in_features)`.

Empty sum returns zero; empty mean returns NaN; zero-width matmul produces zeros. NumPy array arithmetic is rejected: convert arrays explicitly with `dc.tensor`. Modules in ordinary Python lists/dictionaries are not automatically registered. Shared parameters deduplicate during parameter enumeration while all alias names appear in model state dictionaries.

Batched matrix multiplication preserves gradients for both operands:

```python
import densecore as dc

a = dc.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
b = dc.tensor([[[2.], [1.]]], requires_grad=True)
y = dc.bmm(a, b)
assert y.numpy().tolist() == [[[4.], [10.]]]
y.sum().backward()
assert a.grad.numpy().tolist() == [[[2., 1.], [2., 1.]]]
```

`bmm` accepts `(batch, rows, inner)` and `(batch, inner, columns)`. It does not broadcast batches or extend compiler capture. Zero inner width produces zeros; empty batches and output dimensions remain differentiable.

Join tensors while preserving gradients:

```python
import densecore as dc

x = dc.tensor([[1., 2.]], requires_grad=True)
joined = dc.cat([x, x], dim=0)
assert joined.shape == (2, 2)
stacked = dc.stack([x, x], dim=-1)
assert stacked.shape == (1, 2, 2)
joined.sum().backward()
assert x.grad.numpy().tolist() == [[2., 2.]]
```

`cat` requires equal ranks and equal dimensions outside the joining axis; scalar inputs are rejected. `stack` requires identical shapes and supports scalars. Both require a nonempty sequence and accept negative axes. Joining copies logical input values. Concatenation gradients use immutable slices of the upstream buffer, so a small gradient can retain that whole buffer. These operations do not extend compiler capture.

Boolean tensors use separate typed storage and share the checked view rules:

```python
import densecore as dc

mask = dc.tensor([[True, False], [False, True]], dtype="bool")
view = mask.T
assert view.shares_storage(mask)
assert (~view).numpy().tolist() == [[False, True], [True, False]]
assert mask.dtype == "bool" and mask.grad is None
assert bool(dc.tensor([False], dtype="bool")) is False
```

Boolean construction uses numeric nonzero truth conversion, including NaN, and rejects strings/objects. Default construction remains float32. Boolean tensors cannot require gradients or initialize Parameters. Empty and multi-element tensors of either dtype raise on Python truth testing. Unsupported boolean arithmetic raises instead of casting. See [boolean storage validation](docs/BOOLEAN_STORAGE_VALIDATION.md).

Named comparisons and logical methods build nondifferentiable masks:

```python
import densecore as dc

x = dc.tensor([[-1., 0., 2.]], requires_grad=True)
positive = x > 0
bounded = positive & (x < 3)
assert bounded.dtype == "bool" and not bounded.requires_grad
assert bounded.numpy().tolist() == [[False, False, True]]
assert (bounded == True).numpy().tolist() == [[False, False, True]]
assert dc.logical_xor(bounded, True).numpy().tolist() == [[True, True, False]]
```

Use `x == y`, `x.eq(y)` or `dc.eq(x, y)` for elementwise equality. Comparisons return bool tensors, not Python booleans. Use `is` for object identity; hashes retain object identity. Distinct multi-element tensors in list membership/equality can raise ambiguous-truth errors. Float comparisons accept real scalars converted to float32, including float32 rounding. Bool comparisons require bool tensors or boolean scalars and order `False < True`. Logical-and/or/xor (`&`, `|`, `^`) also require bool operands; reflected Python/NumPy boolean scalars work. Parenthesize comparisons in mask expressions, such as `(x > 0) & (x < 3)`. No mixed tensor dtype promotion, boolean indexing or compiler capture is added. See [predicate validation](docs/PREDICATE_DISPATCH_VALIDATION.md), [comparison operator identity rules](docs/COMPARISON_OPERATOR_VALIDATION.md) and [boolean operator validation](docs/BOOLEAN_OPERATOR_VALIDATION.md).

Native graph IR additionally supports internal Add/Mul, matmul and whole-tensor reductions. These private operations do not expand public Python capture beyond the built-in module subset above.

Backward validates exact parent-gradient shapes and stages eager leaf updates before publication. A returned derivative/accumulation error preserves existing leaf gradients and does not consume the graph. Custom first-order Python functions use resumable backward with callbacks outside native locks. Gradient hooks use the same resumable boundary; see [custom function semantics](docs/CUSTOM_FUNCTION.md) and [backward transaction design](docs/BACKWARD_TRANSACTION_DESIGN.md).

## Architecture

The Python layer handles module registration, guarded capture and public API validation. PyO3 binds immutable Rust tensor snapshots and versioned handles to Python. Rust schedules operations and first-order derivatives, stages optimizer state and invokes CPU kernels.

Compiled plans store checked shape/device/dtype specifications for every node. Specialization keys include those specifications and gradient flags; strides remain outside the key because kernels consume logical values. Lowering and workspaces currently support CPU float32 only. This metadata foundation does not add another backend or dtype. See [compiler metadata validation](docs/COMPILER_METADATA_VALIDATION.md).

```text
Python Tensor / Module / optim / checkpoints
                     |
                  PyO3 binding
                     |
       Rust tensor values + versions + autograd
                     |
          checked float32 CPU kernels
          /                         \
macOS Accelerate GEMM       Linux reference GEMM

compile(model) -> guarded capture -> shape/gradient plan
                                      |
                 ordinary: workspace per live output
                 fused: locked reusable workspace
```

NumPy transfers currently pass through Python lists and copy storage. Python guards and boundary crossings remain measured costs. Linux optimized math kernels and GPU execution are roadmap work.

## Measured performance and reproduction

The final fused experiment used an Apple M4, float32 CPU, Accelerate matrix multiplication, a `64 → 128 → ReLU → 32` model and mean MSE. All engines used matching initial arrays, data and update counts. PyTorch 2.14.1 controls used foreach optimizers. Requested thread environment and PyTorch intra/inter-op settings were one.

Five fresh processes per engine/cell ran five blocks of 300 complete steps after 50 warmups, with alternating engine order. Thirty pre-timing gates compared twelve losses and parameters/optimizer state after step 12 at `rtol=1e-4`, `atol=1e-6`. All 150 timed processes passed parity and final-loss/hash gates. Setup/import/first-call costs are outside steady-state timing.

Median microseconds per complete training step:

| Optimizer / batch | DenseCore eager | Ordinary compiled | Fused cached | Fused uncached | PyTorch foreach |
|---|---:|---:|---:|---:|---:|
| SGD / 1 | 17.20 | 23.36 | 18.69 | 20.93 | 62.43 |
| SGD / 32 | 21.70 | 27.04 | 23.38 | 26.81 | 65.17 |
| Momentum SGD / 1 | 19.52 | 25.11 | 20.62 | 23.08 | 71.88 |
| Momentum SGD / 32 | 23.45 | 28.28 | 25.61 | 28.28 | 74.02 |
| Adam / 1 | 23.30 | 28.25 | 24.89 | 26.35 | 98.26 |
| Adam / 32 | 29.86 | 35.44 | 31.09 | 32.69 | 121.94 |

Cached fused medians are approximately 5–15% faster than same-build uncached fused and 10–25% faster than ordinary compiled. Eager remains faster in all six median comparisons. Adam has notable process variation; inspect the recorded min/max and block samples before treating a median as expected latency.

Cached DenseCore peak process resident set size (RSS) was 37.98–39.22 MiB, comparable to uncached 38.02–39.33 MiB. PyTorch controls used roughly 282–283 MiB. RSS includes imports, setup, validation and timing; it does not measure tensor-only memory or energy cost. Results apply to this Mac workload, not all models, Linux performance or GPUs.

The [raw final experiment](benchmarks/native-training-workspace.json), [complete protocol/results](benchmarks/NATIVE_TRAINING_RESULTS.md) and [source/build provenance](docs/NATIVE_TRAINING_BENCHMARK_PROVENANCE.json) preserve variation and earlier unsuccessful attempts. Timings used the completed runtime implementation before the metadata-only 0.10 version bump. Historical [0.7 compilation results](benchmarks/COMPILED_RESULTS.md) and [0.9 classifier results](benchmarks/LETTER_RESULTS.md) remain separate workloads.

After installing development dependencies, reproduce the final five-engine benchmark:

```sh
python benchmarks/benchmark_native_training.py --output benchmarks/native-training-local.json
```

The command creates 150 timed processes plus pre-timing validation processes. Choose a new output filename for subsequent runs; it refuses overwrite. Keep the machine quiet during timing. Numerical gates fail the run before accepting results.

## Current CPU measurement checkpoint

A current Apple M4, one-thread float32 MLP sweep validates identical weights/data
and reports three fresh processes per configuration. DenseCore eager is faster
in the measured batch-1/32 training cases, but loses at batch 1024. DenseCore
compilation also loses against its own eager path in several cases. Whole-process
RSS includes imports and validation overhead; it is not isolated tensor memory.
See [current results, ranges and limitations](benchmarks/CURRENT_CPU_CHECKPOINT.md)
and [selected timing/provenance records](benchmarks/current-cpu-2026-10-15-summary.json).
A [larger 525,568-parameter MLP sweep](benchmarks/LARGE_MODEL_CHECKPOINT.md)
also retains substantial compiled-training regressions and mixed eager/control
results. A later [private gradient-transfer study](docs/COMPILED_GRADIENT_TRANSFER.md)
proves copy removal but finds variable latency, including a regression. Recorded
results describe their identified revisions. These measurements do not establish
general superiority over PyTorch.

A [requested-thread sweep](benchmarks/THREAD_SCALING_RESULTS.md) covers 1/2/4
settings on small and larger MLPs. Small-model training wins remain local to that
workload; larger batch-32 training loses to both Torch controls at every setting.
All numerical gates pass. DenseCore active worker counts are not measured.
The [workspace payload study](docs/WORKSPACE_MEMORY_VALIDATION.md) counts native
float32 capacities directly. Large-model batch-1 tracked payload reaches 10.55 MB
across two live workspaces, borrowed inputs/parameters, output tensors and transferred
VJP gradients after retry. It excludes leaf `.grad` accumulation, allocator overhead
and RSS; it is not total process memory. Separate inference tests record 75% and
66.7% lower workspace capacity than dedicated-per-instruction buffers on an
eight-ReLU chain and a branched chain. These are two measured graph schedules, not a
whole-process memory or speed claim. A separate duplicate-parent test measures
6.31 MB across the existing, incoming and merged gradient buffers for 525,568
elements; it excludes other backward temporaries and is not a full peak-memory
measurement. A four-branch graph records 8.41 MB across unique pending-gradient
storage at its largest tested merge boundary, with seven contributions to one leaf.
On that graph, tracked TensorData backing capacity peaks at 29.43 MB during backward,
up from 16.82 MB after forward. Matmul and LayerNorm cases record 114,692 B and
67,588 B peaks; broadcast and stride-2 slice cases record 66,564 B and 81,924 B.
These values apply to tested shapes. The high-water mark excludes allocator
overhead, metadata and RSS, so it does not claim total process peak memory. Native
hook and custom-VJP session probes peak at 114,688 B and 49,152 B; these exclude
Python callback heap overhead.

A separate [startup and dependency-payload study](benchmarks/STARTUP_PAYLOAD_RESULTS.md)
records median process-to-first-forward times of 229 ms for DenseCore and 1,489 ms
for Torch on this Apple M4. Recorded installed payload including declared runtime
dependencies and NumPy is 35.3 MB versus 711.8 MB. These are a tiny fixed Linear
workload with warm OS caches and logical installed-file sizes, not steady-state
throughput, total environment size or RAM. The study preserves all samples and
states environment and capability differences.

## Gradient hooks

CPU float32 tensors requiring gradients support `register_hook(callback)`, with
ordered callbacks, removable handles and exact-shape detached replacements.
Late registration on existing ancestors is supported. See [hook semantics and
limits](docs/TENSOR_HOOKS.md). Fused training rejects active parameter hooks.

## Singleton dimensions

`Tensor.unsqueeze(dim)` inserts a length-one axis. `Tensor.squeeze()` removes all
length-one axes; `squeeze(dim)` selects one integer axis. Negative axes are
supported. Squeezing a non-singleton axis returns the original tensor. Bool dtype
is preserved. These methods compose reshape: contiguous inputs share storage;
strided inputs may copy. Tuple squeeze axes are currently unsupported.

```python
import numpy as np
import densecore as dc

x = dc.tensor([[1., 2.]], requires_grad=True)
y = x.squeeze(0).unsqueeze(-1)
assert y.shape == (2, 1)
y.sum().backward()
np.testing.assert_array_equal(x.grad.numpy(), [[1., 1.]])
```

## Flattening dense-network inputs

`Tensor.flatten(start_dim=0, end_dim=-1)` collapses an inclusive dimension range.
`nn.Flatten()` defaults to dimensions 1 through the last, preserving batch size.
Contiguous inputs share storage; strided inputs follow reshape copying behavior.
A single-axis flatten returns the same non-scalar tensor. Scalars flatten to `(1,)`.
Models containing `nn.Flatten` currently use compiler eager fallback.

```python
import numpy as np
import densecore as dc

model = dc.nn.Sequential(dc.nn.Flatten(), dc.nn.Linear(6, 2))
x = dc.tensor(np.ones((4, 2, 3), dtype=np.float32), requires_grad=True)
y = model(x)
assert y.shape == (4, 2)
y.sum().backward()
assert x.grad.shape == (4, 2, 3)
```

## Sigmoid and tanh

Eager CPU float32 tensors support `sigmoid()` and `tanh()`, including strided
inputs and first-order gradients. Sigmoid avoids overflowing the exponential by
choosing its formula based on the input sign. Float32 saturation can produce
exact zero gradients. Infinities saturate; NaNs propagate. Matching `nn.Sigmoid` / `nn.Tanh` modules and `functional.sigmoid` /
`functional.tanh` entry points use the same kernels. Modules have no parameters.
Models containing these modules currently use eager compiler fallback.

```python
import numpy as np
import densecore as dc

x = dc.tensor([0., 1.], requires_grad=True)
dc.nn.Sigmoid()(x).sum().backward()
np.testing.assert_allclose(x.grad.numpy(), [.25, .19661193], rtol=1e-6)
np.testing.assert_allclose(dc.functional.tanh(dc.tensor([0., 1.])).numpy(), [0., .7615942], rtol=1e-6)
```

## GELU

`gelu(approximate="none")` computes exact GELU, `x * Phi(x)`, as `0.5 * x * erfc(-x / sqrt(2))`
in float64 before rounding to float32. The `"tanh"` form uses `x * sigmoid(2u)`, which equals
the usual `0.5 * x * (1 + tanh(u))` but keeps accuracy in the negative tail. Both have
first-order gradients. `nn.GELU(approximate)` and `functional.gelu` use the same kernels.
`-inf` gives NaN, matching PyTorch. Models with GELU use eager compiler fallback, and fused
`compiled_train_step` rejects them before changing any parameter.

```python
import numpy as np
import densecore as dc

x = dc.tensor([-1., 0., 1.], requires_grad=True)
dc.nn.GELU()(x).sum().backward()
np.testing.assert_allclose(x.grad.numpy(), [-.08331547, .5, 1.0833155], rtol=1e-6)
```

## Embedding

`nn.Embedding(num_embeddings, embedding_dim)` and `functional.embedding(indices, weight)`
gather weight rows. Indices are integer Python/NumPy metadata of any shape, like
cross-entropy labels; there is no integer Tensor dtype. The output shape is
`indices.shape + (embedding_dim,)`. Backward scatter-adds into the used rows, so repeated
indices accumulate. The weight starts from a standard normal, as in PyTorch.
`padding_idx`, `max_norm` and sparse gradients are not supported.

```python
import numpy as np
import densecore as dc

emb = dc.nn.Embedding(4, 2)
out = emb(np.array([[1, 3], [1, 1]]))
assert out.shape == (2, 2, 2)
out.sum().backward()
np.testing.assert_array_equal(emb.weight.grad.numpy()[1], [3., 3.])
```

## Rank>2 inputs for Linear, LayerNorm and matmul

`Linear`, `functional.linear`, `LayerNorm` and `functional.layer_norm` accept inputs of rank
above two. They flatten the leading dimensions, call the rank-2 kernel, and reshape back, so
`(batch, seq, features)` works as in PyTorch. `a @ b` with two rank-3 operands calls `bmm`, and
a rank>2 left operand with a rank-2 right operand flattens the same way. Gradients flow through
`reshape`. Compiled models fall back to eager for non-rank-2 inputs; the rank-2 path is unchanged.

## Dropout

`nn.Dropout(p)` and `functional.dropout(input, p, training)` zero elements with probability
`p` and scale the rest by `1 / (1 - p)`. Masks come from the same generator that
`manual_seed` seeds for weight initialization, as with PyTorch's global generator, so a seed
reproduces both. In eval mode or with `p=0` the input is returned unchanged. The mask is a
constant Tensor, so gradients use the same mask. Compiled models containing Dropout run eagerly.

## masked_fill

`Tensor.masked_fill(mask, value)` sets elements where `mask` is True to exactly `value`, even
when the input there is infinite or NaN, and gives them zero gradient. `mask` is a bool Tensor
or NumPy/Python bools broadcastable to the input. A causal attention mask is
`np.triu(np.ones((t, t), bool), 1)` filled with `-np.inf` before `softmax(-1)`.

## Scaled dot-product attention

`functional.scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=False,
scale=None)` takes rank-3 `(N, L, E)` or rank-4 `(N, H, L, E)` inputs. Mask meanings follow
PyTorch: a boolean mask is True where attention is allowed, and a float mask is added to the
scores. It is composed from `bmm`, `softmax` and `masked_fill`, so it has no fused kernel and
stores the full `L x S` score matrix. Rows with every position masked give NaN.

## Tiny GPT example

`examples/tiny_gpt.py` trains a 114k-parameter character-level GPT on this README, using
Embedding, rank-3 Linear/LayerNorm, causal attention, GELU, cross-entropy and Adam. `--parity`
checks its first Adam steps against an identical PyTorch model. On this workload DenseCore is
currently about 4.7x slower than PyTorch; see benchmarks/TINY_GPT_RESULTS.md.

## Subtraction and division

CPU float32 tensors support subtraction and true division with checked
broadcasting, scalar/reflected operands and first-order gradients. Strided inputs
read logical values. Division follows IEEE infinity/NaN behavior at zero; no
floor division, dtype promotion or in-place variant is provided.

```python
import numpy as np
import densecore as dc

x = dc.tensor([2., 4.], requires_grad=True)
y = (8. / x - 1.).sum()
y.backward()
np.testing.assert_allclose(x.grad.numpy(), [-2., -.5])
```

These operations are eager Python APIs. Public module compilation retains its
existing capture subset; it does not trace arbitrary arithmetic expressions.

## Limitations and roadmap

DenseCore currently targets dense CPU networks. It has no GPU backend, convolutions, attention, embedding module, distributed training, mixed precision or higher-order autograd. There is no general Python tracing, tensor slicing/indexing, arbitrary broadcasting, arbitrary pretrained-model loader or guarantee of PyTorch API/state-file compatibility.

Planned work, without promised release dates:

1. Profile remaining Python dispatch and optimizer-state copies against matched eager controls.
2. Extend fused training to LayerNorm and cross-entropy, with numerical parity and failed-step checks.
3. Optimize Linux math kernels and measure native Linux training/inference workloads.
4. Expand operations and supported pretrained architectures before adding GPU backends.

These items are roadmap goals, not implemented capabilities. Each addition needs correctness coverage and measured evidence before performance claims.

## Develop and contribute

Use a virtual environment and install the pinned development dependencies. PyTorch 2.14.1 is a test oracle, not a runtime dependency.

```sh
python -m pip install -r requirements-dev.txt
maturin develop --release
cargo fmt --all -- --check
cargo test --release
cargo clippy --release --all-targets -- -D warnings
python -m pytest tests -q
```

Initial 0.10 validation passed 288 Python tests. The release audit adds frozen-parameter regressions and passes 290 tests, including PyTorch parity/conversion tests, plus 53 release Rust tests. Six manual Rust benchmarks are ignored in the ordinary test suite. Installed-wheel tests run without PyTorch and exclude its three oracle-dependent modules.

Existing [GitHub Actions workflow](.github/workflows/wheels.yml) checks Rust formatting/release tests/clippy and installed-wheel Python tests on Ubuntu/macOS with Python 3.11/3.12/3.13. Its wheel jobs exclude PyTorch oracle modules; a separate oracle job runs the full suite with pinned development dependencies. Remote CI has not run; configured matrix coverage is distinct from tested local combinations. The workflow builds artifacts and does not publish packages.

For changes:

1. Describe the supported behavior and limitations before changing an API.
2. Add deterministic regressions for correctness changes; keep established parity tolerances and failure atomicity checks.
3. Run release Rust checks and the full Python oracle suite. Validate a clean installed wheel for packaging changes.
4. Benchmark performance changes against matched controls, preserving raw results and failed experiments. Report variation and memory semantics.
5. Keep commits focused, update documentation and retain dependency/dataset attribution. Do not replace unrelated experimental records.

Repository map:

| Path | Contents |
|---|---|
| `python/densecore/` | Public Python APIs, capture guards, fused runner and checkpoints |
| `src/` | Rust tensors/autograd, graph plans/workspaces, optimizers and CPU kernels |
| `tests/` | API, numerical parity, state, failure and concurrency regressions |
| `examples/`, `models/` | Training/inference scripts and pretrained letter checkpoint |
| `benchmarks/`, `docs/` | Reproducible controls, raw evidence, engineering reports and attribution |

## License and documentation history

DenseCore source uses the [MIT license](LICENSE). Bundled native dependency licenses appear in [third-party notices](THIRD_PARTY_NOTICES.md) and `THIRD_PARTY_LICENSES/`. The UCI dataset has separate [CC BY 4.0 attribution](docs/DATASET_ATTRIBUTION.md).

Historical stage reports `STAGE1.md` through `STAGE5.md`, earlier benchmark reports and the [archived previous README](docs/archive/README_BEFORE_DOCUMENTATION_MILESTONE.md) remain preserved. Current engineering decisions and verification details are in the [0.10 report](docs/NATIVE_TRAINING_ENGINEERING_REPORT.md). Initial README example/command execution evidence is recorded in [documentation verification](docs/README_VERIFICATION.json); its README hash identifies the historical documentation snapshot. Current example rechecks appear in the [release audit](docs/RELEASE_READINESS_AUDIT.json).

## Artifact policy

Git retains reviewed benchmark evidence and historical model checkpoints. New benchmark JSON/archives, profiles and source backups are ignored by default; promote evidence explicitly after checking its provenance, size and attribution. Never stage generated directories indiscriminately.

Source distributions exclude benchmark result data, profiles, benchmark checkpoints and backup files, while retaining benchmark scripts and Markdown reports. Full numerical evidence, artifact manifests and machine-specific audit records live in Git rather than the package archive. Runtime wheels contain the Python package/native extension and license payload, not datasets, repository examples or model files. Wheels include 28 dependency license texts, the project MIT license and readable third-party notices. Preserve earlier artifacts; build candidates into a fresh ignored directory.

Current checks and remaining release gates: [release-readiness report](docs/RELEASE_READINESS.md).
