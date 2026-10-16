# Supported behavior on the storage development branch

Float math supports CPU float32; explicit boolean tensors support construction, checked views and logical operations, while named float32 comparisons return bool. DenseCore does not require PyTorch at runtime. This is a tested subset, not full PyTorch compatibility. GPU devices and dtypes other than float32/bool are rejected. Boolean math outside the documented subset is rejected. Construction and NumPy export copy values; no writable storage export, item assignment or in-place tensor arithmetic exists.

| Operation | Layout and forward behavior | Backward and broadcasting | Differences and limitations |
| --- | --- | --- | --- |
| Tensor creation / numpy | C-order float32 or explicit bool copy | Construction starts a leaf | Python-list transfer overhead; no zero-copy NumPy export |
| Boolean tensors | Native one-byte bool storage; shared views and copying export; logical_not / ~ and named logical_and/or/xor | Nondifferentiable, no requires_grad | No boolean indexing/reductions, NN math, joining or compiler capture |
| Comparisons | ==/!=/</<=/>/>= and eq/ne/lt/le/gt/ge on same-dtype float32 or bool tensors with matching scalar kinds; checked broadcasting | Nondifferentiable real bool outputs; reflected real scalars supported | Eager only; no mixed tensor dtype promotion; hashes use object identity, use is for identity checks |
| add / subtract / multiply / divide | Logical values from contiguous or strided inputs | First-order gradients; right-aligned singleton broadcasting | Scalar and reflected real operands supported; no array fallback; division follows IEEE nonfinite propagation |
| matmul | Rank-2 logical matrices; optimized CPU kernel boundary | First-order VJP for both operands | Rank-2 only; use explicit bmm for rank-3 |
| bmm | Equal-batch rank-3 logical matrices, including strided inputs | First-order VJP for both operands | No broadcasting or compiler capture; CPU float32 |
| Negation / exp / log / sqrt / sin / cos / sigmoid / tanh | Registered eager CPU float32 kernels; shape preserved; strided inputs read logically | First-order gradients, numerical full-Jacobian and PyTorch VJP tests | No compiler capture for these new math operations; domain NaNs/infinities propagate |
| nn.Sigmoid / nn.Tanh and functional equivalents | Parameter-free CPU float32 activations using registered unary kernels | First-order gradients, scalar/empty/strided inputs | Compiler uses eager fallback; no native module capture |
| gelu (exact erf and tanh forms), nn.GELU, functional.gelu | Registered eager CPU float32 kernels computed in float64 then rounded | First-order gradients, numerical full-Jacobian, float64 reference and PyTorch value/gradient tests | Compiler uses eager fallback; fused training rejects GELU models before mutation |
| nn.Embedding, functional.embedding | Native CPU float32 row gather; integer Python/NumPy index metadata of any shape | Scatter-add gradient, full-Jacobian on strided weight, PyTorch value/gradient parity, version checks | No padding_idx, max_norm, sparse gradients or compiler capture |
| Linear / LayerNorm / matmul on rank>2 inputs | Leading dimensions flattened onto rank-2 kernels; rank-3 @ rank-3 uses bmm | Full-Jacobian and PyTorch value/gradient tests on rank-3 inputs | No general matmul broadcasting; compile falls back to eager for non-rank-2 inputs |
| nn.Dropout, functional.dropout | Inverted dropout via a seeded NumPy mask multiplied in; train/eval aware | Rate, scaling, mask-consistent gradient, seed reproducibility, edge p | Mask drawn in Python; compiler runs Dropout models eagerly |
| Tensor.masked_fill | Native CPU float32 elementwise replace; mask broadcast in NumPy | Exact replacement over inf/NaN, zero masked gradient, strided full-Jacobian, causal softmax PyTorch parity | No in-place masked_fill_; no compiler capture |
| functional.scaled_dot_product_attention | Composite of bmm/softmax/masked_fill; rank 3 or 4; bool/float/causal masks | PyTorch value and q/k/v gradient parity for each mask mode, full-Jacobian with causal mask | No fused/flash kernel, no dropout_p, full score matrix in memory |
| relu | Logical elementwise operation | First-order gradient | No in-place variant |
| sum / mean | Registered CPU kernels; whole-tensor or selected axes; keepdims supported; shared compiler whole-reduction shape validation | Broadcasts upstream gradients across reduced axes | Empty means follow existing NaN behavior |
| softmax / log_softmax | Explicit axis, including negative Python axes | First-order gradient | Reduction axis must be nonempty |
| cat / stack | Concatenation copies along an existing axis; stack inserts a new axis | First-order gradients, duplicate parents and view aliases accumulate | Nonempty sequences; stack requires identical shapes; no compiler capture |
| flatten / nn.Flatten | Collapses inclusive dimensions via checked reshape; scalar becomes (1,); bool supported | First-order float32 gradients; no broadcasting | Single-axis non-scalar returns self; module default preserves batch; compiler eager fallback |
| squeeze / unsqueeze | Remove singleton axes or insert one; reshape storage semantics; bool supported | Float32 first-order gradients; no broadcasting | squeeze accepts None or one integer, no tuple axes; strided inputs may copy; no compiler capture |
| reshape | Contiguous inputs share allocation; strided inputs materialize | Restores logical input shape | Conservative copying for all non-contiguous input layouts |
| transpose / permute | Shared immutable metadata views | Inverse permutation | No negative physical strides |
| Basic indexing | Integer indices, positive-step slices, one ellipsis | Scatter into logical parent coordinates | No advanced/boolean indexing, new axes or negative slice steps |
| contiguous / layout inspection | Shares contiguous storage, otherwise copies logical values | Materialization retains gradient path | stride and offset measured in elements |
| Linear | Rank-2 input with weight and optional bias | Input/weight/bias gradients | Dense CPU subset |
| LayerNorm | Registered eager CPU float32 forward kernel; shared eager/compiler rank-2 last-dimension shape validation | Input/affine gradients | Existing supported normalized-shape subset only |
| MSE / cross entropy | Registered CPU float32 mean-loss forward; shared checked shape schemas | First-order loss gradients; MSE differentiates both operands | MSE requires matching nonempty shapes. Eager cross entropy permits empty batches; compiled execution requires nonempty batches. Class labels are metadata, not public integer tensors |
| Tensor.register_hook | Late CPU float32 leaf/nonleaf registration; ordered removable callbacks | Detached exact-shape replacement or None; transactional publication | Entry snapshots; nested hook backward/concurrent same-slot invocation rejected; fused training rejects active parameter hooks |
| autograd.Function | Positional float32 Tensor inputs, one float32 output; forward/backward under no_grad | Detached first-order gradients or None per input; saved versions checked | Tensor hooks on outputs run before custom backward; no higher-order AD, custom compiler capture, non-tensor args or multiple outputs |
| SGD / momentum / Adam | Leaf replacement with versioned immutable snapshots | Reads logical gradients, including strided gradients | Existing views retain old values; saved computations reject changed versions |
| Checkpoints | Saves logical values and optimizer state | Not a differentiable operation | Layout and alias topology are not preserved on load |

Existing kernels materialize strided input values once per TensorData layout when a contiguous slice is needed. Graph workspaces preserve that logical value contract. Compiled capture still supports its existing registered operation subset; the new Python slicing API is an eager operation and is not added as a compiler capture node. A compiled graph may consume an eager strided tensor as input, with gradients continuing through its parent view.

Autograd uses separate Node identities even when storage is shared. Retained graphs and parameter version checks follow the existing engine. Higher-order gradients and forward-mode AD are unsupported. Detached tensors are independent immutable snapshots, so mutable alias behavior is intentionally different from PyTorch.

## PyTorch-named entry points (drop-in subset)

Goal: CPU training scripts written for `torch` run with `import densecore as torch`. Target
script: nanoGPT `model.py` + `train.py`. API probe on 2026-10-16: 52 of 54 calls it uses work,
and `train.py`/`sample.py` run unchanged (`benchmarks/NANOGPT_DROPIN.md`).

| Name | Behavior | Differs from PyTorch |
|---|---|---|
| `zeros`, `ones`, `tril`, `from_numpy` | Same values and dtypes as PyTorch | `from_numpy` copies instead of sharing memory |
| `arange` | int64 for integer arguments; floats are float64 `start + i*step`, rounded once | PyTorch's float32 values can be a few ulp off |
| `Tensor.size/dim/ndim/view/t/to/item` | Same results; `to(dtype)`, `.float()/.double()/.half()/.bfloat16()/.long()/.int()/.bool()` convert | `view` copies where PyTorch would refuse; casts are detached copies |
| `F.softmax`, `F.log_softmax`, `torch.nn.functional` | Same values | float32 only |
| `nn.ModuleList`, `nn.ModuleDict`, `Module.apply` | Same registration names and order | |
| `Module.register_buffer` | Attribute only | Buffers are not in `state_dict` yet |
| `nn.init.normal_`, `nn.init.zeros_` | Replace values in place, seeded by `manual_seed` | Random stream differs from PyTorch's |
| `optimizer.param_groups` | One group; setting `"lr"` applies on the next step | Other keys are read-only; no multiple groups |
| `zero_grad(set_to_none=True)` | Gradients become None | `set_to_none=False` raises |
| dtypes | float32, bool, float64, float16, bfloat16, int8, int16, int32, int64, uint8 | Only float32 has math and gradients; other dtypes are storage-only (views, conversion, indexing). No complex, uint16/32/64 or float8 |
| `cuda`, `backends`, `amp`, `GradScaler` | CPU-only answers; GradScaler disables itself like PyTorch without CUDA | Enabled autocast raises |
| `save`, `load` | Pickle with tensors as raw bytes | `load` refuses every class except tensors and plain containers |

Conversions between all ten dtypes, and bfloat16 rounding, match PyTorch bit for bit
(`tests/test_dtypes.py`). Embedding and cross_entropy accept integer tensors.

Also added for nanoGPT: `split`, `topk`, `multinomial`, `randint`, one list/array/int-tensor
index per key, `None` indexing, `tensor[bool_mask] = scalar`, `tensor[ints/slices] = value`
(outside autograd), `copy_`, `tolist`, `Tensor.grad` assignment, `nn.utils.clip_grad_norm_`,
`optim.AdamW` and parameter groups (fused training and `densecore.checkpoints` reject grouped
optimizers; `densecore.save` handles them), `cross_entropy(ignore_index=)`, attention
`dropout_p` (PyTorch argument order), `Module.to`, and arithmetic/comparisons on integer and
float64/16 tensors with PyTorch's promotion rules.

Known gaps: `torch.tensor([1, 2])` still defaults to float32 (PyTorch infers int64); bool
`cat`/arithmetic stay rejected; in-place ops rebind the tensor instead of mutating shared
storage, so other views keep old values; float64 has no gradients.
