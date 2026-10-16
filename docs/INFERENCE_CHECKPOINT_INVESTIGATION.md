# Inference mode and checkpoint prerequisites

2026-10-15, local macOS arm64; installed PyTorch 2.14.1 used as a behavioral
reference, not a runtime dependency. Raw probe output stays outside Git at
../../work/milestone-audit/inference-prerequisite.json.

The probe creates a tensor under torch.inference_mode. Reading its version raises
`Inference tensors do not track version counter.` Multiplying it into a tracked
computation rejects saving an inference tensor for backward. The same tensor made
under torch.no_grad can participate later as a saved constant; gradient was [2.0].
DenseCore currently has no inference_mode. Merely aliasing no_grad would imply
unsupported semantics. A real implementation needs inference-tensor metadata,
saved-tensor admission checks and a documented conversion boundary. Version
tracking protects current parameter snapshots; disabling it without those changes
would weaken correctness. Defer public inference mode until that boundary has a
validated design. no_grad remains supported and does not claim these optimizations.

Checkpointing can build on custom Function and ordinary recomputation, but safely
supporting arbitrary closures needs isolated gradient computation or a native
allowed-leaf validation boundary. Recomputing then calling ordinary backward can
publish gradients into captured parameters before the outer backward fails.
Nested custom recomputation is also rejected by current reentrancy rules. A first
useful subset should require deterministic tensor-only pure functions, explicit
trainable inputs, unchanged output shape, version-checked saved inputs, and reject
external differentiable leaves before publishing recomputation gradients. RNG and
side-effect preservation must be explicit. Do not publish a generic checkpoint
wrapper until those requirements are implemented and tested.

Higher-order and forward-mode AD remain unsupported. These are reasoned deferrals,
not implemented features or passing milestone gates.
