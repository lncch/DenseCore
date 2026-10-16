# Tensor hooks implementation record

Spec: GRADIENT_HOOK_DESIGN.md and independent read-only review. User explicitly
authorized autonomous milestone execution; no per-task approval pause.

Architecture: GC-visible Python slots contain only native Weak<Node> identities,
ordered Python callbacks and lazy locks. Every differentiable wrapper propagates
slots before descendants. Tagged hook events suspend native backward before VJP/
leaf staging; resume validates detached exact-shape gradients and versions after
each callback. Entry snapshots define removal/registration behavior. Supported
parameter replacement synchronizes active slots. Both public/native fused paths
reject active hooks before mutation.

- [x] Write missing-method tests, observe ten failures.
- [x] Implement Weak identity, tagged scheduler and Python slot ownership/handles.
- [x] Independent review; reproduce/fix native fused guard bypass.
- [x] Reproduce/fix marker loss after parameter replacement.
- [x] Add lifetime, retained/deleted-wrapper, checkpoint, concurrency, deep-graph,
  parity and invalid-result regressions. Native release/session tests pass.
- [x] Measure no-hook allocation/construction regression using separate fresh
  processes; record cost rather than claiming improvement.
- [x] Final full Rust/Python and installed source/wheel gates.
- [x] Commit implementation and separate documentation; update milestone ledger.

No-hook scalar-chain construction overhead is a performance issue for Milestone 2.
Inference mode and checkpointing require the explicit boundaries recorded in
INFERENCE_CHECKPOINT_INVESTIGATION.md, not aliases or generic wrappers.
