# Singleton dimensions and compiler boundary audit

Goal: add Tensor.squeeze(dim=None) and Tensor.unsqueeze(dim) by composing existing
checked reshape. Squeeze supports None or one integer dimension; tuple dimensions
remain unsupported. Unsqueeze inserts one axis with range [-rank-1,rank]. Squeeze
uses [-rank,rank-1], treating scalar rank as one for dimension validation. Reject
boolean and noninteger dimensions. Squeezing a nonsingleton axis returns self.
Bool is preserved; float32 backward and custom ownership use reshape. Strided
inputs retain existing conservative reshape-copy behavior, documented explicitly.

- [x] Audit unsupported Sigmoid/Tanh/Flatten fused training with SGD/Adam; confirm
  values, native handles, gradients and optimizer state remain unchanged.
- [x] Observe 12 missing-method failures before implementation.
- [x] Implement and verify shape/value/gradient/dtype/axis/custom-owner cases.
- [x] Validate Rust/Python/build/install checks and execute README examples.
- [x] Commit code/tests and documentation separately.

No new native kernel, general view-stride transformation or compiler capture.
