# Flatten checkpoint

Date: 2026-10-15. Base: d8717e5. Branch: feat/v0.12-storage.

Tensor.flatten collapses an inclusive normalized axis range using existing
checked reshape. Defaults flatten all dimensions. nn.Flatten defaults preserve
batch dimension. Scalar flatten produces shape (1,); non-scalar single-axis
flatten returns the same object. Axis booleans/nonintegers, out-of-range axes and
reversed ranges reject. Python integer products flow into existing platform-size
checks. Explicit bool dtype is preserved and remains nondifferentiable.

Contiguous flatten shares immutable storage; noncontiguous flatten copies logical
values through reshape. Float32 gradients and custom Function owners follow the
existing reshape path. Flatten modules are parameter-free and currently cause
guarded compiler eager fallback, including inside Sequential.

Fourteen initial tests failed for absent APIs. Final tests cover dimension ranges,
negative axes, scalar/empty shapes, values/gradients, contiguous/strided storage,
boolean tensors, custom ownership, invalid axes and module/compiler fallback.
Optional PyTorch parity checks a permuted tensor and input gradients.

Local macOS arm64, Python 3.13.7. Rust unchanged: debug 87 passed, 6 existing
manual benchmarks ignored; formatting and Clippy all targets with warnings
denied passed. Release editable extension rebuilt. Full Python: 774 passed. Fresh activated source installation: flatten/module/
compiler suites passed 36 tests, 1 optional Torch case skipped. Fresh wheel:
14 passed, 1 optional Torch case skipped. Two existing NumPy empty-mean reference
warnings remain. New README example executes against the fresh wheel.

Artifacts outside Git: ../../work/flatten-orHIPK. Artifact checker passes matching
README, 30 license files and source scratch exclusions.
Wheel SHA256: d1f770bb71309d1d218258f612c1acb381ffb56237335139f325d605ae7a7664.
Source SHA256: 3d5e172e713e96de2e7b5a7de840ac934e1c4e14593c4efe895be3dc2db7d57a.
Archives precede this checkpoint. Generated artifacts/data not committed.

No new native kernel, compiler capture, hardware validation or performance claim.
Package version remains 0.10.0; no push, merge, tag or publication.

Next unit: audit compiler fallback boundaries for the newly added modules,
including fused training rejection before mutation, then select the next bounded
operator gap. General hook ownership still requires independent critique as
recorded in GRADIENT_HOOK_DESIGN.md. Preserve user work and local-only history.
