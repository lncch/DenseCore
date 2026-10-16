# Post-transfer compiled-training profile

2026-10-15, local macOS arm64 Apple M4. Runtime implementation 593d40d;
repository checkpoint a4e3320. This is a sampling investigation, not a latency
benchmark or a claim that any optimization has succeeded.

The existing deterministic profile workload runs a 256-1024-256 ReLU MLP,
batch 32, float32, SGD, 20 warmup steps and 20,000 measured-workload steps.
A three-second native sample at a requested one-millisecond interval completed
successfully after warmup. Workload and sample processes both exited zero.
OMP, OpenBLAS, Accelerate and MKL thread environment controls were set to one.
Raw trace, process log and build/workload hashes remain outside Git under
work/compiled-post-transfer-profile-jcrbgu20 relative to the workspace root.

Observed main-thread call-tree counts include 320 samples under native backward,
301 under Optimizer.step, and 226 under one compiled backward branch. Within
that backward branch, 43 samples descend into zeroing, and two GEMM branches
contain 37 and 14 zeroing samples. A separate workspace-construction branch
contains 25 zeroing samples. These counts are inclusive stack observations,
not independent timers or additive percentages. Sampling/inlining and one short
window limit attribution. Do not compare their counts directly with older traces
to infer a speedup.

Source inspection explains concrete investigation targets:

- graph_api.rs creates a private planned workspace for every differentiable
  compiled output. Its saved activations must survive other forwards and cache
  eviction; a single shared mutable training workspace would violate this.
- graph_backward.rs initializes adjoint buffers during workspace construction,
  then clears them before each backward. Repeated backward and failure retry
  require clean accumulation destinations.
- cpu_kernels.rs::gemm_into resizes its output and fills zero when no bias is
  supplied. On macOS, nonempty GEMM then calls BLAS with beta=0, which overwrites
  output. Zero-inner-dimension output and the reference backend still require
  explicit initialization. Bias uses beta=1 and must preserve bias values.
- Compiled matrix VJPs write scratch GEMM results and add them to parent adjoints.
  Shared parents require accumulation, so overwriting parent gradients without
  a proven first-write schedule would be incorrect.

That GEMM initialization gate is recorded in GEMM_INITIALIZATION_VALIDATION.md.
The bounded workspace-capacity lifecycle study is recorded in
WORKSPACE_MEMORY_VALIDATION.md. Further allocation work must count TensorData and
leaf-gradient storage separately; these workspace numbers do not establish process
peak RAM. No optimizer change follows from this profile.

Provenance: native SHA256 15e946edd04fa9a0e3ae82471795aca36d73466da4295564cbdd7efc0e7a4447; workload SHA256 01a7096fcab3e80f08f79ba153318f84a66708028c8be099684f98da91897628.
