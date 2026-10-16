# Real tabular model implementation plan

> Agent execution: inline with executing-plans and TDD. Root is sole code writer. Independent design and final code review are read-only. Existing user approval authorizes all tasks; publication remains prohibited.

Goal: ship compiled operator support and a reproducible real-data classifier with matched accuracy/speed/RAM comparisons, validated Linux install and local MIT release preparation.

Architecture: reuse shape-specialized IR, checked GEMM, immutable autograd recipes and transactional optimizers. Add rank2 LayerNorm with native first-order gradients and guarded module capture. Keep one workspace per live gradient forward and existing eager fallback.

## Constraints

CPU float32, Linux/macOS, PyTorch-style supported subset, safetensors weights. Do not touch preserved backups/untracked benchmarks. No remote push/publication/paid services. Source tests use repository .venv; Rust task homes work/rust-cargo and work/rust-toolchains. Run release test/clippy and maturin develop after substantial source changes.

## Implementation units

1. Compiled matmul and whole-tensor sum/mean.
   Files: src/graph.rs, graph_workspace.rs, graph_backward.rs, graph_api.rs; tests/test_compile.py.
   API: Graph.matmul(left,right), Graph.reduce(input,mean), private operation specs matmul/sum/mean.
   RED: explicit-seed matmul and reduction VJP parity including branches, duplicate operands, empty tensors and wrong dimensions. GREEN: precomputed GEMM dimensions and transpose gradients; sum empty0/mean emptyNaN; backward reductions need no activation pins. Matmul pins opposite operand for requested derivatives.
2. LayerNorm.
   Files: src/normalization.rs, autograd.rs, lib.rs, graph.rs/workspace/backward/api; Python functional.py, nn.py, compilation.py; tests/test_layer_norm.py.
   API: functional.layer_norm(input, normalized_shape, weight=None,bias=None,eps=1e-5); nn.LayerNorm(int or one-element sequence,eps=1e-5,elementwise_affine=True,bias=True). Rank2 only, positive feature count; eps finite positive representable f32; affine defaults ones/zeros. Biased per-row variance. Same behavior in train/eval. Native eager/compiled forward and gradients share checked kernels. Invalid shapes/hyperparameters reject before state changes. RED tests Torch value/input/weight/bias parity, frozen/empty/bias-free/nonaffine/retained/stale graphs and capture guards. GREEN implementations, release suite and review.
3. Real task and benchmark.
   Dataset UCI Letter Recognition DOI10.24432/C5ZP40,20k rows16features26classes,CC-BY4. Download explicitly into ignored data cache; pin SHA256. Group identical feature tuples before seeded stratified80/10/10 group split; preserve rows, record actual sizes and hashes. Train-only standardization. MLP16→width→LayerNorm→ReLU→26, widths32/128, Adam.lr=.001,100epochs,batch128, fixed seeds/minibatch schedules, validation-best checkpoint. Gate validation accuracy >=.80/.90 for widths32/128; test inspected once after validation checkpoint selection. All engines same architecture/data/split/initialweights/batch schedule, final accuracy gap <=.005 absolute. Report macro-F1 as well as accuracy; no state-of-art claim.
   Fresh-process benchmarks widths32/128,batches32/128: Densecore eager/compiled, Torch eager/optimized-eager/compile when available. Optimized Torch uses preallocated inference and foreach optimizer where valid, complete step equivalence first. Time full zero_grad/forward/CE/backward/Adam.step. Separate setup/compile/firstcall/warmup/steady-state. Three processes per cell, requested1thread, alternating order. Include peak process RSS semantics; imports/setup/data excluded only from steady-state timing, not processRSS. Quality training separately from cached-batch timing; finite loss/output/gradient and20stepweight gates before timing.
4. Linux and local release preparation.
   Validate fresh wheels without Torch on LinuxARM64 and x86_64 where local container runtime available. Preserve task/user data; containers isolated to task source snapshot/cache. Add CI wheel/test matrices, dependency license notices, dataset attribution, reproducible quickstart and changelog. Build local macOS/Linux artifacts. If Linux environment unavailable, preserve evidence and clearly label unvalidated platform; do not claim success. Final fresh read-only review, fix blockers with regression tests, full checks, commit local work and update AUTONOMOUS_PROGRESS.

## Research sources

UCI https://archive.ics.uci.edu/dataset/59/letter%2Brecognition : size/features/license. Download https://archive.ics.uci.edu/static/public/59/letter%2Brecognition.zip . Dataset duplicate finding independently checked by reviewer; group split required.
PyTorch LayerNorm https://docs.pytorch.org/docs/2.13/generated/torch.nn.LayerNorm.html : per-last-dimension normalization, biased variance,eps and affine semantics. Context7 /websites/pytorch_2_13 checked.

## Execution ledger

Base9d40b7c. Design review running. All source writing inline. Progress entries appended after each verified increment.

Ruling: large-offset LayerNorm accuracy uses a double-precision oracle on exactly quantized float32 inputs and float32-rounded epsilon. Ordinary Torchfloat32 parity tolerance remains1e-4/1e-5. PyTorch CPU large-offset derivative suffers cancellation (primary kernel source plus independent analytic/finite-difference reproduction); emulating that error conflicts with numerical accuracy goal. Norm statistics and centered input gradients use f64 arithmetic internally, while tensor storage/API remainsf32. Initial failing stress comparison recorded separately; no tolerance relaxation.
Operator increment:14 compiledmatmul/reduction tests RED unsupportedop→GREEN. Normalization:63 targeted tests pass, including30 eager/compiled Torchfloat32 cases, four doubleprecision stress cases andshape/method mutation guards.

Final execution: five processes per performance cell (stronger replication than original three-process minimum). All accuracy/gradient/parameter gates passed. Cache-binding review fix verified with mutation regression and post-run data integrity check, without repeating heldout evaluation. Linux ARM64 and emulated x86-64 clean installs passed154tests each, macOS clean install154; source255; Rust52 plus6manualignored. Final formatter exposed preexisting cpu_kernels test whitespace; formatting-only correction makes prepared CI check pass. Full evidence: LETTER_RESULTS.md, LOCAL_RELEASE.md and RELEASE_ARTIFACTS.json. Final review and local commit follow.
