# Stage 2 validation record

Date: 2026-10-14. Implementation: dense forward operations and first-order eager autograd owned by Rust, with checked PyO3 bindings. Platform: Apple M4/macOS, CPython 3.13.7; Linux remains unvalidated.

## Verified behavior

Current development suite: 73 Python cases, including 41 PyTorch 2.13.0 oracle cases without skips. Eight Rust tests cover storage/shape validation, broadcast validation, parent release, concurrent retained-backward/reset serialization and dropping a 20,000-node unused graph. Clippy passes with warnings denied. The rebuilt release wheel passes 32 non-oracle tests in a fresh environment without PyTorch, outside the source checkout. The README example prints [2. 4. 6.]. Wheel contents include the extension, Python package, license, metadata and dependency SBOM only.

Coverage includes branching, repeated edges, unequal-length shared paths, retained and consumed graphs, leaf gradient accumulation/reset, preflight errors preserving existing gradients, explicit seeds, matmul/transpose/reshape derivatives, scalar and row-bias reduction, empty tensors, finite differences and NaN/signed-zero ReLU parity. Float32 oracle tolerances are rtol 1e-4 and atol 1e-6. Equal NaNs are allowed in explicitly compared NaN results.

A deep-chain drop probe initially aborted with stack overflow. Iterative ancestor teardown fixed it and the 20,000-node regression passes. No speed benchmark was run on this public API.

## Decisions

Continue the clean standalone checkout on feat/eager-autograd rather than creating another worktree; only one writer edits it. Cost if relocation is needed: move the checkout.

Validate requiring gradient seeds at the binding before extracting raw TensorData; use iterative enter/exit traversal preserving duplicate operand edges; acquire engine lock before node locks in backward/grad/reset; implement ReLU masking with explicit selection. These resolve independent design-review concerns without broadening the API. Cost: serialized backward and validation overhead.

Iterative teardown adds drop-path work but prevents recursive destruction from crashing deep unused graphs. Package version advances to 0.2.0 to distinguish Stage 2 wheels from the Stage 1 artifact.

## Limits

Reference CPU kernels, allocating broadcast index maps, Python-list transfers and global gradient serialization favor correctness in this stage. No optimized math backend, model module, loss, optimizer, GPU, compilation or full PyTorch compatibility is claimed. Private-native conversion exception normalization remains deferred from Stage 1; public methods validate advertised inputs.

## Final independent review

Reviewer reran the original 66-case Python suite, eight Rust tests, clippy and 100 additional random shared-DAG oracle comparisons. One material finding: NumPy ndarray arithmetic could fall back to object arrays instead of rejecting unsupported operands. Four regression cases failed before the fix; explicit NumPy dispatch now rejects ndarray arithmetic in both orders. Scalar arithmetic remains supported and has gradient regression coverage. Final full-suite count is 73, including the 41 oracle cases.

Ruling: override NumPy ufunc dispatch at the Python boundary to enforce the supported operand contract; other ufuncs raise TypeError. Cost if broader NumPy integration is later needed: extend dispatch deliberately. Private-native exception normalization remains the only deferred Stage 1 review item.

Release wheel SHA-256: `aa6a100ee365414702ded07dce12f44fe35fedfca74ded9675c5d8ef948281f8`.
Final review finding is resolved by regression evidence; no second review was dispatched. Local branch is retained per the approved plan, without publishing or pushing.
