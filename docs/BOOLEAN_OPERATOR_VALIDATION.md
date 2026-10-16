# Boolean comparisons and bitwise logical operators

## Supported behavior

Bool Tensor comparisons eq/ne/lt/le/gt/ge now have typed CPU registrations, callable through named methods/package functions or ==/!=/</<=/>/>=. Both tensor operands must be bool. Python bool and NumPy bool scalars are accepted, including reflected comparison forms. Numeric int/float scalar operands and float32 tensor operands are rejected by the bool comparison path. Ordering is False < True; truth tables were checked against installed PyTorch.

Boolean &, | and ^ delegate to the existing logical-and/or/xor kernels. All are commutative and have reflected methods. Operand types remain bool tensors or boolean scalars; integer bitwise arithmetic is not introduced. The existing ~ logical-not behavior is unchanged. Parenthesize comparison subexpressions because Python's bitwise precedence differs from comparison precedence.

Results use real bool storage, require no gradients and create no autograd Node. Checked broadcast shapes/indexing and immutable logical-value materialization are shared with existing predicates. Empty layouts, including huge unreachable dimensions, return empty results without index arithmetic. Tensor hashing remains object-based, and the identity/container rules from the previous operator checkpoint still apply.

## NumPy boundary

Comparison ufunc handling now preserves the Tensor operand's dtype instead of converting every scalar to a float tensor. A scalar on the left uses the inverse ordering operation on the Tensor on the right. Boolean bitwise ufuncs normalize only numeric boolean 0-D arrays; integer 0-D arrays are rejected. Comparison ufuncs retain their existing numeric 0-D normalization. Non-scalar ndarray operands, ufunc kwargs and reduction methods remain unsupported.

A failing regression reproduced direct float Tensor comparison with a NumPy bool scalar raising after the dispatch adjustment. The float path now normalizes that scalar to Python bool, preserving the same float32 scalar-conversion rule as Python bool. This does not promote mixed bool/float tensor operands.

## Tests and scope

New tests cover all six bool predicates with strided broadcasting, named and operator forms, reflected Python/NumPy booleans, all three bitwise operators, invalid mixed operands/shapes, empty/huge-empty layouts, rejected ufunc forms, mask-chain nondifferentiability and continued float backward. Optional PyTorch tests compare bool predicate and bitwise results. Full existing float comparison tests guard reflected real scalars and identity hashing.

No boolean indexing/reductions/joining, boolean NN Parameters, dtype promotion, compiler lowering, GPU or new performance claim is added. This extends the existing operator subset; it does not declare a later release complete. Package metadata remains 0.10.0. Independent review and final validation results are recorded below after gates finish.

Independent read-only review reported no remaining concrete defect in the final tree. It checked boolean ordering truth tables, commutative reflection, inverse comparison mapping, dtype rejection and ufunc boundaries. The reviewer observed the float/NumPy-bool scalar regression in an earlier snapshot and confirmed the final normalization and regression test. It ran no builds/tests; the results below come from direct validation.


## Validated checkpoint and continuation

Implementation: a5d2793. macOS arm64, Python 3.13.7, rustc 1.99.0. Full Python suite: **691 passed**, two existing NumPy empty-mean reference warnings. Rust debug/release: **79 passed**, six manual benchmarks ignored. Formatting and Clippy with warnings denied passed. The updated README mask example, including bool equality and &, executed against editable code and the fresh direct wheel.

Fresh source installation in an activated new environment, maturin 1.15.0 and --no-build-isolation without PyTorch: **554 passed, 28 skipped**, same warnings. Three Torch-only modules were excluded; other optional Torch tests skipped naturally. Direct wheel installation passed **58 targeted bool/float comparison tests**, two optional Torch tests skipped. Artifact checks matched README metadata and all 30 license files and confirmed source scratch exclusions.

Validated archives remain outside Git at workspace directory work/bool-operators-1792040360. Direct wheel SHA256: 7aa09a181f06f6e1e93a77c2bdd8bb45a30bba65ffcd6de33b478f67de39845a. Source archive SHA256: 4e8135327bbdd7f112b7f9da7d4d6f56a716037dfc74ba26be1d4b0e251f4468. This validation section was added after building those archives. Generated benchmark data and release artifacts were not committed.

Protected feat/v0.11 remains at 4044e5c0a415e9e20e0b57cc91b588e5b82f4bbc. No version bump, push, merge, tag or publication occurred. Working branch remains feat/v0.12-storage; package metadata remains 0.10.0.

Next unit under the autonomous mission: design custom first-order autograd functions, then gradient hooks. Inspect engine lock/GIL ownership and callback reentrancy before choosing a Python callback interface. Get an independent design critique before broad changes. Do not invoke user Python callbacks while holding locks they can reacquire; preserve saved-version checks, graph lifetime, exception behavior and optimizer failure atomicity. Start with a small real CPU float32 function protocol and numerical/error tests, not a placeholder API. Higher-order gradients, GPU and distributed work remain incomplete.
