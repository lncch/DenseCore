# Python comparison operators and identity audit

This float32 operator checkpoint predates [bool comparisons and bitwise logical operators](BOOLEAN_OPERATOR_VALIDATION.md). Its boolean-input/bitwise limitations describe that original checkpoint.

## Contract

Float32 Tensor comparison operators ==, !=, <, <=, > and >= delegate to the existing named native predicates. Results are nondifferentiable bool tensors. Real scalars are converted through the same float32 path, and reflected scalar ordering follows Python's opposite-operator dispatch. Broadcasting, strided values, NaN and signed-zero behavior are unchanged.

Unsupported operand types return NotImplemented from the operator method, allowing Python's reflected/fallback protocol. Equality against None or a plain unrelated object returns a Python false value; inequality returns true. Named comparison methods retain their explicit unsupported-type errors. Comparisons with a bool-dtype Tensor input remain unsupported and raise TypeError. No dtype promotion, native kernel or compiler capture extension is added.

## Identity and container behavior

Tensor explicitly retains object.__hash__, so adding __eq__ does not make Parameters unhashable. Hashes describe object identity, while comparison operators describe elementwise values. Use `is` for identity and `any(candidate is p for p in parameters)` for identity-based list membership. Distinct multi-element tensors in list membership or container equality can invoke comparison followed by ambiguous truth, which raises RuntimeError. Distinct single-element tensors can match by value in lists, so list membership is not an identity check. A dictionary/set of tensor keys follows identity-based hashing; it is not a value-deduplication API.

Audit inspected nn.py parameter enumeration/state staging, compilation.py signatures, native_training.py optimizer coverage and optim.py state publication. Parameter deduplication and graph signatures use id(); optimizer/model alignment uses is. No internal value-based tensor identity comparison needed replacement. Existing scalar/shape/config comparisons concern ordinary metadata, not Tensor values.

Regression tests cover all six operators and reflected scalars on strided NaN/signed-zero inputs, explicit object hashing, distinct tensor dictionary keys, shared Parameter registration, ambiguity in list membership, unsupported-object fallback and Parameter/key identity across a fused compiled optimizer step. Full existing tests exercise state loading and compiler guards.

## Limits and continuation

CPU float32 comparisons only; bool results are eager and nondifferentiable. Boolean comparison inputs, bitwise logical overloads, boolean compiler lowering, general dtype promotion and GPU remain unsupported. Package metadata stays 0.10.0. This unit adds no performance claim. Final review, tests, install results and local commits are recorded below after verification.

Reflected NumPy real scalars use the scalar-only ufunc bridge, including float32/float64/int64. Eighteen failing regressions reproduced NumPy invoking comparison ufuncs instead of Python reflection. The bridge now routes those scalar calls to native predicates while still rejecting non-scalar ndarray operands, ufunc kwargs and unsupported methods. All scalar values still use float32 conversion.

Independent read-only review confirmed internal id()/is tracking and found the same NumPy-left scalar bridge gap. It also documented scalar list membership changing from identity to value comparison. Both behaviors now have regression coverage. Preliminary archives built before the complete bridge fix failed eighteen new reflected-NumPy-scalar tests; those preliminary artifacts remain outside Git and are not the validated checkpoint. Final archives are rebuilt after correction.

NumPy 2.5.4 presents reflected numeric scalars to the ufunc hook as zero-dimensional arrays. The comparison bridge normalizes numeric 0-D arrays to Python scalars; direct numeric 0-D ndarray comparisons therefore use the same supported path. Higher-rank arrays remain rejected. Arithmetic ndarray behavior is unchanged.


## Validated checkpoint

Implementation: 98e0335. Local macOS arm64, Python 3.13.7, rustc 1.99.0, NumPy 2.5.4. Full Python suite: **677 passed**, with two existing NumPy empty-mean reference warnings. Rust debug/release: **79 passed**, six manual benchmarks ignored. Formatting and Clippy with warnings denied passed. The updated README example uses > and < and ran successfully against editable code and the fresh direct wheel.

Final source installation used an activated new virtual environment, maturin 1.15.0 and --no-build-isolation without PyTorch: **541 passed, 27 skipped**, with the same reference warnings. Three Torch-only modules were excluded; other optional Torch cases skipped naturally. Direct wheel install passed **45 comparison/predicate tests**, one optional Torch test skipped. README metadata and all 30 license files matched; source scratch exclusions passed.

Only validated archives at workspace directory work/comparison-validated-1792039990 are this checkpoint. Direct wheel SHA256: b25a919344bedce89f3a7de750a81a88745003e317d38ca3fb812cb43c63cdb2. Source archive SHA256: bf042f25151c72cab284b3ee43729d78b477f9d32dd736b2ecbee31a2e30861f. Earlier comparison-operators/comparison-final directories contain preliminary builds that failed new NumPy-left scalar tests and must not be presented as validated artifacts. All archives remain outside Git. This validation section was written after final builds.

No version bump, push, merge, tag or publication occurred. Protected feat/v0.11 remains at 4044e5c0a415e9e20e0b57cc91b588e5b82f4bbc. Current package metadata remains 0.10.0. No new performance, Linux CI or hardware claim is made.

Next dependency-ready unit: real bool-input comparison registrations and/or boolean bitwise logical overloads, with same-dtype rules and reflected boolean scalar tests. Keep bool compiler lowering and mixed dtype promotion explicitly unsupported until implemented. The autonomous mission remains active; no user continue message is needed.
