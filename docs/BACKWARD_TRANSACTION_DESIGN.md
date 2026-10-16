# Backward gradient publication transaction

Status: historical prerequisite design. Custom Function and tensor hooks were
implemented afterward. Current semantics are in CUSTOM_FUNCTION.md and
TENSOR_HOOKS.md; later ownership/review evidence is in HOOK_VALIDATION.md.
Statements below about missing callback APIs describe this design's original
checkpoint, not current supported behavior.


## Prerequisite for custom callbacks

Current eager backward holds ENGINE throughout traversal and derivative computation. It writes each leaf gradient immediately. A later derivative or accumulation error can therefore return an error after earlier leaves changed. Python custom VJP exceptions would make this behavior directly user-visible.

The bounded prerequisite is to stage all incoming leaf gradients, acquire every graph state guard needed for publication/consumption before writing, and prepare every accumulated leaf value before committing. Then assign prepared gradients and mark non-retained non-leaf graphs consumed with no further fallible calls. Keep ENGINE held throughout this unit. Saved-version checks, numerical derivative formulas, traversal ordering and public API stay unchanged.

A Rust fault-injection regression creates an invalid saved reduction axis on one branch and a valid sibling leaf with an existing gradient. Reverse traversal reaches the sibling before the invalid derivative. Before the fix the failed call changes sibling gradient 1 to 2. The test requires existing gradients unchanged and Ready recipes preserved, repairs the injected recipe, retries once successfully and verifies a consumed graph rejects a later call. The invalid recipe exists only in test code; no production test hook or custom Function stub is introduced.

## Commit boundary

Acquire state guards for staged leaves and, when consumption is requested, non-leaf nodes from the unique traversal before any publication. Prepare final leaf values using existing accumulation arithmetic while those guards and ENGINE remain held. Any lock or arithmetic error returns before writes. Commit through the held guards only. Internal compiled backward workspaces can still update scratch/gradient buffers during derivative evaluation; this contract concerns published eager leaf gradients and graph consumption.

No user Python callbacks are introduced here. This does not solve callback deadlocks or claim custom autograd/hooks are supported. That following design must move callback evaluation outside locks callbacks can reacquire, preserve immutable snapshots and revalidate versions/readiness before commit. Callback side effects cannot be rolled back by the engine. Callback/context object destruction also matters: defer Python reference decrements and possible finalizers until after engine/state locks are released.

## Scope

No optimizer/runtime kernel change, dtype/backend extension, performance claim or version bump. Validate Rust failure/retry behavior and full Python/native regression suites. Commit implementation and documentation separately; keep generated build artifacts outside Git. Independent critique and actual final gate results are recorded in the validation checkpoint.

The independent critique supported the staging design and identified exact parent-gradient shape validation as another prerequisite. A second fault-injection test reproduced a same-element-count wrong-shaped VJP being accepted. Contributions now require exact parent shape before accumulation; ordinary numerical formulas are unchanged. The late-error fixture uses siblings on both sides of the bad branch to avoid depending on a single traversal direction.


## Validation checkpoint

Implementation: 9874d8b. Local macOS arm64, Python 3.13.7, rustc 1.99.0. Rust debug/release: **81 passed**, six manual benchmarks ignored. Full Python suite: **691 passed**, with two existing NumPy empty-mean reference warnings. Formatting and Clippy with warnings denied passed. The two new Rust tests were observed failing before the corresponding fixes. They inject malformed saved recipes through existing internal constructors; they do not expose a new public custom-backward API.

Independent read-only design and code critiques found no remaining concrete defect. Review checked Engine/node/workspace lock order, uniqueness of locked nodes, guard/Arc drop lifetimes, retained graphs and exact-shape invariants. Current constructors guarantee that staged leaves and consumed non-leaves are disjoint; future custom constructors must preserve that invariant. Review ran no tests/builds; gate results here were executed directly.

Fresh source installation used an activated new environment with maturin 1.15.0 and --no-build-isolation, without PyTorch: **554 passed, 28 skipped**, same reference warnings. Three Torch-only modules were excluded; other optional Torch cases skipped naturally. Direct wheel installation passed **71 autograd/optimizer/compiler tests**. Artifact inspection matched README metadata and all 30 license files and passed source scratch exclusions. The README mask example ran successfully against the direct wheel.

Validated archives remain outside Git in workspace directory work/backward-transaction-1792040960. Direct wheel SHA256: 77f9c2935fb261de31cd6c1dca3d684301debf066b1e55de03a0dbfe792671c6. Source archive SHA256: 306ac404a05dcc2f729e925519c884c86a5d1cf86d3fcc2b7db50145e55eb52b. This checkpoint section was written after building those archives. Generated benchmark data and release artifacts were not committed.

No version bump, push, merge, tag or publication occurred. Protected feat/v0.11 remains at 4044e5c0a415e9e20e0b57cc91b588e5b82f4bbc. Branch remains feat/v0.12-storage; package metadata remains 0.10.0.

## Next autonomous unit

The backward transaction prerequisite is complete. Custom Function and hook APIs are still unsupported. Next design must snapshot recipes/version/readiness under ENGINE, execute user callbacks without ENGINE/node/workspace locks, then revalidate and commit. Audit compiled backward assumptions before moving VJPs outside ENGINE, and account for concurrent/recursive backward and callback exceptions. Defer Python callback/context reference destruction and finalizers until locks are released. Choose one small real CPU float32 first-order Function protocol, with saved-tensor version tracking and numerical/error/reentrancy tests, before adding gradient hooks. Obtain independent critique before broad changes and do not add placeholder public APIs.

Continue the authorized mission without asking the user to say continue. GPU, higher-order gradients and distributed work remain incomplete; no later milestone is declared complete.
