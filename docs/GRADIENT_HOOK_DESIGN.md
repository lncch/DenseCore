# Gradient hook design constraints

Status: historical investigation with accepted implementation appended below.
Updated 2026-10-15. Tensor hooks are implemented and locally validated; current
API semantics are in TENSOR_HOOKS.md and evidence is in HOOK_VALIDATION.md.
The investigation sections preserve the constraints considered before implementation.

## Required semantics

A CPU float32 tensor hook receives the accumulated gradient for its node once
per backward traversal, before that node's VJP or leaf publication. Registration
order determines callback order. Returning None preserves the incoming gradient;
a detached float32 Tensor with identical shape replaces it. Invalid results or
callback exceptions must preserve this backward's leaf gradients and graph
readiness. Callbacks run outside native engine, node and workspace locks, under
no_grad. Retained graphs invoke hooks on every backward. Disconnected nodes have
no hook invocation. A removable handle must remove a callback idempotently.

Late registration must work when descendants already exist. Parameter hooks must
survive optimizer/state replacement of the wrapper's native handle. Hooks on
non-leaf tensors affect propagation to parents. Hook order across unrelated nodes
is unspecified; order within one node is registration order. User side effects
cannot be rolled back. Recursive callback backward and concurrent invocation of
one callback context require explicit rejection or a validated execution policy.

## Why the existing custom owner DAG is insufficient

Tensor._contexts currently exists only on graphs with a custom Function ancestor.
Each ancestor owner tuple is immutable, and ordinary tensors have an empty tuple.
Adding a hook record to a tensor after descendants exist cannot update those
existing descendant tuples. Looking only at root contexts would silently miss
late hooks. TensorHandle.has_custom also cannot detect ordinary hooked nodes.

A global strong callback registry would keep callback closures and captured
Tensor wrappers alive indefinitely. A global weak registry would lose callbacks
when wrappers disappear while native descendants still need their gradients.
Putting Python callback references into Rust Arc nodes would reintroduce
cross-language GC cycles and finalizers under native locks. These alternatives
are rejected without an explicit ownership solution.

## Candidate boundary to review

One candidate gives each differentiable Python tensor a GC-visible hook slot
before descendants exist. Propagated owner DAGs share these slots. A slot owns
its callbacks and the exact native node identity to avoid address-reuse errors.
Native handle replacement must update the parameter slot through every supported
replacement path, while saved old graphs continue to fail version checks.

At backward entry, Python snapshots live hook slots and supplies node identities
to BackwardSession. The native scheduler yields a hook event after accumulation
and before VJP/leaf staging, then resumes with the checked replacement gradient.
It must yield each node at most once and revalidate versions/readiness at every
callback boundary. Custom VJP and hook events need distinct protocol tags.
Python callback ownership stays outside native Arc graph storage.

This adds owner propagation to every differentiable tensor, including ordinary
graphs. Measure allocation/retention overhead before accepting it. Do not retain
full native graphs through otherwise-unused hook slots without proving that
consumption and GC still release saved activations. Address reuse, stale slots,
wrapper aliases and detached tensors require concrete lifetime tests.

Another candidate is an explicitly GC-traversable native ownership bridge. It
needs a complete cycle-collection and guard-drop proof before adoption; merely
adding Py references to the existing Node is insufficient.

## Gate before implementation

Get independent critique of ownership and lifetime decisions, then choose the
smallest correct design using repository tests. External review attempts on
2026-10-15 could not provide this: Claude had exhausted its weekly quota, and
opencode/deepseek-v4-flash-free and opencode/north-mini-code-free each returned
an unexpected server error. No review result or approval is inferred from these
failures. Resume critique when a provider is available. Do not retry unchanged
failures continuously or ask the sleeping user to approve ordinary decisions.

Required regressions: late leaf/non-leaf registration, shared/duplicate parents,
multiple hooks and removal, exact dtype/shape, None returns, hook/custom ordering,
retained/consumed graphs, callback failure/retry, version changes in callbacks,
GC cycles, parameter replacement, compiled outputs, fused training behavior and
bounded concurrent/reentrant cases. Fused training must explicitly execute hooks
or reject them before committing a parameter update.

## Accepted implementation and review, 2026-10-15

Independent read-only review is now available through the requesting-code-review
workflow, superseding the earlier provider-availability blocker. Selected slots
hold Weak<Node>, never strong graph handles. Native allocation identity remains
stable while Weak exists. Every differentiable wrapper propagates slots; detached,
no_grad and bool outputs do not. Empty slots allocate callback dictionaries and
execution locks lazily. WeakSet tracks active slots for a no-hook backward fast
path, without owning callbacks strongly.

Native Hook and Custom events are distinct. Hook resume continues the same node
without repeating it. Python freezes ordered callback tuples at entry and checks
native readiness/versions/shape after each callback. Hook callbacks reject nested
public backward; existing custom Function independent ordinary backward remains
supported. Parameter replacement eagerly synchronizes active slots, including
optimizer, model and checkpoint paths. Fused execution rejects active hooks at
both Python and native preflight; empty slots synchronize before next use.

Review identified two native bypasses, reproduced by failing regressions and fixed:
missing native fused rejection, and stale markers after parameter replacement.
Final follow-up found no remaining code blocker. Runtime tests and measurements
are recorded separately in HOOK_VALIDATION.md; review itself ran no builds/tests.
