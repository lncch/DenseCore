# Stage 1 verification

Completed safe Rust-owned float32 tensor storage, checked Python bindings and local wheel packaging. Stage 1 is not a training engine yet.

Verified on Apple M4/macOS, CPython 3.13.7: 2 Rust tests, 10 Python tests, clippy with warnings denied, clean installed-wheel tests outside the source checkout, and README example. PyTorch was absent from the clean environment. Wheel contents contain only package code/native extension, metadata, MIT license and dependency SBOM.

Independent review identified oversized public reshape dimensions leaking a native conversion error. A regression test failed before the fix and passed afterward; rebuilt wheel passed all 10 tests. Public oversized dimensions now raise ValueError.

## Decisions and remaining limits

- New standalone feature-branch repository replaces a managed worktree because no production checkout existed. Cost if wrong: relocate checkout.
- Manual execution ledger replaces skill scripts because the plan lives outside this checkout. Cost if wrong: less automated bookkeeping; tests and commits retain evidence.
- Linux and other Python versions remain unvalidated. Cost: later packaging verification.
- Private `_native._Tensor` calls may raise PyO3 conversion exceptions for negative/oversized dimensions; this is not a supported public API. Public reshape validates these inputs. Cost if callers bypass the public interface: exception types differ. This private-interface review finding is deferred.

Copying through Python lists favors checked ownership over transfer speed in this stage. No benchmark claim applies. Package name densecore is provisional; no public registry publication occurred.

Next stage: tensor operations and eager reverse-mode autograd, followed by losses/optimizers, reusable execution, checkpoint import and portable release validation.
