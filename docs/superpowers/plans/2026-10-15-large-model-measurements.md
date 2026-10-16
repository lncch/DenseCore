# Larger-model measurement gate

Goal: measure a genuinely larger MLP without serializing millions of float values
as Python/JSON lists. Add positive input/hidden/output feature dimensions, default
64/128/32. Parent subprocesses save initial output and 20-step weights in binary
NPZ files under a fresh owned directory beside the selected output. Numerical
gates check hashes, keys, exact shapes and allclose before accepting timings.
Retain payloads as raw provenance outside Git; never silently drop validation.
Legacy direct child JSON remains available when no validation-output is provided.

- [x] Observe two missing CLI/helper regressions before implementation.
- [x] Implement configurable features and checked binary validation payloads.
- [x] Verify all engines on a small case; test tampering/shape/content failure.
- [x] Run full tests and appropriate native gates; independent measurement review.
- [x] Measure 256/1024/256 MLP at small and larger batches, three fresh processes,
  preserving losing results. Record provenance and differences in cold/memory gates.
- [x] Update one milestone ledger and commit implementation/docs separately.

No runtime/optimizer/kernel change or new performance assertion before evidence.
