# Private compiled-gradient ownership transfer

Evidence: larger-model native samples include compiled VJP extraction/memmove;
current extraction copies full input adjoints. Independent review supports moving
only private CompiledVjp input buffers into immutable TensorData, with exact refill
before subsequent retained/retried backward. Public/fused borrowed paths unchanged.

- [x] Observe missing extraction method failure.
- [x] Add move helper; observe scalar identity-plan second backward fail before refill.
- [x] Restore drained reachable input lengths before seed/VJP and use immutable
  output spec for implicit scalar-seed validation. Move only in compiled derivative.
- [x] Validate pointer transfer, identity/empty/frozen/duplicate slots, retained
  graphs and hook-failure retry; preserve public workspace/fused tests.
- [x] Full Rust/Python/install gates and paired same-protocol measurements.
- [x] Independent implementation review and separate implementation/docs commits.

Refill allocation failure publishes no leaf updates. Sampling does not prove this
copy is the sole regression; measured outcome decides any speedup claim.
