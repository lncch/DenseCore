# Third-party notices

Densecore project source is MIT licensed. Native distribution depends on the Rust packages listed below, with original license texts included. Dependencies retain their own licenses. NumPy and safetensors are installed separately by pip and are not bundled as Python packages in the Densecore wheel. Dataset attribution appears in docs/DATASET_ATTRIBUTION.md; raw training data is excluded from distribution.

- heck 0.5.0: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/heck-0.5.0/.
- libc 0.2.190: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/libc-0.2.190/.
- once_cell 1.21.4: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/once_cell-1.21.4/.
- portable-atomic 1.15.0: Apache-2.0 OR MIT; texts in THIRD_PARTY_LICENSES/portable-atomic-1.15.0/.
- proc-macro2 1.0.107: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/proc-macro2-1.0.107/.
- pyo3 0.29.3: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/pyo3-0.29.3/.
- pyo3-build-config 0.29.3: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/pyo3-build-config-0.29.3/.
- pyo3-ffi 0.29.3: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/pyo3-ffi-0.29.3/.
- pyo3-macros 0.29.3: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/pyo3-macros-0.29.3/.
- pyo3-macros-backend 0.29.3: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/pyo3-macros-backend-0.29.3/.
- quote 1.0.47: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/quote-1.0.47/.
- syn 2.0.119: MIT OR Apache-2.0; texts in THIRD_PARTY_LICENSES/syn-2.0.119/.
- target-lexicon 0.13.5: Apache-2.0 WITH LLVM-exception; texts in THIRD_PARTY_LICENSES/target-lexicon-0.13.5/.
- unicode-ident 1.0.27: (MIT OR Apache-2.0) AND Unicode-3.0; texts in THIRD_PARTY_LICENSES/unicode-ident-1.0.27/.
- PyTorch 2.14.1 (ported source, not a dependency: src/torch_sum.rs, src/sleef.rs, src/torch_blas.rs, src/torch_softmax.rs, src/torch_attention.rs, src/torch_random.rs, src/normalization.rs, src/loss.rs, python/densecore/optim/lr_scheduler.py, python/densecore/utils/data/): BSD-3-Clause; text in THIRD_PARTY_LICENSES/pytorch-2.14.1/.
- SLEEF commit 5a1d179 (ported source, not a dependency: src/sleef.rs, src/sleef_rempitab.in): BSL-1.0; text in THIRD_PARTY_LICENSES/sleef-5a1d179/.
- Arm Optimized Routines (expf, via PyTorch's vec128_float_neon.h; ported in src/sleef.rs): MIT OR Apache-2.0 WITH LLVM-exception; text in THIRD_PARTY_LICENSES/arm-optimized-routines/.
