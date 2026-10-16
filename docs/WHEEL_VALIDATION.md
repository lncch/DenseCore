# Local wheel validation

Artifact: `densecore-0.7.0-cp313-cp313-macosx_11_0_arm64.whl`
SHA256: `c366bd6c0a5469bed8d71b8fbdf2b86033fe568ab4c0085a25a5c4fc4a453a87`

Built with maturin release, installed into fresh CPython3.13 macOS ARM64 venv without PyTorch. NumPy2.5.4, safetensors0.8.0. 121 non-oracle tests passed, including15 compilation tests. Compiled dense example passed. Source oracle environment:182 tests passed. Linux0.7 and other Python/macOS versions unvalidated. No publication performed.

## Version0.8

Artifact:`densecore-0.8.0-cp313-cp313-macosx_11_0_arm64.whl`
SHA256:`0c6501958882ef1a4ba6df68c3c40b8db64ef8ef1f2b99c5bb342eff4423c2ab`

Final compact-broadcast release wheel installed into same clean no-Torch environment:122tests passed, including16compilation/private-IR tests. Source oracle suite:183Python tests;52Rust tests. Compiled regression example passed. No Linux0.8 or other Python-version validation.
