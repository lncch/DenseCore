# Letter Recognition dataset attribution

Slate, D. (1991). Letter Recognition [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZP40 . Source: https://archive.ics.uci.edu/dataset/59/letter%2Brecognition . Licensed CC BY4.0: https://creativecommons.org/licenses/by/4.0/ .

Dataset contains20,000 rows,16 numerical image features and26 letter classes. Quantized features include1,332 duplicate rows. Our preprocessing preserves rows, groups identical features before approximately80/10/10 stratified group split, and standardizes using training-only statistics. Resulting split15999train/1979validation/2022test differs from UCI's suggested first16000/last4000 split. Font identities unavailable; unseen-font generalization is untested.

Raw dataset remains in ignored data cache and is not bundled in wheels. Packaged example model and example feature row derive from this dataset and carry this attribution. Project source code is MIT licensed; dataset retains its separate license. No endorsement by dataset creator or UCI is claimed.
