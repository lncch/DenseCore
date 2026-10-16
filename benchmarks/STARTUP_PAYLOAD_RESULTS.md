# Fresh-process startup and installed dependency payload

2026-10-15. Benchmark 453057e; DenseCore runtime a2fb278, package metadata 0.10.0.
Apple M4, macOS 27.0 arm64. Both selected interpreters report the same Python
3.13.7 build and NumPy 2.5.4. PyTorch 2.14.1. This is a local supported-workload
comparison, not a capability-equivalence or general throughput claim.

## Process wall time

Five fresh subprocesses per environment/mode, with alternating order. All modes
use Python -I to ignore PYTHONPATH and user site configuration. Empty and NumPy
controls are measured separately in each environment. The parent clock includes
process launch, selected work, output serialization and process exit. Both
framework forward gates complete before recorded timings, and every measured
forward checks finite values and the exact expected shape/value within tolerance.

| Mode | DenseCore environment median ms (min-max) | Torch environment median ms (min-max) |
| --- | --- | --- |
| empty | 36.1 (31.0-60.5) | 39.0 (33.5-49.2) |
| numpy | 127.7 (108.9-466.6) | 165.4 (117.4-379.8) |
| import | 195.2 (154.2-325.3) | 1536.7 (1407.0-2686.2) |
| forward | 228.9 (169.8-373.3) | 1489.4 (1284.1-1923.8) |

Mode definitions:

- empty: interpreter runs `pass` and exits.
- numpy: interpreter imports the small standard-library harness and NumPy, emits
  its version and exits.
- import: adds the framework import and, for Torch, one-thread intra/inter-op
  setup. Both receive one-thread BLAS/OpenMP environment controls.
- forward: adds a public Linear(2,2), initial parameter allocation, loading fixed
  float32 weights/bias, input conversion, one no-grad forward, NumPy export and
  JSON result. Fixed input [2,3] and weights [[1,2],[3,4]], bias [1,2], produce
  [9,20] in both frameworks. No compiled mode or model checkpoint loading here.

Inventory and prechecks warm operating-system caches before measurement. Processes
are fresh; filesystem caches are not cold. Import and forward rows use independent
process samples, so do not subtract medians to estimate a kernel duration. The
Torch import median exceeding its forward median illustrates run variability.
These measurements include application setup/shutdown, not steady-state inference.

## Dependency-inclusive installed payload

DenseCore plus NumPy and declared runtime dependencies: **35,328,152 bytes**
(35.3 decimal MB), 1,418 unique recorded files. Included distributions:
DenseCore 0.10.0, NumPy 2.5.4, safetensors 0.8.0.

Torch plus NumPy and declared runtime dependencies: **711,814,271 bytes**
(711.8 decimal MB), 21,936 unique recorded files. Included distributions:
Torch 2.14.1, NumPy 2.5.4, fsspec 2026.9.0, Jinja2 3.1.6, MarkupSafe 3.0.4,
networkx 3.7, sympy 1.14.0, mpmath 1.3.0, setuptools 84.0.0,
typing-extensions 4.16.0 and filelock 4.1.1.

This counts logical sizes of existing files named by wheel RECORD entries across
the declared dependency closure, plus explicit NumPy. Markers/extras and installed
version constraints are checked; resolved paths deduplicate shared files. It
includes recorded metadata and bytecode. Unrecorded files, optional extras,
Python itself, standard library, OS libraries, package caches and unrelated dev/test
dependencies are excluded. It is neither allocated filesystem blocks nor download
size, total environment size or RAM. Torch has substantially broader capabilities.

DenseCore uses the previously validated installed wheel environment; Torch uses
its installed wheel in the development environment. They are separate environments,
with separate empty/NumPy controls and matched Python/NumPy identities. Remaining
environment differences and bytecode/file inventories can affect results.
Native-entry hashes identify entry binaries, not every transitive shared library.

## Reproduction and evidence

Run the benchmark with paths to installed-wheel environments that each have
`packaging` available for dependency inventory. The tested DenseCore environment
already has it through pytest; the development environment uses requirements-dev.txt.
Use a new output path. The exact executed command and interpreter locations are
recorded in the selected JSON provenance. Command shape used here:

```sh
.venv/bin/python benchmarks/benchmark_startup.py --densecore-python "$PWD/../../work/lifecycle-packaging-41SMg1/validated/wheel-env/bin/python" --torch-python "$PWD/.venv/bin/python" --repeats 5 --output "$PWD/../../work/startup-payload-TdbU5u/results.json"
```

That output exists and is intentionally protected against overwrite. For another
run, choose a fresh output filename. This command was executed with the same
arguments by an enclosing shell that created the fresh artifact directory.

Raw results, exact script copy and provenance stay outside Git at
work/startup-payload-TdbU5u relative to workspace root. All 40 timing samples,
full ranges, dependency versions and interpreter/native identities are deliberately
versioned in startup-payload-2026-10-15-summary.json. Two pre-timing numerical gates
and ten measured forward gates passed. No measurement was dropped.

## Validation

Ten benchmark tests cover numerical rejection, dependency extras/markers/cycles,
file deduplication and invalid installation metadata. A missing-file test initially
failed because Distribution.files silently filtered absent files. Raw RECORD
parsing now detects the incomplete installation; all ten tests pass. Full Python
suite passes 975 tests with two existing NumPy reference warnings. Runtime code
is unchanged; prior Rust debug/release 93-test, formatting/Clippy and installed
artifact checks remain the native baseline.

Independent read-only benchmark review found no concrete measurement invalidator;
its limits are included above. Python/NumPy identity and benchmark-source hash were
checked against the actual recorded run. No remote CI, GPU or Linux execution claim.

Updated wheel/source builds also pass README metadata, all 30 license files and
scratch/result exclusion checks. Sixteen benchmark/archive regressions pass from
the unpacked source distribution using the installed-wheel environment. Package
artifacts and exact SHA256 receipts remain outside Git under
work/startup-payload-TdbU5u/packages and package-receipt.json. These builds follow
the README update but precede this final validation note. The benchmark itself
used the earlier wheel identified in its recorded provenance, with unchanged
native runtime source. Nothing was installed globally or published.
