"""Compare checked GEMM initialization against an explicit local Git baseline.

Standalone Rust probe, not the Python API or an end-to-end performance claim.
Numerical gates precede timings; generated sources/results stay outside Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=False)
    baseline_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.baseline_ref + "^{commit}"],
        cwd=root, text=True,
    ).strip()
    baseline = subprocess.check_output(
        ["git", "show", baseline_commit + ":src/cpu_kernels.rs"], cwd=root, text=True,
    )
    candidate = (root / "src/cpu_kernels.rs").read_text()
    for label, source in [("baseline", baseline), ("candidate", candidate)]:
        # include! inside a module cannot contain an inner module doc comment.
        (output / (label + ".rs")).write_text(source.replace("//!", "//"))
    probe = Path(__file__).with_name("gemm_initialization_probe.rs").read_bytes()
    (output / "probe.rs").write_bytes(probe)
    env = os.environ.copy()
    env.update(VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run(
        ["rustc", "--edition=2024", "-O", str(output / "probe.rs"),
         "-o", str(output / "probe")], env=env, check=True,
    )
    with (output / "timings.jsonl").open("x") as result, (output / "validation.log").open("x") as log:
        subprocess.run([str(output / "probe")], env=env, stdout=result, stderr=log, check=True)
    provenance = {
        "baseline_commit": baseline_commit,
        "baseline_kernel_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
        "candidate_kernel_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "probe_sha256": hashlib.sha256(probe).hexdigest(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "threads": {k: env[k] for k in ["VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"]},
        "scope": "initialized-buffer standalone GEMM, five alternating repetitions",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
