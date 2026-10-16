"""Run PyTorch's own OpInfo samples through DenseCore and compare with PyTorch.

For every OpInfo entry that supports float32 on CPU, each sample input is converted to
DenseCore, called through the same public name, and compared: output dtype, shape and
values, then gradients of every float32 input that requires grad.

    PYTHONPATH=python python scripts/conformance.py [--samples 4] [--only add,mul]

Needs torch and expecttest. Writes benchmarks/CONFORMANCE.json and prints a summary.
Statuses, worst first: missing (no such name), error (raised), mismatch (wrong dtype,
shape or values), forward (values match, gradients do not), pass.
"""
import argparse
import json
import pathlib
import signal
import types
import warnings

import numpy as np
import torch
import densecore as d

warnings.filterwarnings("ignore")
from torch.testing._internal.common_methods_invocations import op_db  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def to_dense(value):
    """Convert sample arguments from torch to DenseCore, recursively."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy() if value.dtype != torch.bfloat16 else value.detach().float().numpy()
        dtype = str(value.dtype).split(".")[1]
        return d.tensor(array, dtype=dtype, requires_grad=bool(value.requires_grad and dtype == "float32"))
    if callable(value) and getattr(value, "__module__", "").startswith("torch.nn.functional"):
        return getattr(d.functional, value.__name__)  # e.g. distance_function=F.pairwise_distance
    if isinstance(value, torch.nn.Module) and not list(value.parameters()) and hasattr(d.nn, type(value).__name__):
        # A parameter-free module argument (e.g. distance_function=nn.PairwiseDistance()): same settings.
        settings = {{"norm": "p"}.get(k, k): v for k, v in vars(value).items()  # PairwiseDistance stores p as norm
                    if not k.startswith("_") and k != "training"}
        return getattr(d.nn, type(value).__name__)(**settings)
    if isinstance(value, torch.dtype):
        return str(value).split(".")[1]
    if isinstance(value, torch.device):
        return "cpu"
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, (list, tuple)):
        return type(value)(to_dense(v) for v in value) if not hasattr(value, "_fields") else tuple(to_dense(v) for v in value)
    if isinstance(value, dict):
        return {k: to_dense(v) for k, v in value.items()}
    return value


def resolve(op, dense_input):
    """The DenseCore callable for an OpInfo name, or None when the name does not exist.

    OpInfo sometimes wraps the call in a test-suite lambda (for example to reorder
    arguments); that lambda is rebuilt with `torch` bound to DenseCore instead.
    """
    wrapper = op.op
    if isinstance(wrapper, types.FunctionType) and wrapper.__module__.startswith("torch.testing"):
        return types.FunctionType(wrapper.__code__, {**wrapper.__globals__, "torch": d},
                                  wrapper.__name__, wrapper.__defaults__, wrapper.__closure__)
    name = op.name
    if name.startswith("__"):
        if not hasattr(dense_input, name):
            return None
        return lambda inp, *a, **k: getattr(inp, name)(*a, **k)
    target = d
    for part in name.split("."):
        target = getattr(target, part, None)
        if target is None:
            break
    if callable(target):
        return target
    if op.method_variant is not None and hasattr(dense_input, name.split(".")[-1]):
        method = name.split(".")[-1]
        return lambda inp, *a, **k: getattr(inp, method)(*a, **k)
    return None


def leaves(value, kind):
    """Flatten tensors out of nested outputs/arguments."""
    if isinstance(value, kind):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in leaves(v, kind)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in leaves(v, kind)]
    return []


def as_array(t):
    if isinstance(t, d.Tensor):
        return t._array() if t.dtype == "bfloat16" else t.numpy()
    if t.dtype == torch.bfloat16:
        return t.detach().float().numpy()
    return t.detach().resolve_conj().resolve_neg().numpy()  # PyTorch may hand back lazy conj views


RANDOM = ("dropout", "rand", "multinomial", "normal", "bernoulli", "poisson", "uniform",
          "exponential", "cauchy", "geometric", "log_normal", "empty", "rrelu")


EXACT = False  # --exact: float results must match PyTorch bit for bit (any NaN equals any NaN)


def _same_bits(x, y):
    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
    if x.dtype != y.dtype or x.shape != y.shape:
        return False
    both_nan = np.isnan(x) & np.isnan(y)
    if x.size == 0:
        return True
    raw = lambda a: a.view(np.uint8).reshape(a.shape + (a.itemsize,))
    return bool(np.all((raw(x) == raw(y)).all(-1) | both_nan))


def compare(ours, theirs, random=False, widened=False):
    """None when outputs agree, else a short reason. Random ops check dtype and shape only."""
    ours_t, theirs_t = leaves(ours, d.Tensor), leaves(theirs, torch.Tensor)
    if not theirs_t:
        return None if ours == theirs or (isinstance(ours, float) and np.isclose(ours, theirs)) else "non-tensor output differs"
    if len(ours_t) != len(theirs_t):
        return f"{len(ours_t)} output tensors, expected {len(theirs_t)}"
    for a, b in zip(ours_t, theirs_t):
        if not widened and a.dtype != str(b.dtype).split(".")[1]:
            return f"dtype {a.dtype}, expected {str(b.dtype).split('.')[1]}"
        if a.shape != tuple(b.shape):
            return f"shape {a.shape}, expected {tuple(b.shape)}"
        if random:
            continue
        x, y = as_array(a), as_array(b)
        if x.dtype.kind in "fc" and EXACT and not widened:
            if not _same_bits(x, y):
                return "values differ"
        elif x.dtype.kind in "fc":
            if not np.allclose(x, y, rtol=1e-5, atol=1e-5, equal_nan=True):
                return "values differ"
        elif not np.array_equal(x, y):
            return "values differ"
    return None


def check_sample(op, fn, sample):
    """Return (status, reason) for one sample."""
    t_args = [sample.input, *sample.args]
    d_args = to_dense(t_args)
    d_kwargs = to_dense(sample.kwargs)
    expected = op(sample.input, *sample.args, **sample.kwargs)
    got = fn(d_args[0], *d_args[1:], **d_kwargs)
    signs = None
    if op.name == "linalg.eig" and expected[1].numel():
        # Eigenvectors are unique only up to a complex phase (PyTorch docs): align columns.
        ours_v, theirs_v = got[1].numpy(), expected[1].detach().resolve_conj().numpy()
        dots = (np.conj(ours_v) * theirs_v).sum(-2, keepdims=True)
        phase = np.where(np.abs(dots) > 0, dots / np.where(np.abs(dots) > 0, np.abs(dots), 1), 1)
        got = (got[0], d.tensor((ours_v * phase).astype(ours_v.dtype)))
    if op.name == "linalg.eigh" and expected[1].numel():
        # Eigenvectors are unique only up to sign (PyTorch docs): align DenseCore's columns.
        dots = (got[1].numpy() * expected[1].detach().numpy()).sum(-2, keepdims=True)
        signs = np.where(dots < 0, -1.0, 1.0).astype(np.float32)
        got = (got[0], got[1] * d.tensor(signs))
    random = any(word in op.name for word in RANDOM) or sample.kwargs.get("dropout_p", 0) > 0 or (
        op.name == "nn.functional.multi_head_attention_forward" and sample.args[9] > 0  # positional dropout_p
        and sample.kwargs.get("training", True))
    reason = compare(got, expected, random)
    if reason == "values differ" and not EXACT:
        # float32 PyTorch can be the less accurate side (trigamma near a pole): accept values
        # that match PyTorch computed in float64, at the same tolerance.
        reference = _float64_forward(op, sample)
        if reference is not None and compare(got, reference, random, widened=True) is None:
            reason = None
    if reason:
        return "mismatch", reason
    if random:
        return "pass", None
    t_in = [t for t in leaves(t_args, torch.Tensor) if t.requires_grad and t.dtype == torch.float32]
    d_in = [t for t in leaves(d_args, d.Tensor) if t.requires_grad]
    # Pair outputs by position; only PyTorch outputs that carry gradients are checked.
    pairs = [(a, b) for a, b in zip(leaves(got, d.Tensor), leaves(expected, torch.Tensor))
             if b.requires_grad and b.dtype == torch.float32]
    d_out, t_out = [a for a, _ in pairs], [b for _, b in pairs]
    if not t_in or not t_out:
        return "pass", None
    rng = np.random.default_rng(0)
    weights = [rng.normal(size=tuple(t.shape)).astype(np.float32) for t in t_out]
    t_loss = sum((t * torch.from_numpy(w)).sum() for t, w in zip(t_out, weights))
    t_grads = torch.autograd.grad(t_loss, t_in, allow_unused=True)
    if not all(t.requires_grad for t in d_out):
        return "forward", "outputs do not carry gradients"
    d_loss = d_out[0].sum() * 0.0
    for t, w in zip(d_out, weights):
        d_loss = d_loss + (t * d.tensor(w)).sum()
    d_loss.backward()
    got = [np.zeros(a.shape, np.float32) if a.grad is None else a.grad.numpy() for a in d_in]
    if all(_close(g, b, a.shape) for g, b, a in zip(got, t_grads, d_in)):
        return "pass", None
    # float32 PyTorch can lose precision (sinc near 0, logcumsumexp over wide ranges); accept
    # DenseCore when it matches PyTorch run in float64 instead.
    reference = None if EXACT else _float64_grads(op, sample, weights)
    if reference is not None and all(_close(g, b, a.shape) for g, b, a in zip(got, reference, d_in)):
        return "pass", None
    return "forward", "gradients differ"


def _close(got, expected, shape):
    zero = expected is None or expected._is_zerotensor()  # PyTorch may return a lazy "ZeroTensor"
    if EXACT:
        want = np.zeros(shape, np.float32) if zero else expected.detach().resolve_conj().numpy()
        return _same_bits(np.asarray(got, want.dtype), want)
    want = np.zeros(shape, np.float32) if zero else expected.detach().double().numpy()
    return np.allclose(got, want, rtol=1e-4, atol=1e-4, equal_nan=True)


def _widen(value):
    if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
        return value.detach().double()
    if isinstance(value, (list, tuple)) and not hasattr(value, "_fields"):
        return type(value)(_widen(v) for v in value)
    return value


def _float64_forward(op, sample):
    try:
        out = op(_widen(sample.input), *_widen(list(sample.args)), **{k: _widen(v) for k, v in sample.kwargs.items()})
        return _widen(out)
    except Exception:
        return None


def _float64_grads(op, sample, weights):
    """Gradients of the same sample computed by PyTorch in float64, or None if unsupported."""
    def widen(value):
        if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
            return value.detach().double().requires_grad_(value.requires_grad)
        if isinstance(value, (list, tuple)) and not hasattr(value, "_fields"):
            return type(value)(widen(v) for v in value)
        return value
    try:
        args = widen([sample.input, *sample.args])
        kwargs = {k: widen(v) for k, v in sample.kwargs.items()}
        out = op(args[0], *args[1:], **kwargs)
        inputs = [t for t in leaves(args, torch.Tensor) if t.requires_grad and t.dtype == torch.float64]
        outputs = [t for t in leaves(out, torch.Tensor) if t.requires_grad and t.dtype == torch.float64]
        loss = sum((t * torch.from_numpy(w.astype(np.float64))).sum() for t, w in zip(outputs, weights))
        return torch.autograd.grad(loss, inputs, allow_unused=True)
    except Exception:
        return None


ORDER = {"missing": 0, "error": 1, "mismatch": 2, "forward": 3, "pass": 4}


def run(samples, only):
    results = {}
    signal.signal(signal.SIGALRM, _alarm)
    for op in op_db:
        key = op.name + (f".{op.variant_test_name}" if op.variant_test_name else "")
        if only and op.name not in only:
            continue
        if torch.float32 not in op.supported_dtypes("cpu"):
            continue
        status, reason = "pass", None
        try:
            signal.alarm(20)
            cases = list(op.sample_inputs("cpu", torch.float32, requires_grad=op.supports_autograd))[:samples]
            for sample in cases:
                fn = resolve(op, to_dense(sample.input))
                if fn is None:
                    status, reason = "missing", "no DenseCore name"
                    break
                try:
                    s, r = check_sample(op, fn, sample)
                except Timeout:
                    raise
                except BaseException as error:  # PanicException from Rust is a BaseException
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    s, r = "error", f"{type(error).__name__}: {str(error)[:120]}"
                    if isinstance(error, AttributeError) and "has no attribute" in str(error) and (
                            "densecore" in str(error) or "'Tensor'" in str(error)):
                        s, r = "missing", str(error)[:120]
                if ORDER[s] < ORDER[status]:
                    status, reason = s, r
                if status in ("error", "mismatch"):
                    break
        except Timeout:
            status, reason = "error", "timed out"
        except Exception as error:  # torch failed to build samples; not DenseCore's fault
            status, reason = "skipped", f"sample setup: {type(error).__name__}"
        finally:
            signal.alarm(0)
        results[key] = {"status": status, "reason": reason}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--only", default="")
    parser.add_argument("--exact", action="store_true", help="bit-for-bit comparison against PyTorch")
    args = parser.parse_args()
    global EXACT
    EXACT = args.exact
    results = run(args.samples, set(filter(None, args.only.split(","))))
    counts = {}
    for r in results.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    tested = sum(v for k, v in counts.items() if k != "skipped")
    print(f"ops tested: {tested}  " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: ORDER.get(kv[0], 9))))
    if not args.only:
        (ROOT / "benchmarks" / f"CONFORMANCE{'_EXACT' if EXACT else ''}.json").write_text(json.dumps(
            {"torch": torch.__version__, "samples_per_op": args.samples, "counts": counts, "ops": results},
            indent=1, sort_keys=True))
        lines = [f"# OpInfo conformance (generated by scripts/conformance.py)", "",
                 f"PyTorch {torch.__version__}, float32 CPU, up to {args.samples} samples per op"
                 + (", bit-for-bit." if EXACT else "."), "",
                 f"**{counts.get('pass', 0)} of {tested} ops pass.** "
                 + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: ORDER.get(kv[0], 9))), "",
                 "| Op | Status | Reason |", "|---|---|---|"]
        lines += [f"| `{k}` | {r['status']} | {r['reason'] or ''} |" for k, r in sorted(results.items())
                  if r["status"] not in ("missing", "skipped")]
        lines += ["", "Missing ops are listed in the local CONFORMANCE.json (git-ignored)."]
        (ROOT / "benchmarks" / f"CONFORMANCE{'_EXACT' if EXACT else ''}.md").write_text("\n".join(lines) + "\n")
    else:
        for key, r in results.items():
            print(f"  {r['status']:9s} {key}: {r['reason']}")


if __name__ == "__main__":
    main()
