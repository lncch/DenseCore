"""One-command demo: nanoGPT trains bit-for-bit identically in PyTorch and DenseCore.

Usage (from the repository root, with DenseCore built and PyTorch 2.14.1 installed):
    PYTHONPATH=python python scripts/demo_bit_exact.py [steps] [dropout]

Downloads Karpathy's nanoGPT model.py at a pinned commit and the tiny-Shakespeare text,
builds train.bin the way nanoGPT's data/shakespeare_char/prepare.py does, then runs
scripts/nanogpt_exact.py: both frameworks start from torch.manual_seed(1337), train on the
same batches, and every loss, gradient and weight is compared bit for bit.
"""
import hashlib
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request

import numpy as np

NANOGPT_COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
FILES = {
    "model.py": (f"https://raw.githubusercontent.com/karpathy/nanoGPT/{NANOGPT_COMMIT}/model.py",
                 "291c6ed8bb3ddf116fda2b90eff338cc4a7fde32"),
    "data/shakespeare_char/input.txt": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "04486597058d11dcc2c556b1d0433891eb639d2e"),
}
TRAIN_BIN_SHA1 = "2ea3c5352f6ec36d6aac2e1201113551079bd99f"
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "densecore-demo", "nanoGPT")


def sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def download(url):
    """urllib, with certifi's CA bundle when present; curl if Python has no CA certificates
    (the python.org macOS build before "Install Certificates.command")."""
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    try:
        with urllib.request.urlopen(url, timeout=60, context=context) as response:
            return response.read()
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLError):
            raise
        return subprocess.run(["curl", "-sfL", url], check=True, capture_output=True).stdout


def fetch(relative, url, expected):
    path = os.path.join(CACHE, relative)
    if not (os.path.exists(path) and sha1(path) == expected):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {url}", flush=True)
        data = download(url)
        if hashlib.sha1(data).hexdigest() != expected:
            sys.exit(f"{relative}: unexpected content (sha1 {hashlib.sha1(data).hexdigest()})")
        with open(path, "wb") as f:
            f.write(data)
    return path


def build_train_bin(text_path):
    """nanoGPT's prepare.py: sorted character vocabulary, first 90% as uint16 ids."""
    path = os.path.join(os.path.dirname(text_path), "train.bin")
    if not (os.path.exists(path) and sha1(path) == TRAIN_BIN_SHA1):
        with open(text_path, "r") as f:
            data = f.read()
        stoi = {ch: i for i, ch in enumerate(sorted(set(data)))}
        np.array([stoi[c] for c in data[:int(len(data) * 0.9)]], dtype=np.uint16).tofile(path)
        if sha1(path) != TRAIN_BIN_SHA1:
            sys.exit("train.bin does not match nanoGPT's prepare.py output")


def main():
    steps = sys.argv[1] if len(sys.argv) > 1 else "50"
    dropout = sys.argv[2] if len(sys.argv) > 2 else "0.2"
    for relative, (url, expected) in FILES.items():
        fetch(relative, url, expected)
    build_train_bin(os.path.join(CACHE, "data/shakespeare_char/input.txt"))
    here = os.path.dirname(os.path.abspath(__file__))
    command = [sys.executable, os.path.join(here, "nanogpt_exact.py"), CACHE, steps, dropout]
    sys.exit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
