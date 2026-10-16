"""Shared test configuration.

`mac_exact` tests compare DenseCore with the installed PyTorch bit for bit. DenseCore copies
PyTorch's arithmetic on Apple-silicon macOS (Accelerate BLAS, Apple libm, clang's
contraction rules); PyTorch itself gives different bits on Linux and x86, so these tests run
only on that platform. Every other test runs everywhere.
"""
import platform
import sys

import pytest

MAC_ARM = sys.platform == "darwin" and platform.machine() == "arm64"


def pytest_configure(config):
    config.addinivalue_line("markers", "mac_exact: bit-exact vs PyTorch; defined on Apple-silicon macOS only")


def pytest_collection_modifyitems(config, items):
    if MAC_ARM:
        return
    skip = pytest.mark.skip(reason="bit-exact vs PyTorch is defined for Apple-silicon macOS; "
                                   "PyTorch's own bits differ on this platform")
    for item in items:
        if "mac_exact" in item.keywords:
            item.add_marker(skip)
