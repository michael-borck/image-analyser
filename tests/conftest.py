"""Shared fixtures for image-analyser tests."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest

# pyzbar uses ctypes to load libzbar at import time. On macOS Apple Silicon,
# Homebrew installs to /opt/homebrew/lib, which is not in the default dynamic
# linker search path; on Intel macs it's /usr/local/lib. Set DYLD_LIBRARY_PATH
# before any test module triggers pyzbar import so a bare `pytest` works.
if platform.system() == "Darwin":
    for _candidate in ("/opt/homebrew/lib", "/usr/local/lib"):
        if Path(_candidate, "libzbar.dylib").exists():
            _existing = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = (
                f"{_candidate}:{_existing}" if _existing else _candidate
            )
            break

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR
