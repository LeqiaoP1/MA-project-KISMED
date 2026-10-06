"""Pytest bootstrap for the ``code/`` test suite.

The runtime code assumes the CWD is ``code/`` (top-level packages ``core``,
``data``, ``engines``, ``models``, ``runners``, ``utils``). Collection happens
before any test runs, so ``code/`` is put on ``sys.path`` here rather than
inside a test module -- that keeps each test file importable on its own.

Run the suite from ``code/``::

    python -m pytest
    python -m pytest tests -q
"""
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import pytest  # noqa: E402  (must follow the sys.path bootstrap above)


@pytest.fixture(scope='session')
def code_dir() -> str:
    """Absolute path of ``code/`` -- use it to build config paths."""
    return CODE_DIR


@pytest.fixture
def write_yaml(tmp_path):
    """Return ``write(body) -> absolute path`` for a throwaway YAML config.

    Each call overwrites ``<tmp_path>/cfg.yaml``; within a single test that is
    what the tests want (write the body, then parse it).
    """
    def _write(body: str, name: str = 'cfg.yaml') -> str:
        path = tmp_path / name
        path.write_text(body)
        return str(path)

    return _write
