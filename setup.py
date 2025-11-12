"""Minimal setup shim.

This module is intentionally small because packaging configuration is
specified in ``pyproject.toml``. The file exists for legacy tooling that
invokes ``setup.py``.
"""

# pylint: disable=missing-module-docstring

# !/usr/bin/env python3
#
# Please see pyproject.toml
#
# see: https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html

from setuptools import setup


setup()
