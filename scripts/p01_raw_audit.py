"""Compatibility wrapper for the P01 module name already registered in steps.yaml."""

from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("p01_audit_raw", run_name="__main__")
