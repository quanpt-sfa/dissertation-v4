"""Compatibility wrapper for the historical P02 module name."""

from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("p02_build_firm_panel", run_name="__main__")
