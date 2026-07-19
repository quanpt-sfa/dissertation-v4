"""Production-registry tests for the sequential primary track and S1 rules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

from core.evidence_registry import logical_evidence_sources
from core.fold_control import require_primary_target
from core.registry_compiler import compile_registry
from evidence.annual import AdjustmentRow, build_a