"""Stable cryptographic seed derivation with local NumPy generators only."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np


def derive_seed(
    protocol_hash: str,
    step_id: str,
    coordinates: Mapping[str, str],
    unit_id: str,
) -> int:
    payload = json.dumps(
        {
            "protocol_hash": protocol_hash,
            "step_id": step_id,
            "coordinates": dict(sorted(coordinates.items())),
            "unit_id": unit_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def generator(
    protocol_hash: str,
    step_id: str,
    coordinates: Mapping[str, str],
    unit_id: str,
) -> np.random.Generator:
    return np.random.default_rng(derive_seed(protocol_hash, step_id, coordinates, unit_id))
