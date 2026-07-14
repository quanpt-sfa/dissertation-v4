"""P10 CLI: fold-specific development-only measurement selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import load_run, mapping, sequence
from core.semantic_keys import OUTER_FOLD
from selection.service import select_measurement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--outer-fold", required=True)
    args = parser.parse_args()
    coordinates = {OUTER_FOLD: args.outer_fold}
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P10",
        state="SPLIT",
        coordinates=coordinates,
    )
    matrices = mapping(loaded.context.read("source_channel_matrices", {}), "source matrices")
    capability = mapping(loaded.context.read("l3_pilot_capability", {}), "L3 capability")
    loaded.context.read("temporal_split_registry", {})
    loaded.context.read("channel_time_split_registry", {})
    loaded.context.read("fold_aware_weights", {"fold_id": args.outer_fold})
    diagnostics = mapping(
        loaded.context.read("weight_diagnostics", {"fold_id": args.outer_fold}),
        "weight diagnostics",
    )
    if diagnostics.get("fit_scope") != "development_history":
        raise ValueError("P10 requires development-history weight diagnostics")
    loaded.context.read("mcse_report", {})
    loaded.context.read("fold_eligibility", {})
    measurement = mapping(loaded.registry.get("measurement"), "measurement")
    raw_candidates = sequence(
        measurement.get("selection_candidates"), "measurement.selection_candidates"
    )
    candidates = [str(value) for value in raw_candidates]
    result = select_measurement(
        matrices=matrices,
        outer_year=int(args.outer_fold),
        candidates=candidates,
        l3_capability=capability,
    )
    loaded.context.write("measurement_candidate_results", result.candidates, coordinates)
    loaded.context.write("measurement_selection_registry", result.selection, coordinates)
    loaded.context.write("channel_measurement_selection", result.channel_selection, coordinates)
    print(
        f"P10 status=PASS fold={args.outer_fold} selected={result.selection['selected_measurement']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
