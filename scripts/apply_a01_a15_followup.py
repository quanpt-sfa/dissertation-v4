#!/usr/bin/env python3
"""Complete the one-time A-01--A-15 remediation after the primary patch."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace(path: str, old: str, new: str) -> None:
    text = read(path)
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"{path}: follow-up anchor not found")
    write(path, text.replace(old, new, 1))


def repair_snapshot_helper_position() -> None:
    path = "src/snapshot/builder.py"
    text = read(path)
    declaration = "def write_snapshot(path: Path, snapshot: dict[str, object]) -> None:\n"
    helper = "def _load_extract_provenance(\n"
    bad = declaration + "\n\n" + helper
    if bad not in text:
        return
    helper_start = text.index(helper)
    body_start = text.index("    data_sources = registry.get", helper_start)
    # The helper ends at the first blank line immediately before the original
    # write_snapshot body, which starts with path = path.resolve().
    write_body = text.index("    path = path.resolve()", body_start)
    helper_text = text[helper_start:write_body]
    text = text[: text.index(declaration)] + helper_text + "\n\n" + declaration + text[write_body:]
    write(path, text)


def wire_gate3_controls() -> None:
    path = "src/gates/service.py"
    replace(
        path,
        "directions.append(_interaction_coefficient(frame, monitoring, outcome))",
        "directions.append(_interaction_coefficient(frame, monitoring, outcome, gate))",
    )
    replace(
        path,
        "point, side = _breakpoint(fold_frame, monitoring, outcome)",
        "point, side = _breakpoint(fold_frame, monitoring, outcome, gate)",
    )
    replace(
        path,
        "point, _ = _breakpoint(domain_frame, monitoring, outcome)",
        "point, _ = _breakpoint(domain_frame, monitoring, outcome, gate)",
    )
    replace(
        path,
        "def _interaction_coefficient(frame: pd.DataFrame, monitoring: str, outcome: str) -> float:\n",
        "def _interaction_coefficient(\n"
        "    frame: pd.DataFrame, monitoring: str, outcome: str, gate: dict[str, Any]\n"
        ") -> float:\n",
    )
    replace(
        path,
        "    model = LogisticRegression(C=1e6, solver=\"lbfgs\", max_iter=2000, fit_intercept=False)\n",
        "    controls = _mapping(gate.get(\"logistic_fit\"))\n"
        "    model = LogisticRegression(\n"
        "        C=float(controls[\"inverse_regularization\"]),\n"
        "        solver=\"lbfgs\",\n"
        "        max_iter=int(controls[\"maximum_iterations\"]),\n"
        "        fit_intercept=False,\n"
        "    )\n",
    )
    replace(
        path,
        "def _breakpoint(frame: pd.DataFrame, monitoring: str, outcome: str) -> tuple[float | None, float]:\n",
        "def _breakpoint(\n"
        "    frame: pd.DataFrame, monitoring: str, outcome: str, gate: dict[str, Any]\n"
        ") -> tuple[float | None, float]:\n",
    )
    replace(
        path,
        "    candidates = np.unique(np.quantile(pressure, np.linspace(0.1, 0.9, 17)))\n",
        "    grid = _mapping(gate.get(\"breakpoint_grid\"))\n"
        "    candidates = np.unique(\n"
        "        np.quantile(\n"
        "            pressure,\n"
        "            np.linspace(\n"
        "                float(grid[\"lower_quantile\"]),\n"
        "                float(grid[\"upper_quantile\"]),\n"
        "                int(grid[\"points\"]),\n"
        "            ),\n"
        "        )\n"
        "    )\n",
    )
    # The second hard-coded LogisticRegression occurrence belongs to _breakpoint.
    text = read(path)
    old = "        model = LogisticRegression(C=1e6, solver=\"lbfgs\", max_iter=2000, fit_intercept=False)\n"
    if old in text:
        new = (
            "        controls = _mapping(gate.get(\"logistic_fit\"))\n"
            "        model = LogisticRegression(\n"
            "            C=float(controls[\"inverse_regularization\"]),\n"
            "            solver=\"lbfgs\",\n"
            "            max_iter=int(controls[\"maximum_iterations\"]),\n"
            "            fit_intercept=False,\n"
            "        )\n"
        )
        write(path, text.replace(old, new, 1))


def retain_known_case_api_compatibility() -> None:
    replace(
        "src/known_cases/service.py",
        "    strong_percentile: float,\n    weak_percentile: float,\n    columns: dict[str, str],\n",
        "    strong_percentile: float,\n    columns: dict[str, str],\n    weak_percentile: float = 0.5,\n",
    )


def main() -> None:
    repair_snapshot_helper_position()
    wire_gate3_controls()
    retain_known_case_api_compatibility()


if __name__ == "__main__":
    main()
