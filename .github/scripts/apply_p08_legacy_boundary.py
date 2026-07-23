from pathlib import Path


path = Path("scripts/p08_profiled_orchestrator.py")
text = path.read_text(encoding="utf-8")

text = text.replace("from typing import cast\n", "from typing import Protocol, cast\n", 1)
text = text.replace(
    "from core.pipeline import LoadedRun, load_run, mapping, sequence\n",
    "from core.pipeline import load_run, mapping, sequence\n",
    1,
)

marker = """class P08Job:
    scenario_id: str
    method_id: str
    scenario_key: str
    method_key: str
    minimum: int
    batch_size: int
    maximum: int


"""
protocols = marker + """class _StoreLike(Protocol):
    def inventory(self) -> list[dict[str, object]]: ...


class _ContextLike(Protocol):
    store: _StoreLike

    def read(self, artifact_id: str, coordinates: Mapping[str, str]) -> object: ...


class _LoadedLike(Protocol):
    context: _ContextLike


def _loaded_like(value: object) -> _LoadedLike:
    return cast(_LoadedLike, value)


"""
if marker not in text:
    raise SystemExit("P08Job insertion point not found")
text = text.replace(marker, protocols, 1)

old = """def _artifact_exists(
    loaded: LoadedRun,
    inventory: Sequence[Mapping[str, object]],
    artifact_id: str,
    coordinates: Mapping[str, str],
) -> bool:
    for item in inventory:
        if item.get("artifact_id") != artifact_id:
            continue
        if _coordinates(item) == dict(coordinates):
            loaded.context.read(artifact_id, coordinates)
            return True
    return False
"""
new = """def _artifact_exists(
    loaded: object,
    inventory: Sequence[Mapping[str, object]],
    artifact_id: str,
    coordinates: Mapping[str, str],
) -> bool:
    context = _loaded_like(loaded).context
    for item in inventory:
        if item.get("artifact_id") != artifact_id:
            continue
        if _coordinates(item) == dict(coordinates):
            context.read(artifact_id, coordinates)
            return True
    return False
"""
if old not in text:
    raise SystemExit("artifact helper block not found")
text = text.replace(old, new, 1)

old = """def _incomplete_batch_commands(
    loaded: LoadedRun,
    *,
    jobs: Sequence[P08Job],
    python: str,
    registry_path: Path,
    run_id: str,
) -> list[list[str]]:
    \"\"\"Return worker commands for every incomplete batch/diagnostics pair.\"\"\"
    inventory = loaded.context.store.inventory()
"""
new = """def _incomplete_batch_commands(
    loaded: object,
    *,
    jobs: Sequence[object],
    python: str,
    registry_path: Path,
    run_id: str,
) -> list[list[str]]:
    \"\"\"Return worker commands for every incomplete batch/diagnostics pair.\"\"\"
    inventory = _loaded_like(loaded).context.store.inventory()
    normalized_jobs = [_coerce_job(value) for value in jobs]
"""
if old not in text:
    raise SystemExit("incomplete command signature not found")
text = text.replace(old, new, 1)
text = text.replace(
    "    jobs_by_keys = {(job.scenario_key, job.method_key): job for job in jobs}\n",
    "    jobs_by_keys = {(job.scenario_key, job.method_key): job for job in normalized_jobs}\n",
    1,
)

old = """def _batch_exists(
    loaded: LoadedRun,
    *,
    inventory: Sequence[Mapping[str, object]],
    scenario_key: str,
    method_key: str,
    start: int,
    batch_size: int,
) -> bool:
    batch_key = f"b{start // batch_size:04d}"
    coordinates = {
        SCENARIO_KEY: scenario_key,
        METHOD_KEY: method_key,
        BATCH_KEY: batch_key,
    }
    batch_written = _artifact_exists(loaded, inventory, "simulation_batches", coordinates)
    diagnostics_written = _artifact_exists(loaded, inventory, "model_diagnostics", coordinates)
    return batch_written and diagnostics_written
"""
new = """def _batch_exists(
    loaded: object,
    *,
    inventory: Sequence[Mapping[str, object]] | None = None,
    scenario_key: str,
    method_key: str,
    start: int,
    batch_size: int,
) -> bool:
    batch_key = f"b{start // batch_size:04d}"
    coordinates = {
        SCENARIO_KEY: scenario_key,
        METHOD_KEY: method_key,
        BATCH_KEY: batch_key,
    }
    available = (
        inventory
        if inventory is not None
        else _loaded_like(loaded).context.store.inventory()
    )
    batch_written = _artifact_exists(loaded, available, "simulation_batches", coordinates)
    diagnostics_written = _artifact_exists(loaded, available, "model_diagnostics", coordinates)
    return batch_written and diagnostics_written
"""
if old not in text:
    raise SystemExit("batch exists block not found")
text = text.replace(old, new, 1)

helper_marker = "\ndef _coordinates(item: Mapping[str, object]) -> dict[str, str]:\n"
helpers = """
def _coerce_job(value: object) -> P08Job:
    if isinstance(value, P08Job):
        return value
    if not isinstance(value, dict):
        raise ValueError("P08 job must be a P08Job or mapping")
    raw = cast(dict[str, object], value)
    job = P08Job(
        scenario_id=_required_text(raw.get(SCENARIO_ID), SCENARIO_ID),
        method_id=_required_text(raw.get(METHOD_ID), METHOD_ID),
        scenario_key=_required_text(raw.get(SCENARIO_KEY), SCENARIO_KEY),
        method_key=_required_text(raw.get(METHOD_KEY), METHOD_KEY),
        minimum=_required_positive_integer(raw.get("minimum"), "minimum"),
        batch_size=_required_positive_integer(raw.get("batch_size"), "batch_size"),
        maximum=_required_positive_integer(raw.get("maximum"), "maximum"),
    )
    if job.maximum < job.minimum:
        raise ValueError("P08 job maximum must be at least minimum")
    return job


def _required_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"P08 job {context}: nonempty string required")
    return value


def _required_positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"P08 job {context}: positive integer required")
    return value


"""
if helper_marker not in text:
    raise SystemExit("coordinate helper insertion point not found")
text = text.replace(helper_marker, helpers + helper_marker, 1)

path.write_text(text, encoding="utf-8")
