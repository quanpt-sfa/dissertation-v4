from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: replacement count={text.count(old)} for {old[:70]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


path = Path("src/selection/nested_refit.py")

replace_once(
    path,
    '''        objectives = [
            float(item["soft_cross_entropy"])
            for item in candidate_cells
            if item.get("soft_cross_entropy") is not None
        ]
''',
    '''        objectives: list[float] = []
        for item in candidate_cells:
            objective = item.get("soft_cross_entropy")
            if (
                isinstance(objective, (int, float))
                and not isinstance(objective, bool)
                and math.isfinite(float(objective))
            ):
                objectives.append(float(objective))
''',
)

replace_once(
    path,
    '''    raw = channel_selection.get("nested_refit_receipt")
    if not isinstance(raw, Mapping):
        raise RuntimeError("NESTED_CHANNEL_REFIT_RECEIPT_MISSING")
    receipt = {str(key): value for key, value in raw.items()}
''',
    '''    raw = channel_selection.get("nested_refit_receipt")
    if not isinstance(raw, dict):
        raise RuntimeError("NESTED_CHANNEL_REFIT_RECEIPT_MISSING")
    receipt = _string_object_dict(raw, "nested-refit receipt")
''',
)

replace_once(
    path,
    '''    candidate_rows = [
        cast(Mapping[str, object], item)
        for item in candidate_rows_raw
        if isinstance(item, Mapping)
    ]
''',
    '''    candidate_rows = [
        _string_object_dict(item, "nested-refit candidate")
        for item in cast(list[object], candidate_rows_raw)
        if isinstance(item, dict)
    ]
''',
)

replace_once(
    path,
    '''    target_rows = pd.DataFrame(
        [cast(dict[str, object], item) for item in target_rows_raw if isinstance(item, dict)]
    )
''',
    '''    target_rows = pd.DataFrame(
        [
            _string_object_dict(item, "OOF training target")
            for item in cast(list[object], target_rows_raw)
            if isinstance(item, dict)
        ]
    )
''',
)

replace_once(
    path,
    '''        parsed = [
            cast(Mapping[str, object], item)
            for item in raw_targets
            if isinstance(item, Mapping)
            and str(item.get(_HELDOUT_CHANNEL)) == heldout_channel
            and isinstance(item.get(FISCAL_YEAR), int)
            and int(cast(int, item[FISCAL_YEAR])) < outer_year
        ]
''',
    '''        parsed: list[dict[str, object]] = []
        for raw_target in cast(list[object], raw_targets):
            if not isinstance(raw_target, dict):
                continue
            item = _string_object_dict(raw_target, "heldout L3 target")
            row_year = item.get(FISCAL_YEAR)
            if (
                str(item.get(_HELDOUT_CHANNEL)) == heldout_channel
                and isinstance(row_year, int)
                and not isinstance(row_year, bool)
                and row_year < outer_year
            ):
                parsed.append(item)
''',
)

replace_once(
    path,
    '''        raw_scores = row.get("channel_evidence_scores")
        if not isinstance(raw_scores, Mapping):
            continue
        remaining = [
            float(value)
            for channel, value in raw_scores.items()
''',
    '''        raw_scores = row.get("channel_evidence_scores")
        if not isinstance(raw_scores, dict):
            continue
        scores = _string_object_dict(raw_scores, "channel evidence scores")
        remaining = [
            float(value)
            for channel, value in scores.items()
''',
)

replace_once(
    path,
    '''        raw_outcomes = row.get("channel_outcomes")
        if not isinstance(raw_outcomes, Mapping):
            continue
        value = raw_outcomes.get(heldout_channel)
''',
    '''        raw_outcomes = row.get("channel_outcomes")
        if not isinstance(raw_outcomes, dict):
            continue
        outcomes = _string_object_dict(raw_outcomes, "channel outcomes")
        value = outcomes.get(heldout_channel)
''',
)

replace_once(
    path,
    '''def _matrix_rows(matrices: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = matrices.get("rows")
    if not isinstance(raw, list):
        raise ValueError("nested channel refit requires matrix rows")
    return [
        cast(Mapping[str, object], item) for item in raw if isinstance(item, Mapping)
    ]


def _string_list(raw: object, context: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"nested channel refit requires {context}")
    values = [str(value) for value in raw]
''',
    '''def _matrix_rows(matrices: Mapping[str, object]) -> list[dict[str, object]]:
    raw = matrices.get("rows")
    if not isinstance(raw, list):
        raise ValueError("nested channel refit requires matrix rows")
    return [
        _string_object_dict(item, "source-channel matrix row")
        for item in cast(list[object], raw)
        if isinstance(item, dict)
    ]


def _string_list(raw: object, context: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"nested channel refit requires {context}")
    values = [str(value) for value in cast(list[object], raw)]
''',
)

replace_once(
    path,
    '''def _physical(columns: Mapping[str, str], key: str) -> str:
''',
    '''def _string_object_dict(raw: dict[object, object], context: str) -> dict[str, object]:
    result = {str(key): value for key, value in raw.items()}
    if len(result) != len(raw):
        raise ValueError(f"{context}: keys collide after string normalization")
    return result


def _physical(columns: Mapping[str, str], key: str) -> str:
''',
)

# Keep the fixture return type invariant under list invariance.
test_path = Path("tests/stages/test_p10_nested_refit.py")
replace_once(
    test_path,
    '''    registry = [
        {"feature_id": "feature_s1", "role": "content", "source_channel": "S1"},
''',
    '''    registry: list[dict[str, object]] = [
        {"feature_id": "feature_s1", "role": "content", "source_channel": "S1"},
''',
)

print("Applied strict typing fixes for nested refit")
