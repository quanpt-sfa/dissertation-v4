from pathlib import Path

path = Path("src/selection/nested_refit.py")
text = path.read_text(encoding="utf-8")

old = '''    removed = receipt.get("heldout_channel_removed_from")
    if not isinstance(removed, list) or set(str(value) for value in removed) != set(
        _REMOVED_INPUTS
    ):
        raise RuntimeError("NESTED_CHANNEL_REFIT_REMOVAL_CONTRACT_INVALID")
'''
new = '''    removed = receipt.get("heldout_channel_removed_from")
    if not isinstance(removed, list):
        raise RuntimeError("NESTED_CHANNEL_REFIT_REMOVAL_CONTRACT_INVALID")
    removed_values = [str(value) for value in cast(list[object], removed)]
    if set(removed_values) != set(_REMOVED_INPUTS):
        raise RuntimeError("NESTED_CHANNEL_REFIT_REMOVAL_CONTRACT_INVALID")
'''
if text.count(old) != 1:
    raise RuntimeError("nested receipt removal block not found exactly once")
text = text.replace(old, new, 1)

old = '''def _string_object_dict(raw: dict[object, object], context: str) -> dict[str, object]:
    result = {str(key): value for key, value in raw.items()}
    if len(result) != len(raw):
        raise ValueError(f"{context}: keys collide after string normalization")
    return result
'''
new = '''def _string_object_dict(raw: object, context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context}: object mapping required")
    typed = cast(dict[object, object], raw)
    result = {str(key): value for key, value in typed.items()}
    if len(result) != len(typed):
        raise ValueError(f"{context}: keys collide after string normalization")
    return result
'''
if text.count(old) != 1:
    raise RuntimeError("string-object dictionary helper not found exactly once")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("Applied remaining nested-refit type boundaries")
