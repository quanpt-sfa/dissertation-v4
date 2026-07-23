from pathlib import Path

path = Path("src/selection/nested_refit.py")
text = path.read_text(encoding="utf-8")
replacements = {
    '_string_object_dict(raw, "nested-refit receipt")': '_string_object_dict(cast(object, raw), "nested-refit receipt")',
    '_string_object_dict(item, "nested-refit candidate")': '_string_object_dict(cast(object, item), "nested-refit candidate")',
    '_string_object_dict(item, "OOF training target")': '_string_object_dict(cast(object, item), "OOF training target")',
    '_string_object_dict(raw_target, "heldout L3 target")': '_string_object_dict(cast(object, raw_target), "heldout L3 target")',
    '_string_object_dict(raw_scores, "channel evidence scores")': '_string_object_dict(cast(object, raw_scores), "channel evidence scores")',
    '_string_object_dict(raw_outcomes, "channel outcomes")': '_string_object_dict(cast(object, raw_outcomes), "channel outcomes")',
    '_string_object_dict(item, "source-channel matrix row")': '_string_object_dict(cast(object, item), "source-channel matrix row")',
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence of {old!r}, found {count}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print("Applied final Unknown-to-object casts at JSON boundaries")
