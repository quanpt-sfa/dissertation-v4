from pathlib import Path

path = Path("scripts/p10_select_measurement.py")
text = path.read_text(encoding="utf-8")
old = '''        random_state=derive_seed(
            loaded.protocol_hash, coordinates={OUTER_FOLD: args.outer_fold}, step_id="P10", purpose="nested_channel_refit"
        )
'''
new = '''        random_state=derive_seed(
            loaded.protocol_hash,
            "P10",
            {OUTER_FOLD: args.outer_fold},
            "nested_channel_refit",
        )
'''
if text.count(old) != 1:
    raise RuntimeError("generated P10 seed call was not found exactly once")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
