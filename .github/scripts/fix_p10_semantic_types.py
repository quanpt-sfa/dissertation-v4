from pathlib import Path

path = Path("scripts/p10_select_measurement.py")
text = path.read_text(encoding="utf-8")
old = '''    nested_cells = nested.receipt.get("cell_results")
    if not isinstance(nested_cells, list):
        raise RuntimeError("P10_NESTED_REFIT_CELL_RESULTS_MISSING")
    proxy_channel_diagnostics = result.channel_selection.get("strict_channel_results", [])
    channel_selection = {
        **dict(result.channel_selection),
        "proxy_only_channel_diagnostics": proxy_channel_diagnostics,
        "strict_channel_results": nested_cells,
        "nested_refit_receipt": nested.receipt,
    }
'''
new = '''    nested_cells_raw = nested.receipt.get("cell_results")
    if not isinstance(nested_cells_raw, list):
        raise RuntimeError("P10_NESTED_REFIT_CELL_RESULTS_MISSING")
    nested_cells = cast(list[object], nested_cells_raw)
    proxy_raw = result.channel_selection.get("strict_channel_results", [])
    proxy_channel_diagnostics = (
        cast(list[object], proxy_raw) if isinstance(proxy_raw, list) else []
    )
    channel_selection: dict[str, object] = {
        str(key): cast(object, value) for key, value in result.channel_selection.items()
    }
    channel_selection["proxy_only_channel_diagnostics"] = proxy_channel_diagnostics
    channel_selection["strict_channel_results"] = nested_cells
    channel_selection["nested_refit_receipt"] = nested.receipt
'''
if text.count(old) != 1:
    raise RuntimeError(f"P10 semantic channel-selection block count={text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("Applied P10 semantic typing boundary")
