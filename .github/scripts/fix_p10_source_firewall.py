from pathlib import Path

modeling = Path("src/modeling/service.py")
text = modeling.read_text(encoding="utf-8")
old = '''from core.semantic_keys import (
    FIRM_ID,
'''
new = '''from core.semantic_keys import (
    ELIGIBLE,
    FIRM_ID,
'''
if text.count(old) != 1:
    raise RuntimeError("modeling semantic-key import anchor not found exactly once")
text = text.replace(old, new, 1)
old = 'and item.get("model_eligibility") in {None, "eligible"}'
new = 'and item.get("model_eligibility") in {None, ELIGIBLE}'
if text.count(old) != 1:
    raise RuntimeError("modeling eligibility status anchor not found exactly once")
modeling.write_text(text.replace(old, new, 1), encoding="utf-8")

nested = Path("src/selection/nested_refit.py")
text = nested.read_text(encoding="utf-8")
old = 'if item.get("model_eligibility") not in {None, "eligible"}:'
new = 'if item.get("model_eligibility") not in {None, ELIGIBLE}:'
if text.count(old) != 1:
    raise RuntimeError("nested-refit eligibility status anchor not found exactly once")
nested.write_text(text.replace(old, new, 1), encoding="utf-8")

print("Replaced physical-column literal with semantic ELIGIBLE key")
