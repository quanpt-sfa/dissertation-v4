from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()

# Alignment audit must resolve methodology modules through config/pipeline.yaml,
# not hard-code source-config paths.
p = root / "src" / "chapter3_v40" / "alignment.py"
s = p.read_text(encoding="utf-8")
s = s.replace(
'''        "config/foundation/steps.yaml",\n        "config/methodology/measurement.yaml",\n        "config/methodology/evaluation.yaml",\n        "config/methodology/inference.yaml",\n''',
'',
)
s = s.replace(
'''    steps = _yaml(root / "config/foundation/steps.yaml").get("steps", {})\n    measurement = _yaml(root / "config/methodology/measurement.yaml").get("measurement", {})\n    evaluation = _yaml(root / "config/methodology/evaluation.yaml").get("evaluation", {})\n    inference = _yaml(root / "config/methodology/inference.yaml").get("inference", {})\n''',
'''    pipeline = _yaml(root / "config" / "pipeline.yaml")\n    modules = pipeline.get("modules", {})\n    if not isinstance(modules, dict):\n        raise ValueError("pipeline modules registry must be a mapping")\n\n    def module_yaml(module_id: str) -> dict[str, Any]:\n        relative = modules.get(module_id)\n        if not isinstance(relative, str) or not relative:\n            raise ValueError(f"pipeline module is not registered: {module_id}")\n        return _yaml(root / "config" / relative)\n\n    steps = module_yaml("steps").get("steps", {})\n    measurement = module_yaml("measurement").get("measurement", {})\n    evaluation = module_yaml("evaluation").get("evaluation", {})\n    inference = module_yaml("inference").get("inference", {})\n''',
)
p.write_text(s, encoding="utf-8")

# Chapter-4 adapter must use semantic-key constants rather than copied physical
# column-name literals.
p = root / "scripts" / "build_chapter4_v40_inputs.py"
s = p.read_text(encoding="utf-8")
needle = "from chapter3_v40.result_ledger import ResultRecord, validate_result_ledger\n"
replacement = needle + "from core.semantic_keys import LEARNER_ID, MEASUREMENT_ID, TARGET_ID\n"
if "from core.semantic_keys import LEARNER_ID" not in s:
    s = s.replace(needle, replacement)
s = s.replace('row.get("learner_id", row.get("model_id", f"model_{index}"))', 'row.get(LEARNER_ID, row.get("model_id", f"model_{index}"))')
s = s.replace('row.get("target_id", "observed_target")', 'row.get(TARGET_ID, "observed_target")')
s = s.replace('        "measurement_id",\n', '        MEASUREMENT_ID,\n')
p.write_text(s, encoding="utf-8")
