from pathlib import Path


path = Path("config/execution/simulation_mcse.yaml")
text = path.read_text(encoding="utf-8")
bad = """- policy_id: descriptive_imbalance_treatment_applied
  priority: 20
  metric_pattern: imbalance_treatment_applied
  role: descriptive
  mcse_gate_required: false
  undefined_policy: report_only
  target_rule: none
  minimum_finite_fraction: 0.0
"""
good = """  - policy_id: descriptive_imbalance_treatment_applied
    priority: 20
    metric_pattern: imbalance_treatment_applied
    role: descriptive
    mcse_gate_required: false
    undefined_policy: report_only
    target_rule: none
    minimum_finite_fraction: 0.0
"""
if bad not in text:
    raise SystemExit("Temporary P08 policy block not found")
path.write_text(text.replace(bad, good, 1), encoding="utf-8")
