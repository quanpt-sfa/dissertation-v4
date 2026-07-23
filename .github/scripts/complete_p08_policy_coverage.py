from pathlib import Path


path = Path("config/execution/simulation_mcse.yaml")
text = path.read_text(encoding="utf-8")

for policy_id in (
    "descriptive_training_and_sampling",
    "descriptive_resampling",
    "descriptive_empirical_context",
):
    old = f"  - policy_id: {policy_id}\n    priority: 20\n"
    new = f"  - policy_id: {policy_id}\n    priority: 30\n"
    if old not in text:
        raise SystemExit(f"P08 policy priority target not found: {policy_id}")
    text = text.replace(old, new, 1)

marker = "  - policy_id: descriptive_latent_probability_mean\n"
policy = """  - policy_id: descriptive_realized_prevalence
    priority: 30
    metric_pattern: realized_prevalence
    role: descriptive
    mcse_gate_required: false
    undefined_policy: report_only
    target_rule: none
    minimum_finite_fraction: 0.0
"""
if policy not in text:
    if marker not in text:
        raise SystemExit("P08 realized-prevalence insertion point not found")
    text = text.replace(marker, policy + marker, 1)

path.write_text(text, encoding="utf-8")
