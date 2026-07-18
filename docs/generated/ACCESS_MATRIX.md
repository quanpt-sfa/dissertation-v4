GENERATED FILE — DO NOT EDIT
Source: config/pipeline.yaml

# Access matrix

```json
{
  "P00": {
    "allowed_next_states": [
      "LOCKED"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "CONFIGURED"
    ],
    "read_sensitivity_classes": [],
    "reads": [],
    "required_receipts": [],
    "writes": [
      "registry_lock",
      "protocol_hash",
      "source_config_manifest",
      "capability_seed",
      "decision_traceability",
      "artifact_catalog",
      "schema_catalog",
      "step_catalog",
      "access_matrix",
      "known_cases_seal",
      "environment_expectation",
      "environment_observation",
      "p00_audit_report",
      "job_manifest",
      "success_receipt"
    ]
  },
  "P01": {
    "allowed_next_states": [
      "AUDITED"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "LOCKED"
    ],
    "read_sensitivity_classes": [],
    "reads": [],
    "required_receipts": [],
    "writes": [
      "raw_audit"
    ]
  },
  "P02": {
    "allowed_next_states": [
      "PANELLED"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "AUDITED"
    ],
    "read_sensitivity_classes": [
      "restricted"
    ],
    "reads": [
      "raw_audit"
    ],
    "required_receipts": [],
    "writes": [
      "firm_master",
      "firm_year_panel",
      "duplicate_map"
    ]
  },
  "P03": {
    "allowed_next_states": [
      "LEDGERED"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "PANELLED"
    ],
    "read_sensitivity_classes": [
      "restricted"
    ],
    "reads": [
      "firm_year_panel",
      "raw_audit"
    ],
    "required_receipts": [],
    "writes": [
      "evidence_ledger",
      "sanction_decision_ledger",
      "availability_registry",
      "lag_decomposition",
      "annual_evidence_audit"
    ]
  },
  "P04": {
    "allowed_next_states": [
      "RISK_SET"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "LEDGERED"
    ],
    "read_sensitivity_classes": [
      "restricted"
    ],
    "reads": [
      "firm_year_panel",
      "evidence_ledger"
    ],
    "required_receipts": [],
    "writes": [
      "risk_sets",
      "maturity_audit",
      "prospective_set",
      "censoring_registry"
    ]
  },
  "P05": {
    "allowed_next_states": [
      "MEASURED"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "RISK_SET"
    ],
    "read_sensitivity_classes": [
      "restricted"
    ],
    "reads": [
      "risk_sets",
      "evidence_ledger"
    ],
    "required_receipts": [],
    "writes": [
      "source_channel_matrices",
      "l0_l1_inputs",
      "l3_pilot_capability",
      "measurement_variable_registry",
      "channel_capability",
      "anchor_capability",
      "fold_eligibility",
      "sealed_outcome_store"
    ]
  },
  "P06": {
    "allowed_next_states": [
      "OBSERVABLE"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "MEASURED"
    ],
    "read_sensitivity_classes": [
      "restricted"
    ],
    "reads": [
      "l3_pilot_capability",
      "source_channel_matrices"
    ],
    "required_receipts": [],
    "writes": [
      "observability_registry"
    ]
  },
  "P07": {
    "allowed_next_states": [
      "FEATURED"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "OBSERVABLE"
    ],
    "read_sensitivity_classes": [
      "restricted"
    ],
    "reads": [
      "observability_registry",
      "firm_year_panel",
      "risk_sets",
      "raw_audit"
    ],
    "required_receipts": [],
    "writes": [
      "feature_panel",
      "feature_registry",
      "leakage_registry",
      "feature_registry_table",
      "feature_registry_csv",
      "feature_lineage_registry",
      "leakage_registry_table",
      "leakage_registry_csv",
      "feature_views",
      "feature_view_matrix",
      "feature_availability_audit",
      "feature_missingness_audit",
      "feature_panel_schema",
      "p07_summary",
      "p07_decision_report",
      "feature_store_manifest_validated",
      "feature_store_validation_report",
      "feature_store_file_audit",
      "feature_store_identifier_crosswalk_audit",
      "feature_store_availability_violations",
      "feature_store_coverage_audit",
      "feature_store_research_decision_audit",
      "feature_value_diagnostic_audit",
      "accounting_identity_audit",
      "audited_unaudited_adjustment_audit",
      "ratio_diagnostic_audit",
      "temporal_feature_audit",
      "feature_redundancy_audit"
    ]
  },
  "P08": {
    "allowed_next_states": [
      "SIMULATED"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "FEATURED"
    ],
    "read_sensitivity_classes": [
      "public",
      "restricted"
    ],
    "reads": [
      "firm_year_panel",
      "feature_panel",
      "feature_registry",
      "source_channel_matrices",
      "simulation_scenario_registry",
      "simulation_batches"
    ],
    "required_receipts": [],
    "writes": [
      "simulation_scenario_registry",
      "simulation_batches",
      "mcse_report"
    ]
  },
  "P09": {
    "allowed_next_states": [
      "SPLIT"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "SIMULATED"
    ],
    "read_sensitivity_classes": [
      "restricted"
    ],
    "reads": [
      "feature_panel",
      "feature_registry",
      "risk_sets",
      "observability_registry",
      "source_channel_matrices",
      "fold_eligibility"
    ],
    "required_receipts": [],
    "writes": [
      "temporal_split_registry",
      "channel_time_split_registry",
      "fold_aware_weights",
      "weight_diagnostics"
    ]
  },
  "P10": {
    "allowed_next_states": [
      "SELECTED"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "SPLIT"
    ],
    "read_sensitivity_classes": [
      "public",
      "restricted"
    ],
    "reads": [
      "fold_aware_weights",
      "weight_diagnostics",
      "temporal_split_registry",
      "channel_time_split_registry",
      "source_channel_matrices",
      "l0_l1_inputs",
      "l3_pilot_capability",
      "fold_eligibility",
      "mcse_report",
      "feature_registry"
    ],
    "required_receipts": [],
    "writes": [
      "measurement_candidate_results",
      "measurement_selection_registry",
      "channel_measurement_selection"
    ]
  },
  "P11": {
    "allowed_next_states": [
      "FROZEN"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "sealed",
    "permitted_states": [
      "SELECTED"
    ],
    "read_sensitivity_classes": [
      "public",
      "restricted"
    ],
    "reads": [
      "measurement_selection_registry",
      "channel_measurement_selection",
      "source_channel_matrices",
      "anchor_capability",
      "temporal_split_registry",
      "feature_panel",
      "feature_registry",
      "l0_l1_inputs",
      "fold_aware_weights",
      "weight_diagnostics",
      "source_config_manifest",
      "environment_observation",
      "fold_eligibility"
    ],
    "required_receipts": [],
    "writes": [
      "model_artifacts",
      "development_oof_predictions",
      "raw_outer_predictions",
      "model_freeze_receipt"
    ]
  },
  "P12": {
    "allowed_next_states": [
      "OUTER_OPEN",
      "EVALUATED"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "open",
    "permitted_states": [
      "FROZEN"
    ],
    "read_sensitivity_classes": [
      "outer",
      "public",
      "restricted"
    ],
    "reads": [
      "mcse_report",
      "model_freeze_receipt",
      "model_artifacts",
      "development_oof_predictions",
      "raw_outer_predictions",
      "sealed_outcome_store",
      "source_channel_matrices",
      "channel_measurement_selection"
    ],
    "required_receipts": [
      "model_freeze_receipt"
    ],
    "writes": [
      "outer_open_receipt",
      "calibration_outputs",
      "evaluation_metrics",
      "bootstrap_batches",
      "utility_scenarios"
    ]
  },
  "P13": {
    "allowed_next_states": [
      "SENSITIVITY"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "open",
    "permitted_states": [
      "EVALUATED"
    ],
    "read_sensitivity_classes": [
      "outer",
      "restricted"
    ],
    "reads": [
      "evaluation_metrics",
      "raw_outer_predictions",
      "sealed_outcome_store",
      "feature_panel",
      "feature_registry",
      "l3_pilot_capability",
      "evidence_ledger",
      "lag_decomposition",
      "censoring_registry",
      "weight_diagnostics",
      "fold_aware_weights",
      "source_channel_matrices"
    ],
    "required_receipts": [],
    "writes": [
      "domain_transfer_outputs",
      "source_sensitivity_outputs",
      "censoring_sensitivity_outputs",
      "hierarchical_pi_sensitivity",
      "ablation_results"
    ]
  },
  "P14": {
    "allowed_next_states": [
      "GATE2"
    ],
    "known_case_access": "none",
    "optional_reads": [],
    "outer_access": "open",
    "permitted_states": [
      "SENSITIVITY"
    ],
    "read_sensitivity_classes": [
      "outer"
    ],
    "reads": [
      "domain_transfer_outputs",
      "source_sensitivity_outputs",
      "censoring_sensitivity_outputs",
      "evaluation_metrics",
      "ablation_results",
      "bootstrap_batches"
    ],
    "required_receipts": [],
    "writes": [
      "gate2_verdict"
    ]
  },
  "P15": {
    "allowed_next_states": [
      "KNOWN_CASES_OPEN"
    ],
    "known_case_access": "open",
    "optional_reads": [],
    "outer_access": "open",
    "permitted_states": [
      "GATE2"
    ],
    "read_sensitivity_classes": [
      "outer",
      "restricted"
    ],
    "reads": [
      "gate2_verdict",
      "model_freeze_receipt",
      "outer_open_receipt",
      "evaluation_metrics",
      "raw_outer_predictions"
    ],
    "required_receipts": [
      "known_cases_seal",
      "gate2_verdict"
    ],
    "writes": [
      "known_case_results"
    ]
  },
  "P16": {
    "allowed_next_states": [
      "GATE3"
    ],
    "known_case_access": "open",
    "optional_reads": [],
    "outer_access": "open",
    "permitted_states": [
      "KNOWN_CASES_OPEN"
    ],
    "read_sensitivity_classes": [
      "known_case",
      "outer",
      "restricted"
    ],
    "reads": [
      "known_case_results",
      "gate2_verdict",
      "measurement_selection_registry",
      "evaluation_metrics",
      "feature_panel",
      "raw_outer_predictions",
      "sealed_outcome_store",
      "feature_registry"
    ],
    "required_receipts": [
      "known_case_results"
    ],
    "writes": [
      "threshold_interaction_results",
      "gate3_verdict"
    ]
  },
  "P17": {
    "allowed_next_states": [
      "REPORTED"
    ],
    "known_case_access": "open",
    "optional_reads": [
      "gate2_verdict",
      "evaluation_metrics",
      "domain_transfer_outputs",
      "source_sensitivity_outputs",
      "censoring_sensitivity_outputs",
      "hierarchical_pi_sensitivity",
      "ablation_results",
      "known_case_results",
      "threshold_interaction_results",
      "calibration_outputs",
      "bootstrap_batches",
      "utility_scenarios"
    ],
    "outer_access": "open",
    "permitted_states": [
      "GATE3"
    ],
    "read_sensitivity_classes": [
      "known_case",
      "outer"
    ],
    "reads": [
      "gate3_verdict"
    ],
    "required_receipts": [
      "gate3_verdict"
    ],
    "writes": [
      "final_result_ledger",
      "final_verdict_matrix",
      "final_artifact_manifest",
      "final_decision_log",
      "chapter4_input_tables",
      "final_gate_figure",
      "final_audit_report",
      "final_report"
    ]
  }
}
```
