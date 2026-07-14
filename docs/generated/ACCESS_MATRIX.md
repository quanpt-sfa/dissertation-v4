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
    "reads": [],
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
    "reads": [],
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
    "reads": [
      "raw_audit"
    ],
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
    "reads": [
      "firm_year_panel"
    ],
    "writes": [
      "evidence_ledger",
      "availability_registry",
      "lag_decomposition"
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
    "reads": [
      "evidence_ledger"
    ],
    "writes": [
      "risk_sets",
      "maturity_audit"
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
    "reads": [
      "risk_sets"
    ],
    "writes": [
      "source_channel_matrices",
      "l0_l1_inputs",
      "l3_pilot_capability",
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
    "reads": [
      "l3_pilot_capability"
    ],
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
    "reads": [
      "observability_registry"
    ],
    "writes": [
      "feature_panel",
      "feature_registry"
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
    "reads": [
      "feature_registry"
    ],
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
    "reads": [
      "feature_panel"
    ],
    "writes": [
      "temporal_split_registry",
      "channel_time_split_registry",
      "fold_aware_weights"
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
    "reads": [
      "fold_aware_weights"
    ],
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
    "reads": [
      "measurement_selection_registry"
    ],
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
    "reads": [
      "model_freeze_receipt",
      "development_oof_predictions",
      "raw_outer_predictions",
      "sealed_outcome_store"
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
    "reads": [
      "evaluation_metrics"
    ],
    "writes": [
      "domain_transfer_outputs",
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
    "reads": [
      "domain_transfer_outputs"
    ],
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
    "reads": [
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
    "reads": [
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
    "optional_reads": [],
    "outer_access": "open",
    "permitted_states": [
      "GATE3"
    ],
    "reads": [
      "gate3_verdict"
    ],
    "writes": [
      "final_result_ledger",
      "final_report"
    ]
  }
}
```
