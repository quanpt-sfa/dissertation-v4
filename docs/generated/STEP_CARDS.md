GENERATED FILE — DO NOT EDIT
Source: config/pipeline.yaml

# Step cards
## P00

Protocol lock

Reads: []

Writes: ['registry_lock', 'protocol_hash', 'source_config_manifest', 'capability_seed', 'decision_traceability', 'artifact_catalog', 'schema_catalog', 'step_catalog', 'access_matrix', 'known_cases_seal', 'environment_expectation', 'environment_observation', 'p00_audit_report', 'job_manifest', 'success_receipt']

Required receipts: []

## P01

Raw audit

Reads: []

Writes: ['raw_audit']

Required receipts: []

## P02

Firm master and as-of panel

Reads: ['raw_audit']

Writes: ['firm_master', 'firm_year_panel', 'duplicate_map']

Required receipts: []

## P03

Evidence ledger

Reads: ['firm_year_panel']

Writes: ['evidence_ledger', 'availability_registry', 'lag_decomposition']

Required receipts: []

## P04

Risk set and maturity

Reads: ['evidence_ledger']

Writes: ['risk_sets', 'maturity_audit']

Required receipts: []

## P05

Measurement construction

Reads: ['risk_sets']

Writes: ['source_channel_matrices', 'l0_l1_inputs', 'l3_pilot_capability', 'sealed_outcome_store']

Required receipts: []

## P06

Verification and observability

Reads: ['l3_pilot_capability']

Writes: ['observability_registry']

Required receipts: []

## P07

Feature and leakage audit

Reads: ['observability_registry']

Writes: ['feature_panel', 'feature_registry']

Required receipts: []

## P08

Method simulation

Reads: ['feature_registry']

Writes: ['simulation_scenario_registry', 'simulation_batches', 'mcse_report']

Required receipts: []

## P09

Splits and weights

Reads: ['feature_panel']

Writes: ['temporal_split_registry', 'channel_time_split_registry', 'fold_aware_weights']

Required receipts: []

## P10

Measurement selection

Reads: ['fold_aware_weights']

Writes: ['measurement_candidate_results', 'measurement_selection_registry', 'channel_measurement_selection']

Required receipts: []

## P11

Learner fitting and freeze

Reads: ['measurement_selection_registry']

Writes: ['model_artifacts', 'development_oof_predictions', 'raw_outer_predictions', 'model_freeze_receipt']

Required receipts: []

## P12

Outer opening and evaluation

Reads: ['model_freeze_receipt', 'development_oof_predictions', 'raw_outer_predictions', 'sealed_outcome_store']

Writes: ['outer_open_receipt', 'calibration_outputs', 'evaluation_metrics', 'bootstrap_batches', 'utility_scenarios']

Required receipts: ['model_freeze_receipt']

## P13

Sensitivity and transfer

Reads: ['evaluation_metrics']

Writes: ['domain_transfer_outputs', 'hierarchical_pi_sensitivity', 'ablation_results']

Required receipts: []

## P14

Gate 2

Reads: ['domain_transfer_outputs']

Writes: ['gate2_verdict']

Required receipts: []

## P15

Known cases

Reads: ['gate2_verdict']

Writes: ['known_case_results']

Required receipts: ['known_cases_seal', 'gate2_verdict']

## P16

Gate 3

Reads: ['known_case_results']

Writes: ['threshold_interaction_results', 'gate3_verdict']

Required receipts: ['known_case_results']

## P17

Final reporting

Reads: ['gate3_verdict']

Writes: ['final_result_ledger', 'final_report']

Required receipts: ['gate3_verdict']

