GENERATED FILE — DO NOT EDIT
Source: config/pipeline.yaml

# Step cards
## P00

Protocol locking and configuration compilation

Reads: []

Writes: ['registry_lock', 'protocol_hash', 'source_config_manifest', 'capability_seed', 'decision_traceability', 'artifact_catalog', 'schema_catalog', 'step_catalog', 'access_matrix', 'known_cases_seal', 'environment_expectation', 'p00_audit_report', 'job_manifest', 'success_receipt']

## P01

Raw-data audit interface

Reads: []

Writes: ['raw_audit']

## P02

Entity and firm-year panel interface

Reads: ['raw_audit']

Writes: ['firm_master', 'firm_year_panel', 'duplicate_map']

## P03

Evidence ledger interface

Reads: ['firm_year_panel']

Writes: ['evidence_ledger', 'availability_registry', 'lag_decomposition']

## P04

Maturity and risk-set interface

Reads: ['evidence_ledger']

Writes: ['risk_sets', 'maturity_audit']

## P05

Measurement inputs and L3 fixed-pi pilot interface

Reads: ['risk_sets']

Writes: ['source_channel_matrices', 'l0_l1_inputs', 'l3_pilot_capability']

## P06

Observability registry interface

Reads: ['l3_pilot_capability']

Writes: ['observability_registry']

## P07

Feature and leakage registry interface

Reads: ['observability_registry']

Writes: ['feature_panel', 'feature_registry']

## P08

Simulation interface

Reads: ['feature_registry']

Writes: ['simulation_scenario_registry', 'simulation_batches', 'mcse_report']

## P09

Temporal splits and fold-aware weights interface

Reads: ['feature_panel']

Writes: ['temporal_split_registry', 'channel_time_split_registry', 'fold_aware_weights']

## P10

Measurement selection interface

Reads: ['fold_aware_weights']

Writes: ['measurement_candidate_results', 'measurement_selection_registry', 'channel_measurement_selection']

## P11

Model fitting and freeze interface

Reads: ['measurement_selection_registry']

Writes: ['model_artifacts', 'model_freeze_receipt']

## P12

Outer opening and evaluation interface

Reads: ['model_freeze_receipt']

Writes: ['outer_open_receipt', 'raw_outer_predictions', 'calibration_outputs', 'evaluation_metrics', 'bootstrap_batches', 'utility_scenarios']

## P13

Sensitivity and internal-external validation interface

Reads: ['evaluation_metrics']

Writes: ['domain_transfer_outputs', 'hierarchical_pi_sensitivity', 'ablation_results']

## P14

Gate 2 interface

Reads: ['domain_transfer_outputs']

Writes: ['gate2_verdict']

## P15

Known-case interface

Reads: ['gate2_verdict']

Writes: ['known_case_results']

## P16

Gate 3 interface

Reads: ['known_case_results']

Writes: ['threshold_interaction_results', 'gate3_verdict']

## P17

Final reporting interface

Reads: ['gate3_verdict']

Writes: ['final_result_ledger', 'final_report']

