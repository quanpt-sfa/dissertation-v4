GENERATED FILE — DO NOT EDIT
Source: config/pipeline.yaml

# Step cards
## P00

Lock protocol configuration

Reads: []

Writes: ['registry_lock', 'protocol_hash', 'source_config_manifest', 'capability_seed', 'decision_traceability', 'artifact_catalog', 'schema_catalog', 'step_catalog', 'access_matrix', 'known_cases_seal', 'environment_expectation', 'p00_audit_report']

## P10

Select locked measurement candidate

Reads: []

Writes: ['measurement_selection_registry']

## P11

Freeze models before outer opening

Reads: ['measurement_selection_registry']

Writes: ['model_freeze_receipt']

## P15

Open sealed known cases after evaluation

Reads: ['model_freeze_receipt']

Writes: ['known_case_results']

## P17

Build final reports only

Reads: ['known_case_results']

Writes: ['final_report']

