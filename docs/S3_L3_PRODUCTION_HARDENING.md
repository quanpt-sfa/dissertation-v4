# S3/L3 production-hardening workflow

L1 annual remains the mandatory primary track. L2 and L3 remain optional and capability-gated. L3 fixed-prevalence assumptions and source-accuracy priors are registered before empirical execution in `config/methodology/l3_scenarios.yaml`; the file is included in the P0 protocol hash.

The workflow never uses outer-fold outcomes or sealed known cases to define scenarios, choose a fixed-π value, set source priors, calibrate the measurement model, or select the primary L3 scenario.

## 1. Branch and protocol

```powershell
cd D:\Quan\dissertation-v4
git fetch origin
git switch agent/l3-preregistered-scenarios
```

The L3 registry contains:

- one scenario with `role: primary`;
- zero or more scenarios with `role: robustness`;
- a fixed prevalence for every scenario;
- a registered source-accuracy prior set;
- `run_all_registered_scenarios: true`;
- `performance_based_scenario_selection_forbidden: true`;
- `outer_outcomes_accessed: false`;
- `known_cases_accessed: false`.

Any edit to this registry changes the P0 protocol hash. There is no post-Preparation parameter worksheet and no `Lock` mode.

## 2. Apply and validate the repository migration

```powershell
.\scripts\s3_l3_production_workflow.ps1 -Mode Migrate
```

The migration and bootstrap must pass before any empirical run. Then review and commit the branch:

```powershell
git status --short
git diff --stat
git add config docs scripts src tests
git commit -m "Pre-register L3 scenarios at P0"
git status --porcelain
git push -u origin agent/l3-preregistered-scenarios
```

`git status --porcelain` must be empty before an immutable run.

## 3. Required raw-source contracts

The raw root must contain:

```text
data/source/firm_event_sanction_panel.csv
data/source/known_case_registry.csv
```

Validate the contracts:

```powershell
uv run python scripts/validate_production_source_contracts.py `
  --raw-root (Get-Location).Path
```

Known cases remain external-validation-only. They cannot enter training, calibration, model selection, scenario construction, or scenario diagnostics.

## 4. Preparation is capability-only

Use a new run ID:

```powershell
.\scripts\s3_l3_production_workflow.ps1 `
  -Mode Prepare `
  -RunId "l3-preparation-preregistered-01"
```

The command runs tests, P00-P06, the S3 year audit, and measurement calibration. It writes:

```text
artifacts/runs/<run-id>/PREPARATION/l3_preparation_receipt.json
```

The receipt status is one of:

```text
L3_AVAILABLE
L3_UNAVAILABLE_BY_DESIGN
L3_BLOCKED_BY_DATA_CONTRACT
```

`L3_AVAILABLE` requires the pre-registered primary scenario to pass the L3 pilot diagnostics and the registered-scenario eligible fraction to satisfy the robustness gate.

`L3_UNAVAILABLE_BY_DESIGN` is not a pipeline failure. L1 continues as the mandatory track, and L2/L3 follow the configured optional-track policy.

`L3_BLOCKED_BY_DATA_CONTRACT` is a hard stop caused by reconciliation errors, unresolved eligible sanction-year mappings, malformed audit outputs, or prohibited outcome access.

The Preparation command does not modify fixed-π values, priors, scenario roles, or the primary scenario.

## 5. Final P00-P17 production run

Use a distinct run ID:

```powershell
.\scripts\s3_l3_production_workflow.ps1 `
  -Mode Final `
  -RunId "dissertation-final-preregistered-l3-v1"
```

The final wrapper:

1. requires a clean committed tree;
2. compiles and validates the P0-locked scenario registry;
3. validates the sanction and known-case raw-source contracts;
4. requires `S3_CONTENT` as the only primary S3 endpoint;
5. runs P00-P06 and repeats the S3/calibration audits;
6. blocks on data-contract violations or prohibited outcome access;
7. runs every registered L3 scenario when L3 capability is available;
8. uses only the pre-registered primary scenario for the primary L3 target;
9. treats strict-channel losses as capability/reporting diagnostics, never as a scenario-selection objective;
10. skips L3 without failing L1 when L3 is unavailable by design;
11. resumes through P17 after preflight passes.

The final receipt is written to:

```text
artifacts/runs/<run-id>/PREFLIGHT/l3_preflight_receipt.json
```

A completed production run has:

```text
status = PASS_P00_P17
scenario_registry_status = LOCKED_AT_P0
scenario_selection_rule = PRE_REGISTERED_PRIMARY
performance_based_scenario_selection = false
outer_outcomes_accessed = false
known_cases_accessed = false
```

`l3_execution_status` is either `EXECUTED_ALL_REGISTERED_SCENARIOS` or `SKIPPED_UNAVAILABLE_BY_DESIGN`.
