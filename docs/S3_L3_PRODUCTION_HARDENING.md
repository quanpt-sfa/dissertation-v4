# S3/L3 production-hardening workflow

This branch adds an explicit four-phase workflow. The scripts do not use outer-fold outcomes or sealed known cases to set fixed-pi values, accuracy priors, measurement selection, or calibration.

## 1. Pull and switch to the branch

Open PowerShell in the repository root, or resolve it from a path relative to your current workspace:

```powershell
$projectRoot = (Resolve-Path ".").Path
Set-Location $projectRoot
git fetch origin
git switch agent/s3-content-l3-production-lock
```

## 2. Apply the code/config migration

```powershell
.\scripts\s3_l3_production_workflow.ps1 -Mode Migrate
```

The migration performs these changes:

- separates included and excluded S3 rows within the same `document_id × firm_id` group;
- restricts eligible endpoint provenance to included rows;
- retains excluded rows in the decision ledger with `EXCLUDED_BY_SOURCE_RULE`;
- defines `primary_misstatement` with `S3_CONTENT` as its only S3 endpoint;
- keeps `S3_REPORTING` and `S3_TIMELINESS` as sensitivity sets;
- keeps `S3_BROAD` descriptive/falsification-only;
- binds L2, the P05 L3 pilot, and fold-local L3 to the same primary source set;
- changes the known-case source to `data/source/known_case_registry.csv`;
- validates the external-validation-only known-case flags;
- adds endpoint-specific S3 development counts and unresolved diagnostics;
- writes regression tests and regenerates the locked catalogs.

After the script passes, inspect and commit the generated diff:

```powershell
git diff
git add config docs scripts src templates tests
git commit -m "Harden S3 content measurement and L3 production workflow"
git status --porcelain
git push
```

`git status --porcelain` must be empty before any immutable run.

## 3. Required raw-source names and contracts

The raw root must contain:

```text
data/source/firm_event_sanction_panel.csv
data/source/known_case_registry.csv
```

Validate them directly:

```powershell
uv run python scripts/validate_production_source_contracts.py `
  --raw-root (Get-Location).Path
```

The known-case registry must contain exactly these firm-years:

```text
K1,TAR,2020
K1,TAR,2022
K2,TTF,2016
K3,ROS,2018
K3,ROS,2019
K4,FHH,2019
```

Each row must use:

```text
case_construct = CONFIRMED_FINANCIAL_REPORTING_CASE
role = SIMULATION_EXTERNAL_VALIDATION
training_include_flag = FALSE
calibration_include_flag = FALSE
model_selection_include_flag = FALSE
external_validation_include_flag = TRUE
```

## 4. Run the preparatory P00-P06 audit

Use a new run ID:

```powershell
.\scripts\s3_l3_production_workflow.ps1 `
  -Mode Prepare `
  -RunId "l3-preparation-20260719-01"
```

The script runs tests, snapshots the registered sources, executes P00-P06, creates `S3_AUDIT` and `CALIBRATION`, and writes:

```text
artifacts/runs/<run-id>/PREPARATION/l3_preparation_receipt.json
```

The receipt is ready for locking only when:

```text
p03_p05_outcome_mismatch_count = 0
p03_p05_missing_key_count = 0
eligible_unresolved_sanction_year_mapping_count = 0
sanction_year_unresolved_firm_year_count = 0
development_positive_count_by_endpoint.S3_CONTENT > 0
outer_outcomes_accessed = false
known_cases_accessed = false
```

Counts are reported separately for `S3_CONTENT`, `S3_REPORTING`, `S3_TIMELINESS`, and `S3_BROAD`. Do not use their sum as an independent positive count because the endpoints are nested.

## 5. Review and lock fixed-pi values and accuracy priors

The registry compiler treats every `.yaml` beneath `config/` as configuration and rejects undeclared files. Therefore both the template and the reviewed working file must remain outside `config/`.

Copy the template to a working lock file:

```powershell
New-Item -ItemType Directory -Force working\l3 | Out-Null
Copy-Item `
  templates\l3_parameter_lock.template.yaml `
  working\l3\l3_parameter_lock.reviewed.yaml
```

Replace every illustrative number and placeholder. Set:

```text
provenance.status = LOCKED
provenance.outer_outcomes_accessed = false
provenance.known_cases_accessed = false
```

Apply the reviewed lock:

```powershell
.\scripts\s3_l3_production_workflow.ps1 `
  -Mode Lock `
  -LockFile "working\l3\l3_parameter_lock.reviewed.yaml" `
  -PreparationReceipt "artifacts\runs\l3-preparation-20260719-01\PREPARATION\l3_preparation_receipt.json"
```

The lock script writes fixed-pi and profile priors to `measurement.yaml`, records the input hashes and preparation protocol hash, regenerates catalogs, and reruns tests. Review and commit the resulting configuration and receipt before the final run. The reviewed working worksheet can remain uncommitted or be archived outside `config/`.

## 6. Run one fail-closed P00-P17 production command

Use another new run ID after the parameter-lock commit:

```powershell
.\scripts\s3_l3_production_workflow.ps1 `
  -Mode Final `
  -RunId "dissertation-final-l1-l2-l3-v1"
```

The final wrapper:

1. requires a clean committed tree;
2. validates the sanction and known-case raw sources;
3. requires `S3_CONTENT` as the only primary S3 endpoint;
4. requires a nonempty fixed-pi grid, complete priors, and a lock receipt;
5. runs P00-P06;
6. reruns S3 and calibration audits;
7. blocks on reconciliation, unresolved mappings, absent S3 content positives, or outer-outcome access;
8. requires the P05 L3 pilot to be `AVAILABLE` and executed;
9. runs through P10 and requires `L3_fixed_pi=AVAILABLE` in every outer fold;
10. resumes through P17 only after all checks pass.

The final preflight receipt is written to:

```text
artifacts/runs/<run-id>/PREFLIGHT/l3_preflight_receipt.json
```

A completed P17 run is not treated as an L3 production run unless this receipt has `status = PASS_P00_P17`.
