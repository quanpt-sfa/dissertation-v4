param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Migrate", "Prepare", "Final")]
    [string]$Mode,

    [string]$RunId,
    [string]$RawRoot,
    [string]$OutputRoot,
    [string]$Config,
    [int]$Workers = 1,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

if ($Workers -lt 1) {
    throw "Workers must be a positive integer"
}

# Derive all default paths from the checked-out repository containing this script.
# This keeps the workflow portable across machines, drive letters, and checkout folders.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($RawRoot)) {
    $RawRoot = $ProjectRoot
} else {
    $RawRoot = (Resolve-Path $RawRoot).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "artifacts\runs"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot $OutputRoot
} else {
    $OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
}

if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $ProjectRoot "config\pipeline.yaml"
} elseif (-not [System.IO.Path]::IsPathRooted($Config)) {
    $Config = Join-Path $ProjectRoot $Config
} else {
    $Config = [System.IO.Path]::GetFullPath($Config)
}

if (-not (Test-Path -LiteralPath $RawRoot -PathType Container)) {
    throw "RawRoot does not exist: $RawRoot"
}
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    throw "Pipeline config does not exist: $Config"
}

function Invoke-UvPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Write-Host "+ uv run python $($Arguments -join ' ')"
    & uv run python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-UvPythonDev {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Write-Host "+ uv run --extra dev python $($Arguments -join ' ')"
    & uv run --extra dev python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

Write-Host "ProjectRoot: $ProjectRoot"
Write-Host "RawRoot:     $RawRoot"
Write-Host "OutputRoot:  $OutputRoot"
Write-Host "Config:      $Config"
if ($Mode -eq "Final") {
    Write-Host "P08 Workers: $Workers"
}

Push-Location $ProjectRoot
try {
    switch ($Mode) {
        "Migrate" {
            # Run branch-specific checks before generating repository artifacts. The
            # static guard scans every production Python file and reports all violations
            # in one failure instead of exposing them one file at a time.
            Invoke-UvPython -m compileall -q scripts src tests
            Invoke-UvPythonDev -m pytest -q `
                tests/core/test_forbidden_pattern_metadata_exceptions.py `
                tests/assurance/test_appendix_b_decisions.py::test_T006_prior_accuracy `
                tests/features/test_pipeline_feature_generator.py `
                tests/features/test_pipeline_feature_registry_grammar.py `
                tests/stages/test_p08_worker_configuration.py `
                tests/stages/test_s3_decision_ledger_grain.py `
                tests/stages/test_s3_calendar_year_targets.py `
                tests/stages/test_l3_latent_class.py `
                tests/stages/test_p07_pipeline_generated_features.py `
                tests/stages/test_remaining_stage_invariants.py `
                tests/test_l2_l3_calibration.py `
                tests/test_l3_preregistered_scenarios.py

            Invoke-UvPython scripts/bootstrap_repository.py --config $Config --write
            Invoke-UvPython scripts/bootstrap_repository.py --config $Config --check
            Invoke-UvPythonDev -m pytest -q
            Write-Host "Scenario-registry migration validated. Review and commit before Prepare."
        }

        "Prepare" {
            if (-not $RunId) { throw "Prepare requires -RunId" }
            Invoke-UvPython scripts/validate_production_source_contracts.py --raw-root $RawRoot
            $runnerArgs = @(
                "scripts/run_l3_preparation.py",
                "--run-id", $RunId,
                "--raw-root", $RawRoot,
                "--output-root", $OutputRoot,
                "--config", $Config
            )
            if ($SkipTests) { $runnerArgs += "--skip-tests" }
            Invoke-UvPython @runnerArgs
        }

        "Final" {
            if (-not $RunId) { throw "Final requires -RunId" }
            Invoke-UvPython scripts/validate_production_source_contracts.py --raw-root $RawRoot
            $runnerArgs = @(
                "scripts/run_final_l3_production.py",
                "--run-id", $RunId,
                "--raw-root", $RawRoot,
                "--output-root", $OutputRoot,
                "--config", $Config,
                "--workers", [string]$Workers
            )
            if ($SkipTests) { $runnerArgs += "--skip-tests" }
            Invoke-UvPython @runnerArgs
        }
    }
}
finally {
    Pop-Location
}
