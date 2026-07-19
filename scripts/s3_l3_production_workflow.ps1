param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Migrate", "Prepare", "Final")]
    [string]$Mode,

    [string]$RunId,
    [string]$RawRoot,
    [string]$OutputRoot,
    [string]$Config,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

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

Push-Location $ProjectRoot
try {
    switch ($Mode) {
        "Migrate" {
            # The historical S3/L3 one-time migration is already part of the base branch.
            # This mode now validates the new P0-locked scenario architecture directly.
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
                "--config", $Config
            )
            if ($SkipTests) { $runnerArgs += "--skip-tests" }
            Invoke-UvPython @runnerArgs
        }
    }
}
finally {
    Pop-Location
}
