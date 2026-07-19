param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Migrate", "Prepare", "Lock", "Final")]
    [string]$Mode,

    [string]$RunId,
    [string]$RawRoot,
    [string]$OutputRoot,
    [string]$Config,
    [string]$LockFile,
    [string]$PreparationReceipt,
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

function Assert-HardeningApplied {
    $checks = @(
        "scripts/repair_s3_l3_migration_checker.py",
        "scripts/apply_s3_l3_production_hardening.py",
        "scripts/finalize_s3_l3_production_hardening.py",
        "scripts/repair_s3_l3_hardening_regressions.py"
    )
    foreach ($script in $checks) {
        Write-Host "+ uv run python $script --check"
        & uv run python $script --check
        if ($LASTEXITCODE -eq 2) {
            throw "S3/L3 hardening is incomplete. Run this workflow with -Mode Migrate, review and commit the generated changes, then use a new run ID."
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Hardening verification failed for $script with exit code $LASTEXITCODE"
        }
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
            # Synchronize the one-time migration generator before asking it to
            # validate a checkout that may already contain the compatibility repairs.
            Invoke-UvPython scripts/repair_s3_l3_migration_checker.py --apply
            Write-Host "+ uv run python scripts/apply_s3_l3_production_hardening.py --check"
            & uv run python scripts/apply_s3_l3_production_hardening.py --check
            if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) {
                throw "Hardening check failed with exit code $LASTEXITCODE"
            }
            Invoke-UvPython scripts/apply_s3_l3_production_hardening.py --apply
            Invoke-UvPython scripts/finalize_s3_l3_production_hardening.py --apply
            Invoke-UvPython scripts/repair_s3_l3_hardening_regressions.py --apply
            Invoke-UvPython scripts/bootstrap_repository.py --config $Config --write
            Invoke-UvPython scripts/bootstrap_repository.py --config $Config --check
            Invoke-UvPythonDev -m pytest -q
            Assert-HardeningApplied
            Write-Host "Migration complete. Review git diff, then commit before Prepare."
        }

        "Prepare" {
            if (-not $RunId) { throw "Prepare requires -RunId" }
            Assert-HardeningApplied
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

        "Lock" {
            if (-not $LockFile) { throw "Lock requires -LockFile" }
            if (-not $PreparationReceipt) { throw "Lock requires -PreparationReceipt" }
            Assert-HardeningApplied
            Invoke-UvPython scripts/lock_l3_parameters.py `
                --lock-file $LockFile `
                --preparation-receipt $PreparationReceipt `
                --write
            Invoke-UvPython scripts/bootstrap_repository.py --config $Config --write
            Invoke-UvPython scripts/bootstrap_repository.py --config $Config --check
            Invoke-UvPythonDev -m pytest -q
            Write-Host "L3 parameters locked. Review and commit the config before Final."
        }

        "Final" {
            if (-not $RunId) { throw "Final requires -RunId" }
            Assert-HardeningApplied
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
