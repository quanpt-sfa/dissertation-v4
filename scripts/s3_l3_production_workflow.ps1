param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Migrate", "Prepare", "Lock", "Final")]
    [string]$Mode,

    [string]$RunId,
    [string]$RawRoot = "D:\Works\dissertation\dissertation-v4",
    [string]$OutputRoot = "D:\Works\dissertation\dissertation-v4\artifacts\runs",
    [string]$Config = "config\pipeline.yaml",
    [string]$LockFile,
    [string]$PreparationReceipt,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

function Invoke-UvPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Write-Host "+ uv run python $($Arguments -join ' ')"
    & uv run python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

switch ($Mode) {
    "Migrate" {
        Write-Host "+ uv run python scripts/apply_s3_l3_production_hardening.py --check"
        & uv run python scripts/apply_s3_l3_production_hardening.py --check
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) {
            throw "Hardening check failed with exit code $LASTEXITCODE"
        }
        Invoke-UvPython scripts/apply_s3_l3_production_hardening.py --apply
        Invoke-UvPython scripts/finalize_s3_l3_production_hardening.py --apply
        Invoke-UvPython scripts/bootstrap_repository.py --config $Config --write
        Invoke-UvPython scripts/bootstrap_repository.py --config $Config --check
        Invoke-UvPython -m pytest -q
        Write-Host "Migration complete. Review git diff, then commit before Prepare."
    }

    "Prepare" {
        if (-not $RunId) { throw "Prepare requires -RunId" }
        Invoke-UvPython scripts/validate_production_source_contracts.py --raw-root $RawRoot
        $args = @(
            "scripts/run_l3_preparation.py",
            "--run-id", $RunId,
            "--raw-root", $RawRoot,
            "--output-root", $OutputRoot,
            "--config", $Config
        )
        if ($SkipTests) { $args += "--skip-tests" }
        Invoke-UvPython @args
    }

    "Lock" {
        if (-not $LockFile) { throw "Lock requires -LockFile" }
        if (-not $PreparationReceipt) { throw "Lock requires -PreparationReceipt" }
        Invoke-UvPython scripts/lock_l3_parameters.py `
            --lock-file $LockFile `
            --preparation-receipt $PreparationReceipt `
            --write
        Invoke-UvPython scripts/bootstrap_repository.py --config $Config --write
        Invoke-UvPython scripts/bootstrap_repository.py --config $Config --check
        Invoke-UvPython -m pytest -q
        Write-Host "L3 parameters locked. Review and commit the config before Final."
    }

    "Final" {
        if (-not $RunId) { throw "Final requires -RunId" }
        Invoke-UvPython scripts/validate_production_source_contracts.py --raw-root $RawRoot
        $args = @(
            "scripts/run_final_l3_production.py",
            "--run-id", $RunId,
            "--raw-root", $RawRoot,
            "--output-root", $OutputRoot,
            "--config", $Config
        )
        if ($SkipTests) { $args += "--skip-tests" }
        Invoke-UvPython @args
    }
}
