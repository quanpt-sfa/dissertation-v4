# P08 worker configuration

The production workflow exposes P08 subprocess parallelism as an explicit
execution parameter:

```powershell
.\scripts\s3_l3_production_workflow.ps1 `
  -Mode Final `
  -RunId "<new-run-id>" `
  -Workers 8
```

`Workers` must be a positive integer and defaults to `1`. The PowerShell wrapper
passes the value to `run_final_l3_production.py`, which exports it as
`P08_WORKERS` for `run_pipeline.py`. The pipeline then passes the resolved value
to `p08_profiled_orchestrator.py --workers`.

Parallelism is across P08B subprocesses. Each P08B worker remains single-threaded
internally through the BLAS/OpenMP thread controls already applied in
`p08b_run_batch.py`.

Worker count is an operational scheduling control, not a methodological or model
selection parameter. It is recorded in `PREFLIGHT/l3_preflight_receipt.json`, but
it is not included in the protocol hash. Simulation random-number streams remain
keyed by protocol hash, scenario, method, and replication identifiers, so changing
worker count must not change the estimand or replication-level results.

Do not pull code changes into a production checkout while a run is active. Finish
or stop the active run first, then update the branch, rerun `-Mode Migrate`, commit
any generated documentation changes, and start a new immutable run ID.
