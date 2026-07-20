# P08 execution batching

P08 separates the methodological replication budget from execution artifact granularity.

The locked simulation configuration continues to determine:

- minimum and maximum replications;
- Monte Carlo standard error stopping criteria;
- scenario and method coverage;
- replication IDs and deterministic RNG seeds.

The production workflow uses an execution-only `BatchMultiplier` with a default of `5`. Five configured chunks are coalesced into one P08B subprocess and one pair of batch/diagnostic artifacts when the minimum-replication boundary permits it.

For the Chapter 3 measurement profile:

| Method family | Configured chunk | Minimum replications | Execution artifact batch | Initial batches per scenario-method |
|---|---:|---:|---:|---:|
| Predictive core | 250 | 2,500 | 1,250 | 2 |
| Standalone/L3 | 100 | 1,000 | 500 | 2 |

With 17 scenarios and 10 active methods, the initial P08 plan is 340 artifact batches rather than approximately 1,700. The maximum adaptive plan is 1,564 artifact batches rather than approximately 7,820.

This compaction does not change estimates because every replication remains keyed by the same protocol hash, scenario ID, method ID, and replication ID. The multiplier is recorded in diagnostics and the final preflight receipt. Resume fails if existing P08 diagnostics were created with a different multiplier.

Example:

```powershell
.\scripts\s3_l3_production_workflow.ps1 `
  -Mode Final `
  -RunId "l3-production-compact-20260720-01" `
  -Workers 8 `
  -BatchMultiplier 5
```
