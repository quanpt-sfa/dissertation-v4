# A-01--A-15 methodological audit remediation

This note records the disposition of the external audit against the current `main` branch. Every code-level correction remains subject to the repository-wide quality workflow before merge.

- **A-01 fixed.** `eligible_observability_view` is accepted by the production and nested-refit feature selectors; model eligibility is now a closed enum; P11 and P12 fail closed when the Gate 2 candidate/reference groups are absent.
- **A-02 controlled, not assumed away.** Production snapshots now require an extract-provenance manifest containing the vendor revision policy and whether point-in-time vintages exist. The locked feature contract states that point-in-time vintages are currently unavailable and that a restatement sensitivity is required. No claim is made that a current FiinPro snapshot is point-in-time.
- **A-03 fixed as a fail-closed reporting contract.** L3 parameters remain empty because no external elicitation has yet justified them. The registry marks L3 `PENDING_EXTERNAL_ELICITATION` and non-reportable. A future change to `report_required: true` fails unless fixed-pi values and accuracy priors are locked.
- **A-04 fixed as an explicit assumption.** The study no longer calls 31 March an observed publication date. It is registered as a synthetic annual anchor, with 30 June and 30 September sensitivity anchors. Implementing those sensitivity runs remains required before the final paper.
- **A-05 fixed at the snapshot boundary.** Snapshots record the raw-root locator and a required extract-provenance manifest. The manifest is included in the source-content hash; moving the same raw files does not alter that content hash.
- **A-06 partially resolved.** The current SMOTE/ADASYN regression test passes and diagnostic collectors retain failure classes and affected replication IDs. The broad resampling exception remains unchanged because editing the historically untyped simulation module would require a separate zero-error remediation; this PR does not weaken the Pyright ratchet to conceal that backlog.
- **A-07 fixed.** The retrospective feature sample ends in 2025. The 2026 year is separately registered as prospective-only and unavailable by the current data cutoff; it is not included in retrospective outer folds.
- **A-08 registered but not falsely claimed complete.** The primary split has zero embargo and a one-year embargo sensitivity is now part of the locked protocol. The final paper still requires the corresponding refit/AP comparison; registration alone is not a result.
- **A-09 retained by design.** Rolling panel forecasts may reuse earlier observations from the same firm. Firm-clustered bootstrap remains the inferential unit; final descriptive output must report both firm-years and distinct firms.
- **A-10 declared explicitly.** No winsorization is applied. P1/P99 values are diagnostic only. Any future winsorization must be fitted inside the development fold.
- **A-11 blocked from claims.** The fixed-accounting/Beneish benchmark is explicitly non-operational until the Vietnam mapping review is complete.
- **A-12 partially eliminated.** Platt regularization/iterations, Gate 3 breakpoint grid/logistic controls, known-case weak percentile, and seed offsets are protocol configuration. Existing D07/D09/D10 hard guards are retained intentionally.
- **A-13 retained.** Comments now identify D07/D09/D10 hard coding as deliberate preregistration guards.
- **A-14 already resolved.** The backup sanction CSV is absent from `main`.
- **A-15 deferred.** Long-function decomposition is maintainability work, not a methodological correction, and should be split into behavior-preserving PRs after the blockers above are stable.
