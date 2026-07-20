# P07 pipeline-generated feature analysis

This document describes the reproducible P07 analysis contract. Run-specific
results are immutable artifacts under `artifacts/runs/<run-id>/P07/`; this
document does not replace those artifacts.

## Generation boundary

P07 no longer consumes an external one-file-per-feature package. It resolves the
single registered `financial_statement_core_long` source from the locked P00
registry, requires its P01 audit to permit advancement, reads it through the
registered P01 reader, and materializes LOCKED feature definitions directly onto
the canonical P02 firm-year spine.

The generator uses the registered entity-resolution policy, expected consolidated
statement scope, expected VND unit, audit status, source item identifiers, lineage,
dependencies, and formula metadata. Duplicate firm-year-status-item measurements,
unsupported transformation steps, unresolved dependencies, undeclared formula
names, and attempts to overwrite upstream columns fail closed.

## Supported registered transformations

The production grammar is deliberately restricted:

- atomic `source_selection` and `component_sum` features;
- audited-versus-unaudited `prepost_difference` features;
- `registered_ratio` features using exact one-year lags where declared;
- observability `coverage_count` and complete-flag features.

Registered formulas are parsed as a restricted expression tree. Only declared
dependencies, numeric constants, arithmetic operators, `abs()`, and `average()`
are accepted. P07 does not use unrestricted expression evaluation. A zero
denominator remains missing, and a missing fiscal-year predecessor is not bridged
by an earlier observation.

## Technical validation

P07 verifies source audit status, source identity, entity mapping, scope and unit,
unique measurement grain, feature dependency resolution, generated feature count,
firm-year uniqueness, value type, target-specific leakage rules, and as-of
availability. The former `feature_store_*` artifact identifiers remain only as
compatibility receipts required by the locked artifact catalog; those receipts
explicitly record that no external manifest, crosswalk, or feature package was
used.

## Identifier mapping

The raw issuer identifier is normalized and reconciled through the same registered
entity-resolution policy used by P02 and P03. The generated values are joined only
to the canonical P02 firm-year spine. Source-only firm-years are excluded and
counted in the generation audit; P02-only firm-years remain with missing feature
values. Ambiguous or duplicate mappings block P07.

## Availability and leakage

`synthetic_annual_anchor` is a conventional anchor, not an observed publication
date. Future-year information, outer outcomes, known cases, prohibited target
components, and undeclared lags are blocked fail closed. Leakage decisions remain
stored per `feature_id × target_id`.

## Descriptive evidence

P07 publishes coverage, missingness, value-scale, accounting-identity,
audited/unaudited adjustment, ratio, temporal, and feature-redundancy audits.
These diagnostics do not impute, winsorize, standardize, select features, or fit a
model. Accounting-identity residuals are data-quality and mapping evidence, not a
fraud classification.

## Confirmatory composition and unresolved decisions

Only LOCKED features allowed by the target-specific leakage registry may enter a
confirmatory view. Nine Beneish candidates remain outside all confirmatory and
default modelling views until their Vietnamese item mappings and denominator
policies receive explicit protocol approval. They are not materialized by the
production generator merely because their metadata exist in `intended_registry`.
