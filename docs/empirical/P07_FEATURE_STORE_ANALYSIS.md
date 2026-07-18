# P07 feature-store analysis

This document describes the reproducible P07 analysis contract. Run-specific
results are immutable artifacts under
`artifacts/runs/<run-id>/P07/`; this document does not replace those artifacts.

## Technical validation

P07 validates the package manifest and build manifest, every registered file hash,
feature version, scalar schema, constant feature ID, fiscal-year range, value type,
source snapshot hash, and unique feature-store firm-year key. Unregistered files
and paths outside the configured store root are rejected.

## Identifier mapping

The immutable store identifier is reconciled through the registered year-valid
crosswalk and normalized at the ingestion boundary. P02-only firm-years remain in
the panel with missing feature values; store-only identifiers are reported and are
never silently discarded. Ambiguous or overlapping mappings block P07.

## Availability and leakage

`synthetic_annual_anchor` is a conventional anchor, not an observed publication
date. Values after `prediction_time`, values with unresolved usable availability,
future-year information, outer outcomes, known cases, and prohibited target
components are blocked fail-closed. Leakage decisions are stored per
`feature_id × target_id`.

## Descriptive evidence

P07 publishes coverage, missingness, value-scale, accounting-identity,
audited/unaudited adjustment, ratio, temporal, and feature-redundancy audits.
These diagnostics do not impute, winsorize, standardize, select features, or fit a
model. Accounting-identity residuals are data-quality and mapping evidence, not a
fraud classification.

## Confirmatory composition and unresolved decisions

Only LOCKED features allowed by the target-specific leakage registry may enter a
confirmatory view. Nine Beneish files remain outside all confirmatory and default
modelling views until TATA, DEPI, receivables, sales, PPE, denominator,
nonpositive-denominator, prior-year, statement-scope, and Vietnamese accounting
mappings receive explicit protocol approval.
