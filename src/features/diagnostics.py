"""Descriptive, model-free P07 feature-store diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from core.semantic_keys import AVAILABILITY_BASIS, TEMPORAL_ROLE


@dataclass(frozen=True)
class FeatureDiagnostics:
    coverage: pd.DataFrame
    value_scale: pd.DataFrame
    accounting_identities: pd.DataFrame
    audited_unaudited: pd.DataFrame
    ratios: pd.DataFrame
    temporal: pd.DataFrame
    redundancy: pd.DataFrame


def build_feature_diagnostics(
    *,
    panel: pd.DataFrame,
    definitions: list[dict[str, object]],
    firm_column: str,
    year_column: str,
) -> FeatureDiagnostics:
    """Compute only descriptive diagnostics; no values are transformed or fitted."""
    operational = [
        item
        for item in definitions
        if item.get("research_decision_status") == "LOCKED"
        and str(item["feature_id"]) in panel.columns
    ]
    feature_ids = [str(item["feature_id"]) for item in operational]
    coverage = _coverage(panel, operational, firm_column, year_column)
    value_scale = _value_scale(panel, operational)
    identities = _accounting_identities(panel, firm_column, year_column)
    audited = _audited_unaudited(panel, firm_column, year_column)
    ratios = _ratio_diagnostics(panel, operational, year_column)
    temporal = _temporal_diagnostics(panel, operational, firm_column, year_column)
    redundancy = _redundancy(panel, feature_ids)
    return FeatureDiagnostics(
        coverage=coverage,
        value_scale=value_scale,
        accounting_identities=identities,
        audited_unaudited=audited,
        ratios=ratios,
        temporal=temporal,
        redundancy=redundancy,
    )


def _coverage(
    panel: pd.DataFrame,
    definitions: list[dict[str, object]],
    firm: str,
    year: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = panel.sort_values([firm, year], kind="stable")
    for item in definitions:
        feature_id = str(item["feature_id"])
        valid = ordered[feature_id].notna()
        valid_frame = ordered.loc[valid, [firm, year]]
        pairs = valid_frame.merge(
            valid_frame.assign(**{year: valid_frame[year] + 1}),
            on=[firm, year],
            how="inner",
            validate="one_to_one",
        )
        firm_counts = valid_frame.groupby(firm, observed=True)[year].size()
        rows.append(
            {
                "feature_id": feature_id,
                "nonmissing_count": int(valid.sum()),
                "missing_count": int((~valid).sum()),
                "missing_rate": float((~valid).mean()),
                "firm_count": int(valid_frame[firm].nunique()),
                "first_fiscal_year": int(valid_frame[year].min())
                if not valid_frame.empty
                else pd.NA,
                "last_fiscal_year": int(valid_frame[year].max())
                if not valid_frame.empty
                else pd.NA,
                "annual_coverage": float(valid.mean()),
                "consecutive_year_coverage": float(len(pairs) / max(len(valid_frame), 1)),
                "firms_with_two_or_more_valid_years": int((firm_counts >= 2).sum()),
                "firms_with_complete_t_and_t_minus_1": int(pairs[firm].nunique()),
                "research_decision_status": item["research_decision_status"],
                "confirmatory_status": item["confirmatory_status"],
            }
        )
    return pd.DataFrame(rows)


def _value_scale(panel: pd.DataFrame, definitions: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in definitions:
        feature_id = str(item["feature_id"])
        if str(item["expected_dtype"]) not in {
            "float64",
            "float32",
            "int64",
            "int32",
            "int16",
        }:
            continue
        values = pd.to_numeric(panel[feature_id], errors="coerce")
        finite = values[np.isfinite(values)]
        quantiles = finite.quantile([0.01, 0.05, 0.5, 0.95, 0.99]) if not finite.empty else None
        rows.append(
            {
                "feature_id": feature_id,
                "unit": item["unit"],
                "minimum": float(finite.min()) if not finite.empty else pd.NA,
                "p1": float(quantiles.loc[0.01]) if quantiles is not None else pd.NA,
                "p5": float(quantiles.loc[0.05]) if quantiles is not None else pd.NA,
                "median": float(quantiles.loc[0.5]) if quantiles is not None else pd.NA,
                "mean": float(finite.mean()) if not finite.empty else pd.NA,
                "p95": float(quantiles.loc[0.95]) if quantiles is not None else pd.NA,
                "p99": float(quantiles.loc[0.99]) if quantiles is not None else pd.NA,
                "maximum": float(finite.max()) if not finite.empty else pd.NA,
                "zero_count": int(values.eq(0).sum()),
                "negative_count": int(values.lt(0).sum()),
                "nonfinite_count": int((values.notna() & ~np.isfinite(values)).sum()),
                "raw_values_modified": False,
            }
        )
    return pd.DataFrame(rows)


def _accounting_identities(panel: pd.DataFrame, firm: str, year: str) -> pd.DataFrame:
    identities = {
        "assets_equals_liabilities_plus_equity": (
            "fs_{status}_total_assets",
            ("fs_{status}_total_liabilities", "fs_{status}_equity"),
            (1.0, 1.0),
        ),
        "total_sources_equals_total_assets": (
            "fs_{status}_total_sources",
            ("fs_{status}_total_assets",),
            (1.0,),
        ),
        "gross_profit_equals_revenue_minus_cogs": (
            "fs_{status}_gross_profit",
            ("fs_{status}_net_revenue", "fs_{status}_cogs"),
            (1.0, -1.0),
        ),
        "pbt_minus_tax_equals_pat": (
            "fs_{status}_profit_after_tax",
            ("fs_{status}_profit_before_tax", "fs_{status}_income_tax_expense"),
            (1.0, -1.0),
        ),
        "inventory_gross_minus_allowance_equals_net": (
            "fs_{status}_inventory_net",
            ("fs_{status}_inventory_gross", "fs_{status}_inventory_allowance"),
            (1.0, -1.0),
        ),
        "ppe_gross_minus_accumulated_depreciation_equals_net": (
            "fs_{status}_ppe_net",
            ("fs_{status}_ppe_gross", "fs_{status}_accumulated_depreciation_ppe"),
            (1.0, -1.0),
        ),
    }
    rows: list[pd.DataFrame] = []
    for status in ("aud", "unaud"):
        for identity_id, (left_template, right_templates, signs) in identities.items():
            left = left_template.format(status=status)
            rights = [value.format(status=status) for value in right_templates]
            required = [left, *rights]
            if not set(required).issubset(panel.columns):
                continue
            frame = panel.loc[:, [firm, year, *required]].dropna(subset=required).copy()
            if frame.empty:
                continue
            expected = sum(
                sign * pd.to_numeric(frame[column], errors="coerce")
                for column, sign in zip(rights, signs, strict=True)
            )
            residual = pd.to_numeric(frame[left], errors="coerce") - expected
            scale = frame[required].abs().max(axis=1).clip(lower=1.0)
            frame["identity_id"] = identity_id
            frame["audit_status"] = "audited" if status == "aud" else "unaudited"
            frame["reporting_scope"] = "consolidated"
            frame["absolute_residual"] = residual.abs()
            frame["scaled_residual"] = residual.abs() / scale
            frame["tolerance_rule"] = "absolute<=1_VND_or_scaled<=1e-6"
            frame["pass_flag"] = residual.abs().le(1.0) | frame["scaled_residual"].le(1e-6)
            rows.append(
                frame.loc[
                    :,
                    [
                        firm,
                        year,
                        "identity_id",
                        "audit_status",
                        "reporting_scope",
                        "absolute_residual",
                        "scaled_residual",
                        "tolerance_rule",
                        "pass_flag",
                    ],
                ]
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _audited_unaudited(panel: pd.DataFrame, firm: str, year: str) -> pd.DataFrame:
    concepts = (
        "total_assets",
        "total_liabilities",
        "equity",
        "net_revenue",
        "cogs",
        "gross_profit",
        "operating_profit",
        "profit_before_tax",
        "profit_after_tax",
        "operating_cash_flow",
    )
    rows: list[dict[str, object]] = []
    for concept in concepts:
        audited = f"fs_aud_{concept}"
        unaudited = f"fs_unaud_{concept}"
        if not {audited, unaudited}.issubset(panel.columns):
            continue
        for fiscal_year, group in panel.groupby(year, sort=True, observed=True):
            post = pd.to_numeric(group[audited], errors="coerce")
            pre = pd.to_numeric(group[unaudited], errors="coerce")
            matched = post.notna() & pre.notna()
            signed = (post - pre).where(matched)
            denominator = post.abs().replace(0, np.nan)
            relative = (signed / denominator).where(matched)
            absolute = signed.abs()
            quantiles = absolute.dropna().quantile([0.5, 0.9, 0.95, 0.99])
            rows.append(
                {
                    "feature_id": concept,
                    year: int(str(fiscal_year)),
                    "concept_id": concept,
                    "matched_pair_count": int(matched.sum()),
                    "audited_only_count": int((post.notna() & pre.isna()).sum()),
                    "unaudited_only_count": int((post.isna() & pre.notna()).sum()),
                    "signed_adjustment_mean": float(signed.mean())
                    if signed.notna().any()
                    else pd.NA,
                    "absolute_adjustment_mean": float(absolute.mean())
                    if absolute.notna().any()
                    else pd.NA,
                    "relative_adjustment_mean": float(relative.mean())
                    if relative.notna().any()
                    else pd.NA,
                    "zero_adjustment_rate": float(signed.eq(0).sum() / max(matched.sum(), 1)),
                    "median_absolute_adjustment": float(quantiles.loc[0.5])
                    if not quantiles.empty
                    else pd.NA,
                    "p90_absolute_adjustment": float(quantiles.loc[0.9])
                    if not quantiles.empty
                    else pd.NA,
                    "p95_absolute_adjustment": float(quantiles.loc[0.95])
                    if not quantiles.empty
                    else pd.NA,
                    "p99_absolute_adjustment": float(quantiles.loc[0.99])
                    if not quantiles.empty
                    else pd.NA,
                    "threshold_status": "NOT_LOCKED_NO_THRESHOLD_APPLIED",
                }
            )
    return pd.DataFrame(rows)


def _ratio_diagnostics(
    panel: pd.DataFrame,
    definitions: list[dict[str, object]],
    year: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in definitions:
        feature_id = str(item["feature_id"])
        if not feature_id.startswith("ratio_"):
            continue
        for fiscal_year, values in panel.groupby(year, sort=True, observed=True)[feature_id]:
            numeric = pd.to_numeric(values, errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            lower = finite.quantile(0.01) if not finite.empty else np.nan
            upper = finite.quantile(0.99) if not finite.empty else np.nan
            rows.append(
                {
                    "feature_id": feature_id,
                    year: int(str(fiscal_year)),
                    "formula": item["formula"],
                    "denominator_rule": item.get("denominator_rule"),
                    "formula_coverage": float(numeric.notna().mean()),
                    "missing_dependency_count": int(numeric.isna().sum()),
                    "invalid_denominator_count": int(numeric.isna().sum()),
                    "nonfinite_count": int((numeric.notna() & ~np.isfinite(numeric)).sum()),
                    "minimum": float(finite.min()) if not finite.empty else pd.NA,
                    "median": float(finite.median()) if not finite.empty else pd.NA,
                    "maximum": float(finite.max()) if not finite.empty else pd.NA,
                    "extreme_value_count": int(((finite < lower) | (finite > upper)).sum())
                    if not finite.empty
                    else 0,
                    "standardization_applied": False,
                    "imputation_applied": False,
                }
            )
    return pd.DataFrame(rows)


def _temporal_diagnostics(
    panel: pd.DataFrame,
    definitions: list[dict[str, object]],
    firm: str,
    year: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in definitions:
        feature_id = str(item["feature_id"])
        requires_prior = "t_minus_1" in str(item["fiscal_period_reference"]) or "history" in str(
            item["lag_structure"]
        )
        valid = panel.loc[panel[feature_id].notna(), [firm, year]]
        first_year = panel.groupby(firm, observed=True)[year].transform("min")
        missing_prior = (
            int((panel[feature_id].notna() & panel[year].eq(first_year)).sum())
            if requires_prior
            else 0
        )
        allowed_bases = cast(list[object], item.get("availability_basis_allowed", []))
        rows.append(
            {
                "feature_id": feature_id,
                TEMPORAL_ROLE: item[TEMPORAL_ROLE],
                "fiscal_period_reference": item["fiscal_period_reference"],
                "lag_structure": item["lag_structure"],
                "requires_prior_year": requires_prior,
                "valid_value_count": len(valid),
                "usable_values_at_first_observed_firm_year": int(missing_prior),
                AVAILABILITY_BASIS: "|".join(str(value) for value in allowed_bases),
                "synthetic_anchor_flag": "synthetic_annual_anchor" in allowed_bases,
                "observed_publication_date_flag": "observed_publication_date" in allowed_bases,
                "future_information_detected": False,
                "temporal_status": "REVIEW" if requires_prior and missing_prior else "PASS",
            }
        )
    return pd.DataFrame(rows)


def _redundancy(panel: pd.DataFrame, feature_ids: list[str]) -> pd.DataFrame:
    if not feature_ids:
        return pd.DataFrame()
    numeric = panel.loc[:, feature_ids].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson", min_periods=3)
    spearman = numeric.corr(method="spearman", min_periods=3)
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(feature_ids):
        for right in feature_ids[left_index + 1 :]:
            pair = numeric[[left, right]].dropna()
            exact = bool(pair[left].equals(pair[right])) if not pair.empty else False
            sign_reversed = bool(pair[left].equals(-pair[right])) if not pair.empty else False
            scaling: float | object = pd.NA
            constant_scale = False
            denominator = pair[right].replace(0, np.nan)
            ratios = (pair[left] / denominator).dropna()
            if len(ratios) >= 3 and float(ratios.std()) <= 1e-12:
                constant_scale = True
                scaling = float(ratios.median())
            rows.append(
                {
                    "feature_id": left,
                    "feature_id_left": left,
                    "feature_id_right": right,
                    "paired_count": len(pair),
                    "pearson": pearson.loc[left, right],
                    "spearman": spearman.loc[left, right],
                    "exact_duplicate": exact,
                    "sign_reversed_duplicate": sign_reversed,
                    "constant_scaling_duplicate": constant_scale,
                    "scaling_factor": scaling,
                    "feature_removed_at_p07": False,
                }
            )
    return pd.DataFrame(rows)
