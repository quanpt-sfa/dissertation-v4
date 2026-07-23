"""Replication plan contract."""

from core.pipeline import mapping
from simulation.method_contract import (
    IMBALANCE_TREATMENT_ID,
    LEARNER_TIER,
    METHOD_FAMILY,
    TRAINING_COST_REGIME_ID,
)


def replication_plan(
    simulation: dict[str, object],
    method_spec: dict[str, object],
) -> dict[str, int]:
    """Dynamically determine the replication requirements and batch size for a method.

    Returns a dict with keys: 'minimum', 'batch_size', and 'maximum'.
    """
    family = str(method_spec[METHOD_FAMILY])
    tier = str(method_spec[LEARNER_TIER])
    training_cost = str(method_spec[TRAINING_COST_REGIME_ID])
    imbalance_treatment = str(method_spec[IMBALANCE_TREATMENT_ID])

    if family == "standalone_estimator":
        settings = mapping(simulation.get("l3"), "simulation.l3")
        return {
            "minimum": int(settings["initial_replications"]),
            "batch_size": int(settings["batch_size"]),
            "maximum": int(settings["maximum_replications"]),
        }
    if imbalance_treatment not in {"none", "not_applicable"} or tier in {
        "extended",
        "methodological",
    }:
        settings = mapping(
            simulation.get("extended_replication"),
            "simulation.extended_replication",
        )
        return {
            "minimum": int(settings["minimum_replications"]),
            "batch_size": int(settings["batch_size"]),
            "maximum": int(settings["maximum_replications"]),
        }

    neutral_id = str(
        mapping(
            simulation.get("cost_sensitive"),
            "simulation.cost_sensitive",
        )["cost_neutral_regime_id"]
    )
    if training_cost != neutral_id:
        settings = mapping(
            simulation.get("cost_sensitive_replication"),
            "simulation.cost_sensitive_replication",
        )
        return {
            "minimum": int(settings["minimum_replications"]),
            "batch_size": int(settings["batch_size"]),
            "maximum": int(settings["maximum_replications"]),
        }

    settings = mapping(simulation.get("core"), "simulation.core")
    return {
        "minimum": int(settings["minimum_replications"]),
        "batch_size": int(settings["batch_size"]),
        "maximum": int(settings["maximum_replications"]),
    }
