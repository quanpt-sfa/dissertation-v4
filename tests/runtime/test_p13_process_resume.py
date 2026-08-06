from __future__ import annotations

from core.resume_p13_v5 import (
    P13_PROCESS_ALLOWED_CHANGED_PATHS,
    P13_PROCESS_COMPATIBILITY_BASE_COMMIT,
    P13_PROCESS_COMPATIBILITY_PATCH_ID,
    P13_PROCESS_REQUIRED_CHANGED_PATHS,
    P13_PROCESS_REQUIRED_SOURCE_DRIFT,
)


def test_p13_process_compatibility_scope_is_exact_and_nonanalytical() -> None:
    assert P13_PROCESS_COMPATIBILITY_PATCH_ID == "P13_PROCESS_CHECKPOINT_EXECUTION_V5"
    assert (
        P13_PROCESS_COMPATIBILITY_BASE_COMMIT
        == "3510eae3681381c655e0b3501f7700e5aa10a5fa"
    )
    assert P13_PROCESS_REQUIRED_SOURCE_DRIFT == {
        "scripts/p13_sensitivity.py",
        "src/sensitivity/parallel_refits.py",
    }
    assert P13_PROCESS_REQUIRED_CHANGED_PATHS == {
        "scripts/p13_sensitivity.py",
        "src/sensitivity/parallel_refits.py",
        "scripts/run_pipeline_p13_process_resume.py",
        "src/core/resume_p13_v5.py",
    }
    assert P13_PROCESS_REQUIRED_CHANGED_PATHS.issubset(P13_PROCESS_ALLOWED_CHANGED_PATHS)
    assert "config/foundation/steps.yaml" not in P13_PROCESS_ALLOWED_CHANGED_PATHS
    assert "config/methodology/evaluation.yaml" not in P13_PROCESS_ALLOWED_CHANGED_PATHS
