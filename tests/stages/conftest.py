from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _align_p08c_exit_policy_test_double(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep the legacy P08C exit-policy mock aligned with artifact types.

    ``test_p08c_exit_code_policies`` predates batch compaction and configures a
    single ``context.read`` side effect that returns a DataFrame for every
    artifact except the scenario registry. Production ``model_diagnostics`` is
    a JSON object, and P08C now reads it to recover compacted batch coordinates.

    This fixture changes only that test double. Production validation remains
    strict, and all other tests execute without this compatibility wrapper.
    """
    if request.node.name != "test_p08c_exit_code_policies":
        yield
        return

    import scripts.p08c_aggregate_batches as p08c

    original_main = p08c.main

    def compatible_main() -> int:
        # The test decorators patch load_run before calling p08c.main, so this
        # resolves the same MagicMock instance used by the test body.
        loaded = p08c.load_run(
            registry_path=None,
            run_id="test-double",
            step_id="P08",
            state="FEATURED",
        )
        read_mock = loaded.context.read
        original_side_effect = getattr(read_mock, "side_effect", None)
        if isinstance(original_side_effect, Callable):
            def typed_side_effect(
                artifact_id: str,
                *args: object,
                **kwargs: object,
            ) -> Any:
                if artifact_id == "model_diagnostics":
                    return {}
                return original_side_effect(artifact_id, *args, **kwargs)

            read_mock.side_effect = typed_side_effect
        return original_main()

    monkeypatch.setattr(p08c, "main", compatible_main)
    yield
