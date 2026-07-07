"""
Integration test for the full hierarchical LangGraph pipeline.

Run directly from the project root:

    python tests/test_agents.py

Exercises the compiled `app` (planner -> priority -> reminder) end to
end against a realistic mock schedule string and verifies that each
stage of state is populated as expected.
"""

import os
import sys

# Ensure the project root is on sys.path so `from src...` imports resolve
# when this script is run directly rather than via a test runner/package.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.graph.workflow import app  # noqa: E402  (import after sys.path fix)

MOCK_INPUT = (
    "Prepare for CS 302 midsem due tomorrow, "
    "fix SoS git bug by Friday, "
    "and solve 2 LeetCode arrays problems."
)


def test_full_pipeline() -> bool:
    """
    Invoke the full compiled graph on a mock schedule and assert that
    each stage of the state was populated correctly.

    Returns:
        True if all assertions pass, False if any check fails.
    """
    print("=" * 60)
    print("Running full pipeline integration test")
    print("=" * 60)
    print(f"Mock input:\n  {MOCK_INPUT}\n")

    try:
        final_state = app.invoke({"raw_input": MOCK_INPUT})
    except Exception as exc:  # noqa: BLE001
        print(f"[TEST_AGENTS ERROR] Graph invocation raised an exception: {exc}")
        return False

    all_checks_passed = True

    # --- tasks_backlog checks -------------------------------------------------
    tasks_backlog = final_state.get("tasks_backlog")
    try:
        assert tasks_backlog is not None, "tasks_backlog key is missing from final state"
        assert isinstance(tasks_backlog, list), "tasks_backlog is not a list"
        assert len(tasks_backlog) > 0, "tasks_backlog is empty"
        for task in tasks_backlog:
            assert "title" in task, f"Task missing 'title': {task}"
            assert "track" in task, f"Task missing 'track': {task}"
            assert "original_deadline" in task, f"Task missing 'original_deadline': {task}"
        print(f"✅ tasks_backlog populated correctly ({len(tasks_backlog)} tasks).")
    except AssertionError as exc:
        print(f"❌ tasks_backlog check FAILED: {exc}")
        all_checks_passed = False

    # --- prioritized_list checks -----------------------------------------------
    prioritized_list = final_state.get("prioritized_list")
    try:
        assert prioritized_list is not None, "prioritized_list key is missing from final state"
        assert isinstance(prioritized_list, list), "prioritized_list is not a list"
        assert len(prioritized_list) > 0, "prioritized_list is empty"
        for task in prioritized_list:
            assert "priority_score" in task, f"Task missing 'priority_score': {task}"

        scores = [task["priority_score"] for task in prioritized_list]
        assert scores == sorted(scores, reverse=True), (
            f"prioritized_list is not sorted in descending priority order: {scores}"
        )
        print(f"✅ prioritized_list populated and sorted correctly ({len(prioritized_list)} tasks).")
    except AssertionError as exc:
        print(f"❌ prioritized_list check FAILED: {exc}")
        all_checks_passed = False

    # --- telegram_status check (informational, not delivery-dependent) --------
    telegram_status = final_state.get("telegram_status")
    if telegram_status in ("SUCCESS", "FAILED"):
        print(f"ℹ️  telegram_status returned as expected: '{telegram_status}'.")
    else:
        print(f"❌ telegram_status has an unexpected value: {telegram_status!r}")
        all_checks_passed = False

    # --- summary ----------------------------------------------------------------
    print("\n--- Final tasks_backlog ---")
    for task in tasks_backlog or []:
        print(f"  - [{task.get('track')}] {task.get('title')} (due: {task.get('original_deadline')})")

    print("\n--- Final prioritized_list ---")
    for task in prioritized_list or []:
        print(
            f"  - [{task.get('track')}] {task.get('title')} "
            f"(score: {task.get('priority_score')}, urgency_days: {task.get('urgency_days')})"
        )

    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED — pipeline integration test succeeded.")
    else:
        print("❌ ONE OR MORE CHECKS FAILED — see details above.")
    print("=" * 60)

    return all_checks_passed


if __name__ == "__main__":
    passed = test_full_pipeline()
    sys.exit(0 if passed else 1)