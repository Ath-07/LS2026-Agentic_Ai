"""
Priority Agent.

Consumes the raw `tasks_backlog` and computes an explicit numeric
priority score per task, based on:
  1. Track weight (IITB Coursework > SoS Project > LeetCode)
  2. Deadline urgency (fewer days remaining => higher priority)

Produces the sorted `prioritized_list`.
"""

from datetime import datetime
from typing import Dict, Any, List

from src.graph.state import AgentState

# Higher weight = inherently more important track.
_TRACK_WEIGHTS = {
    "IITB Coursework": 30,
    "SoS Project": 20,
    "LeetCode": 10,
}

_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_UNSPECIFIED_URGENCY_DAYS = 14  # treat unknown deadlines as low urgency
_URGENCY_SCALE = 2.0            # points deducted per day of remaining slack


def _compute_urgency_days(deadline_label: str) -> int:
    """
    Convert a deadline label ('Today', 'Tomorrow', 'Friday', 'Unspecified')
    into an integer number of days from now.
    """
    label = deadline_label.strip().lower()

    if label == "today":
        return 0
    if label == "tomorrow":
        return 1
    if label in _WEEKDAY_INDEX:
        today_idx = datetime.now().weekday()
        target_idx = _WEEKDAY_INDEX[label]
        delta = (target_idx - today_idx) % 7
        # If the named day is "today" by weekday match, treat as imminent (1),
        # not 0, since it wasn't explicitly stated as "today".
        return delta if delta != 0 else 7
    return _UNSPECIFIED_URGENCY_DAYS


def _compute_priority_score(track: str, urgency_days: int) -> float:
    """
    Combine track weight and urgency into a single descending priority score.
    Higher score == handle sooner.
    """
    track_weight = _TRACK_WEIGHTS.get(track, 5)
    urgency_penalty = urgency_days * _URGENCY_SCALE
    return round(track_weight - urgency_penalty, 2)


def run_priority_agent(state: AgentState) -> Dict[str, Any]:
    """
    Rank `state['tasks_backlog']` into a sorted `prioritized_list`.

    Args:
        state: Current graph state containing `tasks_backlog`.

    Returns:
        A partial state update dict containing the sorted `prioritized_list`.
    """
    tasks_backlog = state.get("tasks_backlog", [])

    if not tasks_backlog:
        print("[PRIORITY WARNING] tasks_backlog is empty; nothing to prioritize.")
        return {"prioritized_list": []}

    try:
        prioritized_list: List[Dict[str, Any]] = []

        for task in tasks_backlog:
            track = task.get("track", "LeetCode")
            deadline_label = task.get("original_deadline", "Unspecified")

            urgency_days = _compute_urgency_days(deadline_label)
            priority_score = _compute_priority_score(track, urgency_days)

            enriched_task = dict(task)
            enriched_task["urgency_days"] = urgency_days
            enriched_task["priority_score"] = priority_score
            prioritized_list.append(enriched_task)

        # Highest priority_score first (most urgent / most important track).
        prioritized_list.sort(key=lambda t: t["priority_score"], reverse=True)

        print(f"[PRIORITY] Ranked {len(prioritized_list)} task(s).")
        return {"prioritized_list": prioritized_list}

    except Exception as exc:  # noqa: BLE001
        print(f"[PRIORITY ERROR] Failed to rank tasks_backlog: {exc}")
        # Fall back to the original, unranked order rather than losing data.
        return {"prioritized_list": tasks_backlog}