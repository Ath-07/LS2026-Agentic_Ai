"""
Planner Agent (Supervisor Node).

Responsible for parsing the student's raw, chaotic schedule string and
segmenting it into discrete, classified task items. This is the first
node in the hierarchical pipeline and produces `tasks_backlog`.
"""

import re
from datetime import datetime
from typing import Dict, Any, List

from src.graph.state import AgentState

# --- Track classification keyword banks -------------------------------------

_IITB_KEYWORDS = [
    r"\bcs\d{3}\b", r"\bma\d{3}\b", r"\bee\d{3}\b", r"\bme\d{3}\b",
    r"\bhs\d{3}\b", r"\bph\d{3}\b", r"\blab\b", r"\bassignment\b",
    r"\bquiz\b", r"\bmidsem\b", r"\bendsem\b", r"\blecture\b",
    r"\btutorial\b", r"\bhomework\b", r"\bexam\b",
]

_SOS_KEYWORDS = [
    r"\bsos\b", r"summer of science", r"\bcheckpoint\b", r"\bmilestone\b",
    r"\bresearch\b", r"\bmentor\b", r"\breport\b", r"\bdocker\b",
]

_LEETCODE_KEYWORDS = [
    r"\bleetcode\b", r"\bdp\b", r"dynamic programming", r"\bquestions?\b",
    r"\bproblems?\b", r"\bgrind\b", r"\bmedium\b", r"\beasy\b", r"\bhard\b",
    r"\bgraph\b", r"\btree\b", r"\bdsa\b",
]

_WEEKDAYS = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]


def _classify_track(segment: str) -> str:
    """Classify a task segment into one of the three known tracks."""
    text = segment.lower()

    for pattern in _IITB_KEYWORDS:
        if re.search(pattern, text):
            return "IITB Coursework"
    for pattern in _SOS_KEYWORDS:
        if re.search(pattern, text):
            return "SoS Project"
    for pattern in _LEETCODE_KEYWORDS:
        if re.search(pattern, text):
            return "LeetCode"

    # Default fallback: unclassified tasks are treated as coursework-adjacent
    # miscellaneous work rather than silently dropped.
    return "IITB Coursework"


def _extract_deadline(segment: str) -> str:
    """Extract a human-readable deadline token from a task segment."""
    text = segment.lower()

    if "today" in text:
        return "Today"
    if "tomorrow" in text:
        return "Tomorrow"

    for day in _WEEKDAYS:
        if day in text:
            return day.capitalize()

    return "Unspecified"


def _split_into_segments(raw_input: str) -> List[str]:
    """
    Split a chaotic schedule string into discrete task segments.

    Splits on commas, semicolons, newlines, and the word 'and', then
    strips whitespace and drops empty fragments.
    """
    if not raw_input or not raw_input.strip():
        return []

    # Normalize separators: newlines/semicolons -> commas, then split on
    # commas and standalone 'and' conjunctions.
    normalized = raw_input.replace("\n", ",").replace(";", ",")
    rough_segments = re.split(r",|\band\b", normalized, flags=re.IGNORECASE)

    segments = [seg.strip(" .\t") for seg in rough_segments]
    segments = [seg for seg in segments if seg]
    return segments


def run_planner_agent(state: AgentState) -> Dict[str, Any]:
    """
    Parse `state['raw_input']` into a structured `tasks_backlog`.

    Args:
        state: Current graph state containing the raw schedule string.

    Returns:
        A partial state update dict containing the new `tasks_backlog`.
    """
    raw_input = state.get("raw_input", "")

    try:
        segments = _split_into_segments(raw_input)

        if not segments:
            print("[PLANNER WARNING] No parsable tasks found in raw_input.")
            return {"tasks_backlog": []}

        tasks_backlog: List[Dict[str, Any]] = []
        for segment in segments:
            track = _classify_track(segment)
            deadline = _extract_deadline(segment)
            title = segment.strip()

            tasks_backlog.append({
                "title": title,
                "track": track,
                "original_deadline": deadline,
            })

        print(f"[PLANNER] Extracted {len(tasks_backlog)} task(s) from raw_input.")
        return {"tasks_backlog": tasks_backlog}

    except Exception as exc:  # noqa: BLE001
        print(f"[PLANNER ERROR] Failed to parse raw_input: {exc}")
        # Fail gracefully with an empty backlog rather than crashing the graph.
        return {"tasks_backlog": []}