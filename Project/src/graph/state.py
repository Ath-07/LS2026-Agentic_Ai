"""
State schema for the IITB Study/Deadline Assistant.

This module defines the single shared state object that flows through
every node in the LangGraph pipeline (planner -> priority -> reminder).
"""

from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    """
    Shared state passed between agents in the hierarchical graph.

    Attributes:
        raw_input: The initial unstructured schedule string provided by the student.
        tasks_backlog: Raw extracted tasks, each a dict with keys such as
            'title', 'track', and 'original_deadline'.
        prioritized_list: Ranked tasks with an added numeric 'priority_score'
            and 'urgency_days' field, sorted from most to least urgent.
        telegram_status: Final delivery status, either "SUCCESS" or "FAILED".
    """
    raw_input: str
    tasks_backlog: List[Dict[str, Any]]
    prioritized_list: List[Dict[str, Any]]
    telegram_status: str