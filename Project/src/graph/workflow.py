"""
Hierarchical StateGraph wiring.

Constructs a strict, linear supervisor pipeline:

    START -> planner -> priority -> reminder -> END

Each node reads and writes to the shared `AgentState`, with the planner
acting as the top-level supervisor that seeds the pipeline.
"""

from langgraph.graph import StateGraph, START, END

from src.graph.state import AgentState
from src.agents.planner import run_planner_agent
from src.agents.priority import run_priority_agent
from src.agents.reminder import run_reminder_agent


def build_workflow():
    """Construct and compile the linear hierarchical agent graph."""
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("planner", run_planner_agent)
    graph_builder.add_node("priority", run_priority_agent)
    graph_builder.add_node("reminder", run_reminder_agent)

    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "priority")
    graph_builder.add_edge("priority", "reminder")
    graph_builder.add_edge("reminder", END)

    return graph_builder.compile()


# Compiled, executable application — importable as `from src.graph.workflow import app`
app = build_workflow()