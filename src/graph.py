"""LangGraph StateGraph wiring and deterministic Critic routing."""

from typing import Literal
from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.nodes import critic_node, researcher_node, writer_node
from src.schemas import ResearchState


def should_revise(state: ResearchState) -> Literal["writer", "__end__"]:
    """Route back to Writer if Critic rejects and revisions remain; else terminate."""
    if not state.get("sources"):
        return END

    critique = state.get("critique")
    if critique is None or critique.passed:
        return END

    if state.get("revision_count", 0) >= settings.max_revisions:
        return END

    return "writer"


def build_research_graph():
    """Compile the cyclic Researcher -> Writer -> Critic state machine."""
    builder = StateGraph(ResearchState)

    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges(
        "critic",
        should_revise,
        {
            "writer": "writer",
            END: END,
        },
    )

    return builder.compile()
