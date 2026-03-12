import os
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .state import AgentState
from .researcher import researcher_node
from .coder import coder_node
from .reviewer import reviewer_node


def route_after_reviewer(state: AgentState) -> Literal["researcher", "coder", "human_checkpoint", "__end__"]:
    decision = state.get("review_decision", "revise")
    cycle = state.get("review_cycle", 0)
    max_cycles = state.get("max_review_cycles", 3)
    human_enabled = os.getenv("HUMAN_CHECKPOINT_ENABLED", "true").lower() == "true"

    if decision == "approve":
        if human_enabled:
            return "human_checkpoint"
        return "__end__"

    if cycle >= max_cycles:
        # Force approval after max cycles
        return "human_checkpoint" if human_enabled else "__end__"

    # Revise: send back to researcher for context update, then coder
    return "coder"


def human_checkpoint_node(state: AgentState) -> AgentState:
    """Pause point for human review. The graph will interrupt here."""
    return {
        **state,
        "awaiting_human": True,
        "current_agent": "human",
    }


def route_after_human(state: AgentState) -> Literal["researcher", "__end__"]:
    feedback = state.get("human_feedback", "").strip()
    if feedback and feedback.lower() not in ("ok", "approve", "lgtm", "done"):
        # Human has feedback — restart from researcher
        return "researcher"
    return "__end__"


def finalize_node(state: AgentState) -> AgentState:
    return {
        **state,
        "final_output": state.get("code_output", ""),
        "current_agent": "end",
        "awaiting_human": False,
    }


def build_graph(db_path: str = None) -> StateGraph:
    db_path = db_path or os.getenv("SQLITE_DB_PATH", "./data/sessions.db")

    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("human_checkpoint", human_checkpoint_node)
    builder.add_node("finalize", finalize_node)

    # Entry point
    builder.set_entry_point("researcher")

    # Edges
    builder.add_edge("researcher", "coder")
    builder.add_edge("coder", "reviewer")

    builder.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "coder": "coder",
            "human_checkpoint": "human_checkpoint",
            "__end__": "finalize",
        },
    )

    builder.add_conditional_edges(
        "human_checkpoint",
        route_after_human,
        {
            "researcher": "researcher",
            "__end__": "finalize",
        },
    )

    builder.add_edge("finalize", END)

    return builder


async def create_compiled_graph(db_path: str = None):
    db_path = db_path or os.getenv("SQLITE_DB_PATH", "./data/sessions.db")
    dir_path = os.path.dirname(db_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    builder = build_graph(db_path)

    # Use from_conn_string without context manager so the connection stays open
    checkpointer = AsyncSqliteSaver.from_conn_string(db_path)
    await checkpointer.__aenter__()

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_checkpoint"],
    )
    return graph, checkpointer
