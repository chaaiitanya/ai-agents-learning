from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Core task
    task: str
    session_id: str

    # Message history (append-only via add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Agent outputs
    research_output: str
    code_output: str
    review_output: str

    # Workflow control
    current_agent: Literal["researcher", "coder", "reviewer", "human", "end"]
    review_cycle: int
    max_review_cycles: int
    review_decision: Literal["approve", "revise", "reject", ""]

    # Human-in-the-loop
    awaiting_human: bool
    human_feedback: str

    # Final output
    final_output: str
    error: str
