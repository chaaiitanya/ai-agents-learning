from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from typing import Literal

from .state import AgentState
from .llm import get_llm

SYSTEM_PROMPT = """You are a specialized Code Reviewer Agent. Your job is to:
1. Critically evaluate the code produced by the Coder agent
2. Check for: correctness, security issues, performance, readability, edge cases
3. Verify the code fully addresses the original task requirements
4. Provide specific, actionable feedback if revisions are needed

Your response MUST follow this exact format:

DECISION: <approve|revise|reject>

REASONING:
<detailed explanation>

FEEDBACK (if revise/reject):
<specific, actionable items the coder must address>

Only use "approve" if the code is complete, correct, and production-ready.
Use "revise" for fixable issues. Use "reject" only for fundamentally flawed approaches."""


def reviewer_node(state: AgentState) -> AgentState:
    llm = get_llm()

    task = state["task"]
    research = state.get("research_output", "")
    code = state.get("code_output", "")
    cycle = state.get("review_cycle", 0)
    max_cycles = state.get("max_review_cycles", 3)

    content = (
        f"Original Task: {task}\n\n"
        f"Research Context:\n{research}\n\n"
        f"Code to Review (Cycle {cycle + 1}/{max_cycles}):\n{code}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    response = llm.invoke(messages)
    review_output = response.content

    # Parse decision from response
    decision = _parse_decision(review_output)

    return {
        **state,
        "review_output": review_output,
        "review_decision": decision,
        "review_cycle": cycle + 1,
        "current_agent": "human",
        "messages": [AIMessage(content=review_output, name="reviewer")],
    }


def _parse_decision(text: str) -> Literal["approve", "revise", "reject"]:
    for line in text.splitlines():
        if line.strip().upper().startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ("approve", "revise", "reject"):
                return val
    # Default: if no clear decision found, request revision
    return "revise"
