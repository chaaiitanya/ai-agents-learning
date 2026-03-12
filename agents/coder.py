from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from .state import AgentState
from .llm import get_llm

SYSTEM_PROMPT = """You are a specialized Coder Agent. Your job is to:
1. Read and understand the research findings provided
2. Implement a clean, working solution based on those findings
3. Follow best practices: proper error handling, type hints, docstrings where needed
4. Write production-quality code that is readable and maintainable
5. Include any necessary setup instructions or dependencies

If this is a revision cycle, incorporate all reviewer feedback and human feedback precisely.
Always output complete, runnable code with proper structure."""


def coder_node(state: AgentState) -> AgentState:
    llm = get_llm()

    task = state["task"]
    research = state.get("research_output", "")
    review = state.get("review_output", "")
    human_feedback = state.get("human_feedback", "")
    cycle = state.get("review_cycle", 0)

    content = f"Task: {task}\n\nResearch Findings:\n{research}"

    if cycle > 0 and review:
        content += f"\n\nReviewer Feedback (Cycle {cycle}):\n{review}"
    if human_feedback:
        content += f"\n\nHuman Feedback:\n{human_feedback}"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    response = llm.invoke(messages)
    code_output = response.content

    return {
        **state,
        "code_output": code_output,
        "current_agent": "reviewer",
        "messages": [AIMessage(content=code_output, name="coder")],
    }
