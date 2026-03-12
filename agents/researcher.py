from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from .state import AgentState
from .llm import get_llm
from rag import RAGRetriever, VectorStore

SYSTEM_PROMPT = """You are a specialized Research Agent. Your job is to:
1. Analyze the given task thoroughly
2. Search through available knowledge sources to gather relevant information
3. Synthesize findings into a clear, structured research report
4. Identify key requirements, constraints, and technical considerations
5. Provide context that will help the Coder agent implement the solution

Be thorough, factual, and well-organized. Structure your output with clear sections."""


def researcher_node(state: AgentState) -> AgentState:
    llm = get_llm()
    vector_store = VectorStore()
    retriever = RAGRetriever(vector_store)

    task = state["task"]
    human_feedback = state.get("human_feedback", "")

    # Retrieve relevant context from RAG
    rag_context = retriever.retrieve_as_text(task, k=5)

    # Build prompt
    user_content = f"Task: {task}"
    if human_feedback:
        user_content += f"\n\nHuman Feedback to incorporate: {human_feedback}"
    if rag_context and rag_context != "No relevant documents found.":
        user_content += f"\n\nRelevant context from knowledge base:\n{rag_context}"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    research_output = response.content

    return {
        **state,
        "research_output": research_output,
        "current_agent": "coder",
        "messages": [AIMessage(content=research_output, name="researcher")],
    }
