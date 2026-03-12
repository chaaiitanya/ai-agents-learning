import os
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File
from langchain_core.messages import HumanMessage

from .schemas import TaskRequest, HumanFeedbackRequest, IngestRequest, RunResponse
from agents import create_compiled_graph, AgentState
from rag import VectorStore

router = APIRouter()

# In-memory graph registry (for demo; production should use persistent store)
_graphs: dict = {}


async def _get_or_create_graph(session_id: str):
    if session_id not in _graphs:
        graph, checkpointer = await create_compiled_graph()
        _graphs[session_id] = (graph, checkpointer)
    return _graphs[session_id]


@router.post("/run", response_model=RunResponse)
async def run_task(request: TaskRequest):
    session_id = str(uuid.uuid4())
    graph, _ = await _get_or_create_graph(session_id)

    initial_state: AgentState = {
        "task": request.task,
        "session_id": session_id,
        "messages": [HumanMessage(content=request.task)],
        "research_output": "",
        "code_output": "",
        "review_output": "",
        "current_agent": "researcher",
        "review_cycle": 0,
        "max_review_cycles": request.max_review_cycles,
        "review_decision": "",
        "awaiting_human": False,
        "human_feedback": "",
        "final_output": "",
        "error": "",
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RunResponse(
        session_id=session_id,
        status="paused" if final_state.get("awaiting_human") else "completed",
        current_agent=final_state.get("current_agent", ""),
        awaiting_human=final_state.get("awaiting_human", False),
        research_output=final_state.get("research_output"),
        code_output=final_state.get("code_output"),
        review_output=final_state.get("review_output"),
        review_cycle=final_state.get("review_cycle", 0),
        final_output=final_state.get("final_output"),
        error=final_state.get("error"),
    )


@router.post("/feedback", response_model=RunResponse)
async def submit_feedback(request: HumanFeedbackRequest):
    session_id = request.session_id
    if session_id not in _graphs:
        raise HTTPException(status_code=404, detail="Session not found")

    graph, _ = _graphs[session_id]
    config = {"configurable": {"thread_id": session_id}}

    # Get current state and inject human feedback
    current_state = await graph.aget_state(config)
    updated_values = {
        "human_feedback": request.feedback,
        "awaiting_human": False,
    }
    await graph.aupdate_state(config, updated_values, as_node="human_checkpoint")

    # Resume execution
    try:
        final_state = await graph.ainvoke(None, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RunResponse(
        session_id=session_id,
        status="paused" if final_state.get("awaiting_human") else "completed",
        current_agent=final_state.get("current_agent", ""),
        awaiting_human=final_state.get("awaiting_human", False),
        research_output=final_state.get("research_output"),
        code_output=final_state.get("code_output"),
        review_output=final_state.get("review_output"),
        review_cycle=final_state.get("review_cycle", 0),
        final_output=final_state.get("final_output"),
        error=final_state.get("error"),
    )


@router.post("/ingest/text")
async def ingest_text(request: IngestRequest):
    vs = VectorStore()
    ids = vs.ingest_text(request.text, request.metadata)
    return {"ingested": len(ids), "ids": ids}


@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    import tempfile, shutil
    suffix = "." + file.filename.split(".")[-1] if "." in file.filename else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    vs = VectorStore()
    ids = vs.ingest_file(tmp_path)
    os.unlink(tmp_path)
    return {"filename": file.filename, "ingested": len(ids), "ids": ids}


@router.get("/health")
async def health():
    return {"status": "ok"}
