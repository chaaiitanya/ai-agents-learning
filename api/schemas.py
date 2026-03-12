from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TaskRequest(BaseModel):
    task: str
    max_review_cycles: int = 3


class HumanFeedbackRequest(BaseModel):
    session_id: str
    feedback: str


class IngestRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None


class SessionResponse(BaseModel):
    session_id: str
    status: str
    task: str
    created_at: datetime


class RunResponse(BaseModel):
    session_id: str
    status: str
    current_agent: str
    awaiting_human: bool
    research_output: Optional[str] = None
    code_output: Optional[str] = None
    review_output: Optional[str] = None
    review_cycle: int = 0
    final_output: Optional[str] = None
    error: Optional[str] = None
