from pydantic import BaseModel
from typing import Optional

class QuestionRequest(BaseModel):
    question: str
    thread_id: Optional[str] = "default-thread"

class AnswerResponse(BaseModel):
    question: str
    generated_sql: Optional[str]
    result: str
    error: Optional[str] = None
    retries: int = 0
