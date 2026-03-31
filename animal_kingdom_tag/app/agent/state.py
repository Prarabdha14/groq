from typing import TypedDict, Optional

class AgentState(TypedDict):
    question: str
    generated_sql: Optional[str]
    sql_result: Optional[str]
    error: Optional[str]
    retries: int
    final_answer: Optional[str]
