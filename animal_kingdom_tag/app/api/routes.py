from fastapi import APIRouter
from app.models.schemas import QuestionRequest, AnswerResponse
from app.agent.graph import tag_agent_graph_schema, get_checkpointer
from app.core.database import db

router = APIRouter()

@router.get("/", tags=["Info"])
def home():
    """Welcome endpoint."""
    return {
        "message": "Production Animal Kingdom TAG Agent is running!",
        "architecture": "LangGraph + Modular Monolith",
        "endpoints": {
            "/api/ask": "POST a question to get a generated answer",
            "/api/schema": "GET the DB schema",
            "/api/tables": "GET table row counts",
        }
    }

@router.get("/schema", tags=["Schema"])
def get_schema():
    """View the auto-tagged database schema that LangChain reads."""
    return {"tagged_schema": db.get_table_info()}

@router.get("/tables", tags=["Schema"])
def get_tables():
    """Get row counts for all tables."""
    counts = {}
    for table in ["habitats", "species", "animals", "diet_logs", "observations"]:
        result = db.run(f"SELECT COUNT(*) FROM {table}")
        counts[table] = result
    return {"tables": counts}

@router.post("/ask", response_model=AnswerResponse, tags=["Query"])
def ask_question(req: QuestionRequest):
    """Ask a natural language question. Handled by LangGraph."""
    
    config = {"configurable": {"thread_id": req.thread_id}}
    
    with get_checkpointer() as checkpointer:
        tag_agent_graph = tag_agent_graph_schema.compile(checkpointer=checkpointer)
        final_state = tag_agent_graph.invoke(
            {"question": req.question, "retries": 0},
            config=config
        )

    return AnswerResponse(
        question=final_state.get("question", req.question),
        generated_sql=final_state.get("generated_sql"),
        result=final_state.get("final_answer", "No answer generated."),
        error=final_state.get("error"),
        retries=final_state.get("retries", 0)
    )
