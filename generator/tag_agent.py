import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

load_dotenv()

db = SQLDatabase.from_uri(
    os.getenv("DATABASE_URL"),
    include_tables=["habitats", "species", "animals", "diet_logs", "observations"]
)
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

sql_chain = create_sql_query_chain(llm, db)
execute_tool = QuerySQLDataBaseTool(db=db)

app = FastAPI(
    title="Animal Kingdom TAG Agent",
    description="Ask natural language questions about the Animal Kingdom database. "
                "Uses LangChain + Groq LLM to auto-tag schema and generate SQL.",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    generated_sql: str
    result: str


@app.get("/", tags=["Info"])
def home():
    """Welcome endpoint."""
    return {
        "message": "Animal Kingdom TAG Agent is running!",
        "docs": "Go to /docs for interactive Swagger UI",
        "endpoints": {
            "/ask": "POST a question to get SQL + results",
            "/schema": "GET the auto-tagged database schema",
            "/tables": "GET table row counts",
        }
    }


@app.get("/schema", tags=["Schema"])
def get_schema():
    """View the auto-tagged database schema that LangChain reads."""
    return {"tagged_schema": db.get_table_info()}


@app.get("/tables", tags=["Schema"])
def get_tables():
    """Get row counts for all tables."""
    counts = {}
    for table in ["habitats", "species", "animals", "diet_logs", "observations"]:
        result = db.run(f"SELECT COUNT(*) FROM {table}")
        counts[table] = result
    return {"tables": counts}


@app.post("/ask", response_model=AnswerResponse, tags=["Query"])
def ask_question(req: QuestionRequest):
    """
    Ask a natural language question about the Animal Kingdom database.

    Examples:
    - "What do endangered birds eat?"
    - "How many mammals are there?"
    - "Show me sick animals in tropical habitats"
    - "Which habitat has the most species?"
    """
    generated_sql = sql_chain.invoke({"question": req.question})

    result = execute_tool.invoke(generated_sql)

    return AnswerResponse(
        question=req.question,
        generated_sql=generated_sql,
        result=result
    )


if __name__ == "__main__":
    import uvicorn
    print("Starting Animal Kingdom TAG Agent...")
    print("Open http://localhost:8000/docs for Swagger UI")
    uvicorn.run(app, host="0.0.0.0", port=8000)
