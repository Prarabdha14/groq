import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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


def extract_sql(llm_output: str) -> str:
    """Extract clean SQL from LLM response that may contain explanations."""
    sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', llm_output, re.DOTALL)
    if sql_blocks:
        return sql_blocks[0].strip()

    code_blocks = re.findall(r'```\s*(.*?)\s*```', llm_output, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    cleaned = llm_output.strip()
    if cleaned.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
        return cleaned

    match = re.search(r'(SELECT\s+.*?;)', llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r'(SELECT\s+.+)', llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return cleaned


def clean_sql(sql: str) -> str:
    """Fix common LLM SQL quoting mistakes for PostgreSQL."""
    # Fix "table.column" → table.column (remove quotes around dotted names)
    sql = re.sub(r'"(\w+)\.(\w+)"', r'\1.\2', sql)
    # Fix "table"."column" → table.column (remove unnecessary quotes)
    sql = re.sub(r'"(\w+)"\."(\w+)"', r'\1.\2', sql)
    # Fix "alias" → alias for simple table aliases
    sql = re.sub(r'"(\w{1,3})"\.', r'\1.', sql)
    # Remove trailing semicolons (SQLAlchemy doesn't like them)
    sql = sql.rstrip(';').strip()
    return sql


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
    raw_output = sql_chain.invoke({"question": req.question})
    clean = clean_sql(extract_sql(raw_output))

    try:
        result = execute_tool.invoke(clean)
    except Exception as e:
        result = f"SQL execution error: {str(e)[:200]}"

    return AnswerResponse(
        question=req.question,
        generated_sql=clean,
        result=result
    )


if __name__ == "__main__":
    import uvicorn
    print("Starting Animal Kingdom TAG Agent...")
    print("Open http://localhost:8000/docs for Swagger UI")
    uvicorn.run(app, host="0.0.0.0", port=8000)
