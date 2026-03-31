import logging
from app.agent.state import AgentState
from app.core.llm import llm
from app.core.database import db
from app.core.prompts import sql_prompt_template, answer_prompt_template
from app.utils.parsers import extract_sql, clean_sql
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser

log = logging.getLogger("animal_kingdom.agent")

execute_tool = QuerySQLDataBaseTool(db=db)
sql_chain = sql_prompt_template | llm | StrOutputParser()
answer_chain = answer_prompt_template | llm | StrOutputParser()

def generate_sql_node(state: AgentState) -> dict:
    log.info(f"Generating SQL. Retries: {state.get('retries', 0)}")
    error_context = ""
    if state.get("error"):
        error_context = f"PREVIOUS ERROR (fix this!): {state['error']}"
    
    schema = db.get_table_info()
    
    raw_output = sql_chain.invoke({
        "question": state["question"],
        "table_info": schema,
        "input": state["question"],
        "top_k": 5,
        "error_context": error_context
    })
    
    clean = clean_sql(extract_sql(raw_output))
    return {"generated_sql": clean}


def execute_sql_node(state: AgentState) -> dict:
    sql = state["generated_sql"]
    log.info(f"Executing SQL: {sql}")
    retries = state.get('retries', 0)
    
    if any(forbidden in sql.upper() for forbidden in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER"]):
        return {
            "error": "Cannot execute DML statements. Only SELECT is allowed.",
            "retries": retries + 1,
            "sql_result": None
        }

    try:
        result = execute_tool.invoke(sql)
        if isinstance(result, str) and result.strip().startswith("Error:"):
            return {
                "error": result,
                "retries": retries + 1,
                "sql_result": None
            }
        return {
            "sql_result": str(result),
            "error": None
        }
    except Exception as e:
        log.warning(f"SQL Execution Error: {e}")
        return {
            "error": str(e),
            "retries": retries + 1,
            "sql_result": None
        }


def generate_answer_node(state: AgentState) -> dict:
    log.info("Generating final natural language answer.")
    if state.get("error") and state.get("retries", 0) >= 3:
        log.error("Max retries reached. Returning error answer.")
        return {
            "final_answer": f"I hit an error trying to query the database and couldn't resolve it. Details: {state['error']}"
        }
    
    sql = state.get("generated_sql", "")
    result = state.get("sql_result", "")
    question = state.get("question", "")
    
    try:
        natural_answer = answer_chain.invoke({
            "question": question,
            "query": sql,
            "result": result
        })
        return {"final_answer": natural_answer}
    except Exception as e:
        log.warning(f"Answer generation failed: {e}")
        return {"final_answer": f"Raw Result: {result}"}
