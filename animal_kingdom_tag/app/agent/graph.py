import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from app.agent.state import AgentState
from app.agent.nodes import generate_sql_node, execute_sql_node, generate_answer_node
from app.core.config import settings

log = logging.getLogger("animal_kingdom.graph")

def route_after_sql(state: AgentState) -> str:
    """Determine where to go after SQL execution."""
    error = state.get("error")
    if not error:
        return "generate_answer"
    
    if error and state.get("retries", 0) < 3:
        log.warning("Graph Routing: Re-trying SQL generation due to error.")
        return "generate_sql"
        
    log.error("Graph Routing: Out of retries, proceeding to answer node.")
    return "generate_answer"


workflow = StateGraph(AgentState)

workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("generate_answer", generate_answer_node)

workflow.set_entry_point("generate_sql")
workflow.add_edge("generate_sql", "execute_sql")

workflow.add_conditional_edges(
    "execute_sql",
    route_after_sql,
    {
        "generate_sql": "generate_sql",
        "generate_answer": "generate_answer"
    }
)

workflow.add_edge("generate_answer", END)

# Instead of a global checkpointer using a 'with' block context everywhere,
# For this simplified FastAPI pattern, we'll initialize the saver via connect string
# which manages its own connection pool safely.
import contextlib

@contextlib.contextmanager
def get_checkpointer():
    with PostgresSaver.from_conn_string(settings.DATABASE_URL) as saver:
        saver.setup()
        yield saver

# We don't pre-compile the persistent graph as a module-level global
# because the PostgresSaver strongly prefers working within a context manager.
# We will compile it in the execution function itself.
tag_agent_graph_schema = workflow
