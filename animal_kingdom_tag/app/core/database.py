from langchain_community.utilities import SQLDatabase
from app.core.config import settings

def get_db() -> SQLDatabase:
    return SQLDatabase.from_uri(
        settings.DATABASE_URL,
        include_tables=["habitats", "species", "animals", "diet_logs", "observations"]
    )

db = get_db()
