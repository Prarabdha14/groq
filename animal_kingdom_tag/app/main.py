from fastapi import FastAPI
from app.api.routes import router
from app.core.config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Ask natural language questions about the Animal Kingdom database. "
                "Powered by LangGraph for robust error-correction loops.",
    version="2.0.0"
)

app.include_router(router, prefix="/api")

@app.get("/")
def read_root():
    from app.api.routes import home
    return home()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
