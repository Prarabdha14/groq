from langchain_groq import ChatGroq
from app.core.config import settings

def get_llm():
    return ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0
    )

llm = get_llm()
