import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Animal Kingdom TAG Service"
    DATABASE_URL: str
    GROQ_API_KEY: str

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
