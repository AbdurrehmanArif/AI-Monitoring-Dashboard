import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MONGO_URL: str = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "ai_monitoring_dashboard")
    API_BASE: str = os.getenv("API_BASE", "http://127.0.0.1:8000")

settings = Settings()