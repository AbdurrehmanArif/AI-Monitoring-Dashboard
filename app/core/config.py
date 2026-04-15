import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    API_BASE: str = os.getenv("API_BASE", "http://127.0.0.1:8000")

settings = Settings()
