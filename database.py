from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

# PostgreSQL URL — .env file mein DATABASE_URL set karo
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/cv_system"  # default fallback
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def get_db():
    """FastAPI dependency — DB session provide karta hai"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Sab tables create karo (agar exist na karein)"""
    from models import UserVideo, DistractionAlert  # noqa
    Base.metadata.create_all(bind=engine)
