from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

if not settings.DATABASE_URL:
    raise ValueError("DATABASE_URL is missing in environment variables.")

# Neon Postgres uses connection pooling, but standard create_engine works well
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # In real world, use Alembic, but we'll use create_all for simplicity
    Base.metadata.create_all(bind=engine)
