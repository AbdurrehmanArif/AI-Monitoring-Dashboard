from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from database.models import save_alert, get_all_alerts
import os

from app.core.database import init_db
from app.api.routes import router

# ── App setup ────────────────────────────────────────────
app = FastAPI(
    title="CV Unified System API",
    description="Mobile Detection + Face Recognition — Merged Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# DB tables create karo on startup
@app.on_event("startup")
def startup():
    init_db()
    os.makedirs(os.path.join(PROJECT_ROOT, "screenshots"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "known_faces"), exist_ok=True)


# ── Root ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "CV Unified System API — Active ✅"}

# Include routers
app.include_router(router)
