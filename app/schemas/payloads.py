from pydantic import BaseModel
from datetime import datetime
from typing import Optional


# ── UserVideo Schemas ─────────────────────────────────────
class UserVideoResponse(BaseModel):
    id: int
    user_id: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── DistractionAlert Schemas ──────────────────────────────
class DistractionAlertCreate(BaseModel):
    duration_sec: float
    screenshot_path: Optional[str] = None
    email_sent: bool = False
    email_to: Optional[str] = None
    employee_user_id: Optional[str] = None
    face_recognized: Optional[str] = None


class DistractionAlertResponse(BaseModel):
    id: int
    timestamp: datetime
    duration_sec: float
    screenshot_path: Optional[str] = None
    email_sent: bool
    email_to: Optional[str] = None
    employee_user_id: Optional[str] = None
    face_recognized: Optional[str] = None

    class Config:
        from_attributes = True


# ── Stats Schema ──────────────────────────────────────────
class AlertStats(BaseModel):
    total_alerts: int
    emails_sent: int
    avg_duration: float
    unique_employees_detected: int
