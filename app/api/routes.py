from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
import os

from app.core.database import get_db
from app.models.domain import UserVideo, DistractionAlert
from app.schemas.payloads import (
    UserVideoResponse, DistractionAlertCreate,
    DistractionAlertResponse, AlertStats
)

router = APIRouter()

# ─────────────────────────────────────────────────────────
# EMPLOYEE VIDEO ENDPOINTS
# ─────────────────────────────────────────────────────────

@router.post("/upload", response_model=UserVideoResponse)
async def upload_video(
    name:    str        = Form(...),
    user_id: str        = Form(...),
    video:   UploadFile = File(...),
    db:      Session    = Depends(get_db)
):
    """Employee ka registration video upload karo — DB mein save hoga"""
    # Check — duplicate user_id
    existing = db.query(UserVideo).filter(UserVideo.user_id == user_id).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Employee ID '{user_id}' already registered! Alag ID use karo."
        )

    video_bytes = await video.read()

    # Duration validate karo (moviepy optional)
    try:
        from moviepy import VideoFileClip
        temp_path = f"temp_check_{user_id}.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        clip     = VideoFileClip(temp_path)
        duration = clip.duration
        clip.close()
        os.remove(temp_path)
        if duration > 15:
            raise HTTPException(
                status_code=400,
                detail=f"Video max 15 sec honi chahiye. Tumhari: {duration:.1f} sec"
            )
    except ImportError:
        pass  # moviepy nahi — skip duration check

    record = UserVideo(user_id=user_id, name=name, video_data=video_bytes)
    db.add(record)
    db.commit()
    db.refresh(record)

    return record


@router.get("/videos", response_model=List[UserVideoResponse])
def list_videos(db: Session = Depends(get_db)):
    """Sab registered employees ki list"""
    return db.query(UserVideo).all()


@router.get("/video/{user_id}")
def get_video(user_id: str, db: Session = Depends(get_db)):
    """Employee ka video bytes return karo"""
    record = db.query(UserVideo).filter(UserVideo.user_id == user_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Employee nahi mila!")
    return Response(content=record.video_data, media_type="video/mp4")


@router.delete("/video/{user_id}")
def delete_video(user_id: str, db: Session = Depends(get_db)):
    """Employee ko DB se delete karo"""
    record = db.query(UserVideo).filter(UserVideo.user_id == user_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Employee nahi mila!")
    db.delete(record)
    db.commit()
    return {"message": f"Employee '{user_id}' delete ho gaya!"}


# ─────────────────────────────────────────────────────────
# DISTRACTION ALERT ENDPOINTS
# ─────────────────────────────────────────────────────────

@router.post("/alerts", response_model=DistractionAlertResponse)
def create_alert(
    alert: DistractionAlertCreate,
    db:    Session = Depends(get_db)
):
    """Naya distraction alert PostgreSQL mein save karo"""
    record = DistractionAlert(**alert.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/alerts", response_model=List[DistractionAlertResponse])
def list_alerts(
    limit: int = 100,
    db:    Session = Depends(get_db)
):
    """Last N alerts ki list"""
    return (
        db.query(DistractionAlert)
        .order_by(DistractionAlert.timestamp.desc())
        .limit(limit)
        .all()
    )


@router.get("/alerts/stats", response_model=AlertStats)
def get_stats(db: Session = Depends(get_db)):
    """Dashboard ke liye aggregate stats"""
    total_alerts  = db.query(DistractionAlert).count()
    emails_sent   = db.query(DistractionAlert).filter(DistractionAlert.email_sent == True).count()
    avg_duration  = db.query(func.avg(DistractionAlert.duration_sec)).scalar() or 0.0
    unique_emp    = db.query(DistractionAlert.employee_user_id).filter(
        DistractionAlert.employee_user_id.isnot(None)
    ).distinct().count()

    return AlertStats(
        total_alerts=total_alerts,
        emails_sent=emails_sent,
        avg_duration=round(float(avg_duration), 1),
        unique_employees_detected=unique_emp
    )


@router.delete("/alerts")
def clear_alerts(db: Session = Depends(get_db)):
    """Sab alerts delete karo (dashboard clear button)"""
    db.query(DistractionAlert).delete()
    db.commit()
    return {"message": "Sab alerts clear ho gaye!"}
