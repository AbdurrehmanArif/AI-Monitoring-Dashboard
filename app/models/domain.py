from sqlalchemy import Column, Integer, String, LargeBinary, Boolean, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class UserVideo(Base):
    """Employee ka registration video — face extraction ke liye"""
    __tablename__ = "user_videos"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(String, unique=True, index=True)
    name       = Column(String, nullable=False)
    video_data = Column(LargeBinary, nullable=False)  # Video bytes
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    alerts = relationship("DistractionAlert", back_populates="employee", foreign_keys="DistractionAlert.employee_user_id")


class DistractionAlert(Base):
    """Mobile distraction alert log — PostgreSQL mein persist hota hai"""
    __tablename__ = "distraction_alerts"

    id              = Column(Integer, primary_key=True, index=True)
    timestamp       = Column(DateTime, default=datetime.utcnow, index=True)
    duration_sec    = Column(Float, default=0.0)       # Kitni der mobile use hua
    screenshot_path = Column(String, nullable=True)    # Screenshot file path
    email_sent      = Column(Boolean, default=False)   # Email gaya ya nahi
    email_to        = Column(String, nullable=True)    # Kis ko email gaya
    employee_user_id= Column(String, ForeignKey("user_videos.user_id"), nullable=True)
    face_recognized = Column(String, nullable=True)    # Recognized face ka naam

    # Relationship
    employee = relationship("UserVideo", back_populates="alerts", foreign_keys=[employee_user_id])
