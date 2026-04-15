from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, _schema_generator):
        return {"type": "string"}


class UserVideoDocument(BaseModel):
    """Employee ka registration video — MongoDB document"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., index=True, unique=True)
    name: str
    video_data: bytes  # Video binary data
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class DistractionAlertDocument(BaseModel):
    """Mobile distraction alert log — MongoDB document"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    duration_sec: float = 0.0  # Kitni der mobile use hua
    screenshot_path: Optional[str] = None  # Screenshot file path
    email_sent: bool = False  # Email gaya ya nahi
    email_to: Optional[str] = None  # Kis ko email gaya
    employee_user_id: Optional[str] = Field(None, index=True)  # FK reference
    face_recognized: Optional[str] = None  # Recognized face ka naam

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}