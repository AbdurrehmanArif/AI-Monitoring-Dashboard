from pymongo import MongoClient
from app.core.config import settings

if not settings.MONGO_URL:
    raise ValueError("MONGO_URL is missing in environment variables.")

# MongoDB Client Setup
client = MongoClient(settings.MONGO_URL)
db = client[settings.DB_NAME]

# Collections
user_videos_collection = db["user_videos"]
distraction_alerts_collection = db["distraction_alerts"]

# Create indexes for better performance
user_videos_collection.create_index("user_id", unique=True)
distraction_alerts_collection.create_index("timestamp")
distraction_alerts_collection.create_index("employee_user_id")

def get_db():
    """MongoDB database dependency for FastAPI"""
    return db

def init_db():
    """Initialize MongoDB collections and indexes"""
    print("✅ MongoDB Connected Successfully")
    print(f"📦 Database: {settings.DB_NAME}")
    print(f"📊 Collections: user_videos, distraction_alerts")