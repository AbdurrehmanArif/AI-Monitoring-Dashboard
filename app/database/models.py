from .connection import db
from datetime import datetime

# collection create
alerts_collection = db["alerts"]

def save_alert(name, image_path):
    data = {
        "person": name,
        "image": image_path,
        "time": datetime.now()
    }
    alerts_collection.insert_one(data)
    return True


def get_all_alerts():
    alerts = list(alerts_collection.find({}, {"_id": 0}))
    return alerts