import cv2
import os
from deepface import DeepFace
from app.core.database import SessionLocal
from app.models.domain import UserVideo

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KNOWN_FACES_DIR = os.path.join(PROJECT_ROOT, "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


def extract_faces_from_db():
    """
    PostgreSQL DB se employee videos read karo
    aur known_faces/ folder mein face images save karo.
    
    Baad mein streamlit dashboard ya recognize.py in images se
    embeddings banata hai face recognition ke liye.
    """
    db = SessionLocal()
    users = db.query(UserVideo).all()

    if not users:
        print("❌ DB mein koi employee nahi mila. Pehle video upload karo.")
        db.close()
        return

    print(f"✅ {len(users)} employees DB mein mile — faces extract ho rahe hain...\n")

    for user in users:
        print(f"📹 Processing: {user.name} (ID: {user.user_id})")

        # Video bytes → temp file
        temp_video = f"temp_{user.user_id}.mp4"
        with open(temp_video, "wb") as f:
            f.write(user.video_data)

        cap = cv2.VideoCapture(temp_video)
        saved_count = 0
        frame_count = 0
        max_faces   = 5  # Per employee max 5 face images

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 != 0:  # Har 5th frame pe process karo
                continue

            temp_frame = f"temp_frame_{user.user_id}.jpg"
            cv2.imwrite(temp_frame, frame)

            try:
                faces = DeepFace.extract_faces(
                    img_path=temp_frame,
                    enforce_detection=True,
                    detector_backend='opencv'
                )

                if faces:
                    fa = faces[0]['facial_area']
                    x, y, w, h = fa['x'], fa['y'], fa['w'], fa['h']
                    face_crop  = frame[y:y+h, x:x+w]
                    save_path  = os.path.join(KNOWN_FACES_DIR, f"{user.user_id}_{user.name}_{saved_count}.jpg")
                    cv2.imwrite(save_path, face_crop)
                    print(f"  ✅ Face saved: {save_path}")
                    saved_count += 1

            except Exception:
                pass  # Frame mein face nahi mila — skip

            finally:
                if os.path.exists(temp_frame):
                    os.remove(temp_frame)

            if saved_count >= max_faces:
                break

        cap.release()
        if os.path.exists(temp_video):
            os.remove(temp_video)

        if saved_count == 0:
            print(f"  ❌ Koi face nahi mila: {user.name}")
        else:
            print(f"  ✅ {saved_count} faces saved: {user.name}")

    db.close()
    print("\n✅ Face extraction complete! Ab 'streamlit run frontend/dashboard.py' chalao.")


if __name__ == "__main__":
    extract_faces_from_db()
