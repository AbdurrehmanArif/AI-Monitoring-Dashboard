import cv2
import os
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm

KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")
MATCH_THRESHOLD = 0.55   # SFace cosine distance — kam = strict


def load_embeddings():
    """
    known_faces/ folder se sab employee face embeddings load karo.
    Returns: (face_cascade, known_embeddings list)
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    known_embeddings = []

    if not os.path.exists(KNOWN_FACES_DIR):
        return face_cascade, known_embeddings

    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.endswith(".jpg"):
            continue
        # Filename format: userID_Name_index.jpg
        parts   = filename.replace(".jpg", "").split("_")
        user_id = parts[0]
        name    = "_".join(parts[1:-1])
        fpath   = os.path.join(KNOWN_FACES_DIR, filename)

        try:
            result = DeepFace.represent(
                img_path=fpath,
                model_name="SFace",
                enforce_detection=False,
                detector_backend="opencv"
            )
            known_embeddings.append({
                "embedding": np.array(result[0]["embedding"]),
                "user_id":   user_id,
                "name":      name
            })
        except Exception:
            pass

    return face_cascade, known_embeddings


def match_face(face_img, known_embeddings):
    """
    Ek face image ke against known employees mein se best match dhundo.
    Returns: matched employee dict ya None
    """
    temp_path = "temp_match.jpg"
    cv2.imwrite(temp_path, face_img)
    try:
        result = DeepFace.represent(
            img_path=temp_path,
            model_name="SFace",
            enforce_detection=False,
            detector_backend="skip"
        )
        face_emb  = np.array(result[0]["embedding"])
        best_match = None
        best_dist  = float("inf")

        for known in known_embeddings:
            cos_dist = 1 - np.dot(face_emb, known["embedding"]) / (
                norm(face_emb) * norm(known["embedding"]) + 1e-6
            )
            if cos_dist < best_dist:
                best_dist  = cos_dist
                best_match = known

        if best_dist < MATCH_THRESHOLD:
            return best_match, best_dist

    except Exception:
        pass

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return None, None
