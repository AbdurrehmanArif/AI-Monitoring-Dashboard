# 🎯 CV Unified System

**Mobile Distraction Detection + Employee Face Recognition** — merged into one Streamlit dashboard with PostgreSQL backend.

---

## 📁 Project Structure

```
marge/
├── dashboard.py        ← Main Streamlit App (streamlit run dashboard.py)
├── main.py             ← FastAPI Backend    (uvicorn main:app --reload)
├── database.py         ← PostgreSQL connection
├── models.py           ← DB tables (UserVideo + DistractionAlert)
├── schemas.py          ← Pydantic schemas
├── extract_faces.py    ← Face extraction script
├── recognize.py        ← Face recognition module
├── .env.example        ← Environment variables template
├── requirements.txt    ← All dependencies
├── yolo11n.pt          ← YOLO object model (copy karo)
└── yolo11n-pose.pt     ← YOLO pose model   (copy karo)
```

---

## ⚡ Quick Setup

### 1. YOLO Models Copy Karo
```bash
# Computer-Vision-mobile-detection folder se copy karo
copy ..\merged-project\Computer-Vision-mobile-detection\yolo11n.pt .
copy ..\merged-project\Computer-Vision-mobile-detection\yolo11n-pose.pt .
```

### 2. Environment Variables
```bash
copy .env.example .env
# .env file kholo aur apni details daalo:
# DATABASE_URL=postgresql://user:pass@localhost:5432/cv_system
```

### 3. Dependencies Install
```bash
pip install -r requirements.txt
```

### 4. PostgreSQL Database Setup
```sql
-- PostgreSQL mein yeh run karo:
CREATE DATABASE cv_system;
-- Tables automatically create ho jayenge jab FastAPI start hoga
```

### 5. FastAPI Backend Start Karo
```bash
uvicorn main:app --reload
# http://127.0.0.1:8000/docs pe API docs milenge
```

### 6. Streamlit Dashboard Start Karo (alag terminal mein)
```bash
streamlit run dashboard.py
```

---

## 🔄 Employee Registration Flow

1. **Database Tab** → Employee register karo (naam + ID + video)
2. Script chalao:
   ```bash
   python extract_faces.py
   ```
3. **Face Recognition Tab** → Video upload karke employees detect karo

---

## 📊 Dashboard Tabs

| Tab | Feature |
|-----|---------|
| 📹 Live Detection | YOLO-based mobile distraction detection (webcam) |
| 👥 Face Recognition | Video upload + live webcam face recognition |
| 📊 Statistics | Charts — session + DB stats |
| 🖼️ Alert History | Screenshots + email logs |
| 🗄️ Database | Employee register/delete + alerts log |

---

## 🗄️ PostgreSQL Tables

| Table | Purpose |
|-------|---------|
| `user_videos` | Employee registration data + video bytes |
| `distraction_alerts` | Mobile alert logs (timestamp, duration, email status) |
