# 🛡️ VisionGuard AI
### Intelligent Workplace Surveillance & Employee Recognition System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO_v11-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)

*Real-time mobile distraction detection + employee face recognition — one camera, one dashboard.*

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📱 **Mobile Detection** | YOLOv11-based detection — identifies phone usage via wrist proximity |
| 👤 **Face Recognition** | DeepFace SFace model — identifies registered employees in real-time |
| 🔗 **Unified Camera** | Single webcam handles both detection tasks simultaneously |
| 🚨 **Smart Alerts** | Configurable timer — triggers alert only after sustained distraction |
| 📧 **Email Notifications** | Auto-sends alert email with screenshot when threshold exceeded |
| 🗄️ **PostgreSQL Backend** | Persistent storage for employees, alerts, and email logs |
| 📊 **Live Analytics** | Real-time charts and metrics dashboard |
| 🖼️ **Evidence Capture** | Auto-screenshots saved with timestamp on every alert |
| 📹 **Video Analysis** | Analyze recorded videos for employee identification |
| 📷 **Live Registration** | Register employees via 10-second live webcam capture |

---

## 📁 Project Structure

```
visionguard/
│
├── 📄 dashboard.py          ← Main Streamlit App  (streamlit run dashboard.py)
├── 📄 main.py               ← FastAPI Backend      (uvicorn main:app --reload)
├── 📄 database.py           ← PostgreSQL connection setup
├── 📄 models.py             ← SQLAlchemy table models
├── 📄 schemas.py            ← Pydantic validation schemas
├── 📄 recognize.py          ← Face embedding + matching logic
├── 📄 extract_faces.py      ← Face extraction from employee videos
│
├── 🤖 yolo11n.pt            ← YOLO object detection model (phone detection)
├── 🤖 yolo11n-pose.pt       ← YOLO pose estimation model (wrist detection)
│
├── 📁 screenshots/          ← Auto-saved alert screenshots
├── 📁 known_faces/          ← Extracted employee face images
│
├── 📄 requirements.txt      ← All Python dependencies
├── 📄 .env                  ← Environment variables (create from .env.example)
└── 📄 README.md             ← This file
```

---

## ⚡ Quick Setup

### Prerequisites

- Python 3.9 or higher
- PostgreSQL installed and running
- Webcam (or IP camera with RTSP URL)

---

### Step 1 — Clone & Install Dependencies

```bash
# Install all Python packages
pip install -r requirements.txt
```

---

### Step 2 — Configure Environment

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/cv_system
API_BASE=http://127.0.0.1:8000
```

Replace `YOUR_PASSWORD` with your PostgreSQL password.

---

### Step 3 — Setup PostgreSQL Database

Open PostgreSQL and run:

```sql
CREATE DATABASE cv_system;
```

> Tables (`user_videos`, `distraction_alerts`) are created **automatically** when FastAPI starts.

---

### Step 4 — Start FastAPI Backend

Open **Terminal 1** and run:

```bash
uvicorn main:app --reload
```

✅ Backend starts at `http://127.0.0.1:8000`  
📚 API docs available at `http://127.0.0.1:8000/docs`

---

### Step 5 — Start Dashboard

Open **Terminal 2** and run:

```bash
streamlit run dashboard.py
```

✅ Dashboard opens at `http://localhost:8501`

---

## 👤 Employee Registration Flow

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  EMPLOYEES tab  │────▶│  Register via video  │────▶│ python extract_     │
│  in Dashboard   │     │  (upload or webcam)  │     │ faces.py            │
└─────────────────┘     └──────────────────────┘     └──────────┬──────────┘
                                                                  │
                                                                  ▼
                                                     ┌─────────────────────┐
                                                     │  known_faces/ folder│
                                                     │  populated ✅        │
                                                     └──────────┬──────────┘
                                                                  │
                                                                  ▼
                                                     ┌─────────────────────┐
                                                     │  Face Recognition   │
                                                     │  active in dashboard│
                                                     └─────────────────────┘
```

1. Go to **🗄 EMPLOYEES** tab → **Register Employee**
2. Enter Name + Employee ID
3. Upload a 10-15 second face video, OR use **Live Webcam** capture button
4. Click **Register Employee**
5. Run in terminal:
   ```bash
   python extract_faces.py
   ```
6. Face recognition is now active in **Live Monitor**

---

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🎥 **Live Monitor** | Real-time webcam feed with mobile detection + face recognition |
| 🎬 **Video Analysis** | Upload recorded videos — identifies employees frame by frame |
| 📊 **Analytics** | Charts showing alert frequency, duration, and distribution |
| 🖼 **Evidence** | Saved alert screenshots, email logs, and database alert history |
| 🗄 **Employees** | Register new employees, view directory, delete employees |

---

## ⚙️ Sidebar Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Camera Source | `0` | `0` = default webcam, or RTSP/HTTP URL |
| Confidence Threshold | `0.50` | Detection confidence (higher = stricter) |
| Wrist Proximity | `150px` | Max distance between wrist and phone to count as "in use" |
| Alert Timer | `120s` | How long distraction must persist before alert fires |
| Grace Period | `7s` | Tolerance before resetting timer when phone disappears |
| Face ID | `ON` | Enable/disable employee recognition |
| Email Alerts | `OFF` | Enable email notifications on alert |

---

## 🗄️ Database Tables

### `user_videos`
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `user_id` | String | Unique employee ID |
| `name` | String | Employee full name |
| `video_data` | Binary | Registration video bytes |
| `created_at` | DateTime | Registration timestamp |

### `distraction_alerts`
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `timestamp` | DateTime | Alert time |
| `duration_sec` | Float | Duration of distraction |
| `screenshot_path` | String | Path to captured screenshot |
| `email_sent` | Boolean | Whether email was sent |
| `email_to` | String | Recipient email address |
| `employee_user_id` | String | FK to `user_videos.user_id` |
| `face_recognized` | String | Recognized employee name |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Register new employee with video |
| `GET` | `/videos` | List all registered employees |
| `GET` | `/video/{user_id}` | Stream employee video |
| `DELETE` | `/video/{user_id}` | Delete employee |
| `POST` | `/alerts` | Create new distraction alert |
| `GET` | `/alerts` | List recent alerts |
| `GET` | `/alerts/stats` | Aggregate statistics |
| `DELETE` | `/alerts` | Clear all alerts |

Full interactive docs: `http://127.0.0.1:8000/docs`

---

## 🛠️ How Detection Works

```
Webcam Frame
     │
     ├──▶ YOLOv11-pose ──▶ Detects person bounding boxes
     │                      Detects wrist keypoints (left + right)
     │
     ├──▶ YOLOv11      ──▶ Detects mobile phone + confidence score
     │
     ├──▶ Proximity Check
     │         If  distance(wrist, phone_center) < threshold
     │         Then  mobile_in_use = True
     │
     ├──▶ DeepFace SFace ──▶ Detects faces in frame (every 5th frame)
     │                        Matches against known employee embeddings
     │                        Assigns identity to person bounding box
     │
     └──▶ Single Box Drawn ──▶ Green = Known employee (Name + ID)
                                Red   = Unknown person
```

---

## 🚨 Alert Flow

```
Mobile detected → Timer starts
                      │
                      ▼
              [alert_time] seconds pass?
                      │
              YES ────┤
                      │
                      ├──▶ Screenshot saved  → screenshots/alert_TIMESTAMP.jpg
                      ├──▶ Alert logged      → PostgreSQL distraction_alerts
                      ├──▶ Email sent        → (if enabled)
                      └──▶ Dashboard updated → Stats, Evidence tab
```

---

## ❗ Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Camera nahi mila` | Webcam not connected or busy | Check device, try source `1` or `2` |
| `API se connect nahi hua` | FastAPI not running | Run `uvicorn main:app --reload` in terminal |
| `known_faces/ empty` | Faces not extracted | Run `python extract_faces.py` |
| `psycopg2 error` | Wrong DB credentials | Check `.env` DATABASE_URL |
| `Employee ID already exists` | Duplicate registration | Use a different Employee ID |
| `YOLO model not found` | Missing `.pt` files | Ensure `yolo11n.pt` and `yolo11n-pose.pt` are in project root |
| Unknown shown despite employee registered | Faces not extracted | Re-run `python extract_faces.py` |

---

## 📦 Dependencies

```
# Computer Vision
ultralytics          # YOLO v11
opencv-python        # Camera + image processing
numpy                # Array operations
Pillow               # Image handling

# Face Recognition
deepface             # SFace model
tf-keras             # TensorFlow backend

# Backend
fastapi              # REST API
uvicorn[standard]    # ASGI server
python-multipart     # File upload support
moviepy              # Video duration validation

# Database
sqlalchemy           # ORM
psycopg2-binary      # PostgreSQL driver
python-dotenv        # Environment variables

# Dashboard
streamlit            # Web UI
plotly               # Interactive charts
pandas               # Data processing
requests             # HTTP client
```

---

## 👨‍💻 Author

Built with ❤️ using **YOLO v11**, **DeepFace**, **FastAPI**, **Streamlit**, and **PostgreSQL**.

---

<div align="center">
<sub>VisionGuard AI — Smart Workplace Monitoring System</sub>
</div>
