import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import cv2
import time
import os
import numpy as np
import smtplib
import json
import tempfile
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CV Unified System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value { font-size: 36px; font-weight: 700; color: #ffffff; }
    .metric-label { font-size: 13px; color: #8892a4; margin-top: 5px; }
    .status-distracted {
        background: linear-gradient(90deg, #ff4444, #cc0000);
        color: white; padding: 15px 25px; border-radius: 10px;
        text-align: center; font-size: 22px; font-weight: 700;
    }
    .status-normal {
        background: linear-gradient(90deg, #00cc44, #009933);
        color: white; padding: 15px 25px; border-radius: 10px;
        text-align: center; font-size: 22px; font-weight: 700;
    }
    .status-table {
        background: linear-gradient(90deg, #ffaa00, #cc8800);
        color: white; padding: 15px 25px; border-radius: 10px;
        text-align: center; font-size: 22px; font-weight: 700;
    }
    .log-entry {
        background: #1a1d2e; border-left: 3px solid #ff4444;
        padding: 10px 15px; margin: 5px 0;
        border-radius: 0 8px 8px 0; font-size: 13px;
    }
    .face-card {
        background: linear-gradient(135deg, #0d2d1a, #0f3320);
        border: 1px solid #00cc44; border-radius: 10px;
        padding: 12px; margin: 6px 0; font-size: 14px;
    }
    footer { display: none !important; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    'alert_log':        [],
    'email_log':        [],
    'total_alerts':     0,
    'total_emails':     0,
    'detection_start':  None,
    'last_seen':        None,
    'alert_triggered':  False,
    'camera_active':    False,
    'known_embeddings': None,
    'face_cascade':     None,
    'face_cam_active':  False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

os.makedirs('screenshots', exist_ok=True)
os.makedirs('known_faces', exist_ok=True)

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

# ============================================================
# LOAD YOLO MODELS
# ============================================================
@st.cache_resource
def load_yolo_models():
    pose  = YOLO('yolo11n-pose.pt')
    phone = YOLO('yolo11n.pt')
    return pose, phone

pose_model, phone_model = load_yolo_models()

MOBILE      = 67
LEFT_WRIST  = 9
RIGHT_WRIST = 10

# ============================================================
# LOAD FACE RECOGNITION
# ============================================================
@st.cache_resource
def load_face_recognizer():
    from app.services.face_recognition import load_embeddings
    return load_embeddings()

# ============================================================
# HELPER FUNCTIONS — MOBILE DETECTION
# ============================================================
def get_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

def point_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1-25), (x1+w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def run_mobile_detection(frame, conf_thresh, wrist_dist):
    pose_res  = pose_model(frame, conf=conf_thresh)[0]
    phone_res = phone_model(frame, conf=conf_thresh-0.1, iou=0.3)[0]
    phones, wrists = [], []
    person_boxes = []  # Person boxes return karenge — face recognition milayega

    for box in phone_res.boxes:
        if int(box.cls) == MOBILE:
            bbox = box.xyxy[0].tolist()
            conf = float(box.conf)
            phones.append({'box': bbox, 'conf': conf})
            draw_box(frame, bbox, f"Phone {conf:.2f}", (255,165,0))

    if pose_res.keypoints is not None:
        for kps in pose_res.keypoints.xy:
            for idx in [LEFT_WRIST, RIGHT_WRIST]:
                kp = kps[idx]
                x, y = float(kp[0]), float(kp[1])
                if x > 0 and y > 0:
                    wrists.append((x, y))
                    cv2.circle(frame, (int(x), int(y)), 8, (0,255,255), -1)

    # Person boxes collect karo — draw NAHI karo (face recognition karega)
    for box in pose_res.boxes:
        person_boxes.append(box.xyxy[0].tolist())

    mobile_in_use = False
    for phone in phones:
        pc = get_center(phone['box'])
        for wrist in wrists:
            if point_distance(pc, wrist) < wrist_dist:
                mobile_in_use = True
                cv2.line(frame, (int(wrist[0]), int(wrist[1])),
                         (int(pc[0]), int(pc[1])), (0,0,255), 2)
                draw_box(frame, phone['box'], "IN USE!", (0,0,255))

    return frame, mobile_in_use, len(phones), len(wrists)//2, person_boxes


# ============================================================
# HELPER FUNCTIONS — FACE RECOGNITION (on frame)
# ============================================================
def face_overlaps_person(fx, fy, fw, fh, person_boxes):
    """
    Face box kisi person box ke andar hai ya nahi check karo.
    Agar hai toh woh person box return karo.
    """
    face_cx = fx + fw // 2
    face_cy = fy + fh // 2
    for pb in person_boxes:
        px1, py1, px2, py2 = map(int, pb)
        if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
            return (px1, py1, px2, py2)
    return None


def run_face_recognition(frame, face_cascade, known_embeddings, fr_cnt, last_fr_results, person_boxes):
    """
    Ek frame pe face recognition run karo.
    - Person box aur face box ko MERGE karta hai — sirf ek box draw hota hai
    - Known: green box + Name | ID
    - Unknown: red box + Unknown
    Har 5th frame pe matching karo (performance ke liye).
    """
    from app.services.face_recognition import match_face as _match_face

    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    if fr_cnt % 5 == 0:
        last_fr_results = []
        for (x, y, w, h) in detected:
            face_crop = frame[y:y+h, x:x+w]
            identity, _ = _match_face(face_crop, known_embeddings)
            # Person box dhundo jo is face ke saath match kare
            matched_person_box = face_overlaps_person(x, y, w, h, person_boxes)
            last_fr_results.append((x, y, w, h, identity, matched_person_box))

    # Person boxes jo kisi face se match nahi hue — Unknown draw karo
    matched_person_boxes = set()
    for (x, y, w, h, identity, mpb) in last_fr_results:
        if mpb:
            matched_person_boxes.add(tuple(mpb))

    html_out = ""
    drawn_boxes = set()

    for (x, y, w, h, identity, mpb) in last_fr_results:
        if identity:
            label    = f"{identity['name']} | {identity['user_id']}"
            color    = (0, 255, 0)
            html_out += f'<div class="face-card">✅ {label}</div>'
        else:
            label = "Unknown"
            color = (0, 0, 255)
            html_out += f'<div class="face-card" style="border-color:#ff4444;">❓ Unknown</div>'

        # Person box use karo agar mila, warna face box
        if mpb:
            bx1, by1, bx2, by2 = mpb
            key = (bx1, by1, bx2, by2)
        else:
            bx1, by1, bx2, by2 = x, y, x+w, y+h
            key = (bx1, by1, bx2, by2)

        if key not in drawn_boxes:
            drawn_boxes.add(key)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (bx1, by1 - th - 10), (bx1 + tw + 6, by1), color, -1)
            cv2.putText(frame, label, (bx1 + 3, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    # Bache hue person boxes (jinka koi face match nahi hua) — Unknown draw karo
    for pb in person_boxes:
        key = tuple(map(int, pb))
        if key not in matched_person_boxes and key not in drawn_boxes:
            drawn_boxes.add(key)
            px1, py1, px2, py2 = key
            label = "Unknown"
            color = (0, 0, 255)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (px1, py1 - th - 10), (px1 + tw + 6, py1), color, -1)
            cv2.putText(frame, label, (px1 + 3, py1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    return frame, last_fr_results, html_out


# ============================================================
# HELPER FUNCTIONS — EMAIL
# ============================================================
def send_alert_email(screenshot_path, elapsed, sender, password, receiver):
    try:
        msg            = MIMEMultipart()
        msg['From']    = sender
        msg['To']      = receiver
        msg['Subject'] = '🚨 Distraction Alert — Mobile Detected!'
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"""
🚨 DISTRACTION ALERT!

📅 Time     : {now_str}
⏱ Duration  : {int(elapsed)} seconds
📱 Status   : Person using mobile detected

Screenshot attached.
-- CV Unified System
        """
        msg.attach(MIMEText(body, 'plain'))
        with open(screenshot_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment',
                           filename=os.path.basename(screenshot_path))
            msg.attach(img)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent!"
    except Exception as e:
        return False, str(e)

# ============================================================
# HELPER — SAVE ALERT TO DB (via FastAPI)
# ============================================================
def save_alert_to_db(duration, screenshot_path, email_sent, email_to=None,
                     employee_user_id=None, face_recognized=None):
    try:
        requests.post(f"{API_BASE}/alerts", json={
            "duration_sec":      duration,
            "screenshot_path":   screenshot_path,
            "email_sent":        email_sent,
            "email_to":          email_to,
            "employee_user_id":  employee_user_id,
            "face_recognized":   face_recognized,
        }, timeout=3)
    except Exception:
        pass

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎯 CV Unified System")
    st.divider()

    st.subheader("📹 Detection Settings")
    cam_source  = st.text_input("Camera Source", value="0", help="0=webcam, ya CCTV URL")
    conf_thresh = st.slider("Confidence",     0.1, 0.9, 0.5, 0.05)
    wrist_dist  = st.slider("Wrist Distance", 50, 300, 150, 10)
    alert_time  = st.slider("Alert Time (sec)", 10, 300, 120, 10)
    tolerance   = st.slider("Tolerance (sec)",   1,  30,   7,  1)

    st.divider()
    st.subheader("👥 Face Recognition")
    face_recog_enabled = st.toggle("Enable Face Recognition", value=True,
                                   help="Live Detection mein face recognition bhi on karo")

    st.divider()
    st.subheader("📧 Email Alerts")
    email_enabled  = st.toggle("Enable Email", value=False)
    email_sender   = st.text_input("Sender Email",   placeholder="your@gmail.com")
    email_password = st.text_input("App Password",   type="password")
    email_receiver = st.text_input("Receiver Email", placeholder="receiver@gmail.com")

    st.divider()
    if st.button("🗑️ Clear Session Logs", type="secondary"):
        st.session_state.alert_log    = []
        st.session_state.email_log    = []
        st.session_state.total_alerts = 0
        st.session_state.total_emails = 0
        st.rerun()

    st.divider()
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

# ============================================================
# MAIN HEADER
# ============================================================
st.title("🎯 CV Unified System")
st.caption("Mobile Distraction Detection + Employee Face Recognition — Powered by YOLO & DeepFace")
st.divider()

# known_faces check — sidebar mein use hota hai
known_faces_exist = (
    os.path.exists('known_faces') and
    len([f for f in os.listdir('known_faces') if f.endswith('.jpg')]) > 0
)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📹 Live Detection",
    "👥 Face Recognition (Video)",
    "📊 Statistics",
    "🖼️ Alert History",
    "🗄️ Database"
])

# ============================================================
# TAB 1 — LIVE DETECTION (Mobile + Face — EK WEBCAM)
# ============================================================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Detection — Mobile + Face Recognition")

        if face_recog_enabled:
            if known_faces_exist:
                st.info("✅ Face Recognition **ON** — same camera se mobile detection + face recognition dono chalenge.")
            else:
                st.warning("⚠️ Face Recognition on hai lekin `known_faces/` empty hai. Pehle **Database** tab mein employees register karo, phir `extract_faces.py` chalao.")

        c1, c2 = st.columns(2)
        start_btn = c1.button("▶ Start Detection", type="primary",  use_container_width=True)
        stop_btn  = c2.button("⏹ Stop",            type="secondary", use_container_width=True)

        frame_placeholder  = st.empty()
        status_placeholder = st.empty()
        timer_placeholder  = st.empty()

    with col2:
        st.subheader("📊 Live Stats")
        m1 = st.empty(); m2 = st.empty(); m3 = st.empty(); m4 = st.empty()
        st.divider()
        st.subheader("👥 Detected Faces")
        face_results_placeholder = st.empty()
        st.divider()
        st.subheader("📋 Live Log")
        log_placeholder = st.empty()

    if start_btn:
        st.session_state.camera_active   = True
        st.session_state.detection_start = None
        st.session_state.last_seen       = None
        st.session_state.alert_triggered = False

    if stop_btn:
        st.session_state.camera_active = False

    if st.session_state.camera_active:
        # ── Face recognizer load karo (agar enabled) ──────
        face_cascade     = None
        known_embeddings = []
        if face_recog_enabled and known_faces_exist:
            face_cascade, known_embeddings = load_face_recognizer()

        # ── EK CAMERA OPEN KARO ───────────────────────────
        src = int(cam_source) if cam_source.isdigit() else cam_source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fps_counter = 0; fps_display = 0; fps_time = time.time()
        fr_cnt = 0; last_fr_results = []
        last_recognized_name = None; last_recognized_uid = None

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera nahi mila!")
                break

            frame = cv2.resize(frame, (640, 480))
            fr_cnt += 1

            # ── Step 1: Mobile Detection ───────────────────
            frame, mobile_in_use, phone_count, person_count, person_boxes = run_mobile_detection(
                frame, conf_thresh, wrist_dist
            )

            # ── Step 2: Face Recognition (same frame) ─────
            face_html = "👤 Face recognition off ya known_faces empty"
            if face_recog_enabled and face_cascade is not None:
                frame, last_fr_results, face_html = run_face_recognition(
                    frame, face_cascade, known_embeddings, fr_cnt, last_fr_results, person_boxes
                )
            else:
                # Face recognition off — sirf person boxes draw karo as Unknown
                for pb in person_boxes:
                    px1, py1, px2, py2 = map(int, pb)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (px1 + 3, py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                # Last recognized person store karo (alert ke liye)
                for (x, y, w, h, identity) in last_fr_results:
                    if identity:
                        last_recognized_name = identity['name']
                        last_recognized_uid  = identity['user_id']
                        break

            now = time.time()

            # ── Timer logic ───────────────────────────────
            if mobile_in_use:
                st.session_state.last_seen = now
                if st.session_state.detection_start is None:
                    st.session_state.detection_start = now
                    st.session_state.alert_triggered = False
                    st.session_state.alert_log.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'event': 'Detection Started', 'duration': 0
                    })
            else:
                if st.session_state.last_seen is not None:
                    if (now - st.session_state.last_seen) > tolerance:
                        st.session_state.detection_start = None
                        st.session_state.last_seen       = None
                        st.session_state.alert_triggered = False

            # ── Alert logic ───────────────────────────────
            if (st.session_state.detection_start is not None
                    and not st.session_state.alert_triggered):
                elapsed   = now - st.session_state.detection_start
                remaining = alert_time - elapsed

                if elapsed >= alert_time:
                    st.session_state.alert_triggered = True
                    st.session_state.total_alerts   += 1

                    filename = f"screenshots/alert_{int(now)}.jpg"
                    cv2.imwrite(filename, frame)

                    st.session_state.alert_log.append({
                        'time':     datetime.now().strftime("%H:%M:%S"),
                        'event':    '🚨 ALERT TRIGGERED',
                        'duration': int(elapsed),
                        'file':     filename
                    })

                    email_ok  = False
                    email_msg = None
                    if email_enabled and email_sender and email_password and email_receiver:
                        email_ok, email_msg = send_alert_email(
                            filename, elapsed,
                            email_sender, email_password, email_receiver
                        )
                        st.session_state.email_log.append({
                            'time':   datetime.now().strftime("%H:%M:%S"),
                            'to':     email_receiver,
                            'status': '✅ Sent' if email_ok else f'❌ {email_msg}',
                            'file':   os.path.basename(filename)
                        })
                        if email_ok:
                            st.session_state.total_emails += 1

                    # PostgreSQL mein save karo — face info bhi
                    save_alert_to_db(
                        duration=int(elapsed),
                        screenshot_path=filename,
                        email_sent=email_ok,
                        email_to=email_receiver if email_enabled else None,
                        employee_user_id=last_recognized_uid,
                        face_recognized=last_recognized_name,
                    )

                    st.session_state.detection_start = None
                    st.session_state.last_seen       = None

                # Progress bar on frame
                mins = int(remaining) // 60; secs = int(remaining) % 60
                cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (0,0,200), -1)
                cv2.putText(frame, f"DISTRACTED! Alert in: {mins:02d}:{secs:02d}",
                            (15,38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
                progress = min(elapsed / alert_time, 1.0)
                bar_w    = int(frame.shape[1] * progress)
                cv2.rectangle(frame, (0, frame.shape[0]-15),
                              (bar_w, frame.shape[0]), (0,0,255), -1)

                status_placeholder.markdown(
                    '<div class="status-distracted">🚨 DISTRACTED!</div>',
                    unsafe_allow_html=True)
                timer_placeholder.progress(progress, text=f"Alert in {mins:02d}:{secs:02d}")

            elif not mobile_in_use and phone_count > 0:
                cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (0,200,200), -1)
                cv2.putText(frame, "Phone on TABLE", (15,38),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                status_placeholder.markdown(
                    '<div class="status-table">📱 Phone on Table — OK</div>',
                    unsafe_allow_html=True)
                timer_placeholder.empty()
            else:
                cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (0,150,0), -1)
                cv2.putText(frame, "Normal", (15,38),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                status_placeholder.markdown(
                    '<div class="status-normal">✅ Normal</div>',
                    unsafe_allow_html=True)
                timer_placeholder.empty()

            # FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_counter; fps_counter = 0; fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps_display}",
                        (frame.shape[1]-120, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            m1.markdown(f'<div class="metric-card"><div class="metric-value">{person_count}</div><div class="metric-label">👤 Persons</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-value">{phone_count}</div><div class="metric-label">📱 Phones</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_alerts}</div><div class="metric-label">🚨 Total Alerts</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_emails}</div><div class="metric-label">📧 Emails Sent</div></div>', unsafe_allow_html=True)

            face_results_placeholder.markdown(
                face_html or "👤 Koi face detect nahi hua",
                unsafe_allow_html=True
            )

            if st.session_state.alert_log:
                log_html = ""
                for entry in st.session_state.alert_log[-5:][::-1]:
                    log_html += f'<div class="log-entry">⏰ {entry["time"]} — {entry["event"]}</div>'
                log_placeholder.markdown(log_html, unsafe_allow_html=True)

        cap.release()

# ============================================================
# TAB 2 — FACE RECOGNITION (Video Upload only)
# ============================================================
with tab2:
    st.subheader("👥 Employee Face Recognition — Video Upload")
    st.info("💡 **Live Webcam** ke liye **Tab 1 (Live Detection)** use karo — wahan ek hi camera se mobile detection + face recognition dono hoti hai.")

    st.markdown("Video upload karo — known employees detect karke annotated video milegi")

    known_faces_exist2 = (
        os.path.exists('known_faces') and
        len([f for f in os.listdir('known_faces') if f.endswith('.jpg')]) > 0
    )

    if not known_faces_exist2:
        st.warning("⚠️ known_faces/ folder empty hai. Pehle **Database** tab mein employee register karo, phir extract_faces.py chalao.")
    else:
        face_cascade2, known_embeddings2 = load_face_recognizer()
        unique_emp = len(set(e['user_id'] for e in known_embeddings2))
        st.success(f"✅ {unique_emp} employees ka face data loaded!")

    uploaded_file = st.file_uploader("📹 Video upload karo", type=["mp4","avi","mov"])

    if uploaded_file and known_faces_exist2:
        st.video(uploaded_file)

        if st.button("🚀 Process Karo", type="primary"):
            face_cascade2, known_embeddings2 = load_face_recognizer()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(uploaded_file.read())
                input_path = tmp_in.name
            output_path = input_path.replace(".mp4", "_output.mp4")

            from app.services.face_recognition import match_face

            cap = cv2.VideoCapture(input_path)
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            all_frames   = []
            frame_count  = 0

            prog = st.progress(0, text="Step 1: Frames read ho rahi hain...")
            while True:
                ret, frame = cap.read()
                if not ret: break
                all_frames.append(frame.copy())
                frame_count += 1
                prog.progress(frame_count / max(total_frames,1) * 0.3,
                              text=f"Step 1: Frame {frame_count}/{total_frames}")
            cap.release()

            prog.progress(0.3, text="Step 2: Faces detect + match ho rahi hain...")
            face_identities = {}

            for i, frame in enumerate(all_frames):
                if i % 10 != 0: continue
                gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected = face_cascade2.detectMultiScale(gray, 1.3, 5)
                if len(detected) > 0:
                    frame_dets = []
                    for (x, y, w, h) in detected:
                        face_crop = frame[y:y+h, x:x+w]
                        identity, dist = match_face(face_crop, known_embeddings2)
                        frame_dets.append((x, y, w, h, identity))
                    face_identities[i] = frame_dets
                prog.progress(0.3 + (i / max(len(all_frames),1)) * 0.4,
                              text=f"Step 2: {i}/{len(all_frames)} frames...")

            prog.progress(0.7, text="Step 3: Video annotate ho rahi hai...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            recognized_people = set()

            for i, frame in enumerate(all_frames):
                if i in face_identities:
                    for (x, y, w, h, identity) in face_identities[i]:
                        if identity:
                            display = f"{identity['name']} | ID: {identity['user_id']}"
                            color   = (0,255,0)
                            recognized_people.add(f"{identity['name']} — ID: {identity['user_id']}")
                        else:
                            display = "Unknown"
                            color   = (0,0,255)
                        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                        if display:
                            (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (x, y-th-10), (x+tw, y), color, -1)
                            cv2.putText(frame, display, (x, y-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                out.write(frame)
                prog.progress(0.7 + (i / max(len(all_frames),1)) * 0.3,
                              text=f"Step 3: {i+1}/{len(all_frames)}")

            out.release()
            prog.progress(1.0, text="✅ Done!")

            st.subheader("👥 Recognized Employees:")
            if recognized_people:
                for person in recognized_people:
                    st.success(f"✅ {person}")
            else:
                st.warning("⚠️ Koi known employee detect nahi hua")

            st.subheader("📹 Annotated Output:")
            with open(output_path, "rb") as f:
                vid_bytes = f.read()
                st.video(vid_bytes)
                st.download_button("⬇️ Download Annotated Video",
                                   vid_bytes, "recognized_output.mp4", "video/mp4")

            os.remove(input_path); os.remove(output_path)

# ============================================================
# TAB 3 — STATISTICS
# ============================================================
with tab3:
    st.subheader("📊 Detection Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_alerts}</div><div class="metric-label">🚨 Session Alerts</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_emails}</div><div class="metric-label">📧 Emails Sent</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{len(os.listdir("screenshots"))}</div><div class="metric-label">📸 Screenshots</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{len(st.session_state.alert_log)}</div><div class="metric-label">📋 Log Entries</div></div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("🗄️ Database (PostgreSQL) Stats")
    try:
        r = requests.get(f"{API_BASE}/alerts/stats", timeout=3)
        if r.status_code == 200:
            s = r.json()
            d1, d2, d3, d4 = st.columns(4)
            d1.markdown(f'<div class="metric-card"><div class="metric-value">{s["total_alerts"]}</div><div class="metric-label">🚨 DB Total Alerts</div></div>', unsafe_allow_html=True)
            d2.markdown(f'<div class="metric-card"><div class="metric-value">{s["emails_sent"]}</div><div class="metric-label">📧 DB Emails</div></div>', unsafe_allow_html=True)
            d3.markdown(f'<div class="metric-card"><div class="metric-value">{s["avg_duration"]}s</div><div class="metric-label">⏱ Avg Duration</div></div>', unsafe_allow_html=True)
            d4.markdown(f'<div class="metric-card"><div class="metric-value">{s["unique_employees_detected"]}</div><div class="metric-label">👤 Employees Detected</div></div>', unsafe_allow_html=True)
    except Exception:
        st.info("ℹ️ FastAPI backend offline — DB stats nahi dikh rahi. `uvicorn app.main:app --reload` chalao.")

    st.divider()

    if st.session_state.alert_log:
        df = pd.DataFrame(st.session_state.alert_log)
        col_left, col_right = st.columns(2)

        with col_left:
            events = df['event'].value_counts().reset_index()
            fig = px.pie(events, values='count', names='event',
                         title='Event Distribution',
                         color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            alerts_only = df[df['event'] == '🚨 ALERT TRIGGERED']
            if not alerts_only.empty:
                fig2 = px.bar(alerts_only, x='time', y='duration',
                              title='Alert Duration (sec)', color='duration',
                              color_continuous_scale='Reds')
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Abhi koi session data nahi — detection start karo!")

# ============================================================
# TAB 4 — ALERT HISTORY (Screenshots)
# ============================================================
with tab4:
    st.subheader("🖼️ Alert Screenshots")

    screenshots = sorted(
        [f for f in os.listdir('screenshots') if f.endswith('.jpg')],
        reverse=True
    )

    if screenshots:
        cols_per_row = 3
        for i in range(0, len(screenshots), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i+j < len(screenshots):
                    fname = screenshots[i+j]
                    fpath = os.path.join('screenshots', fname)
                    with col:
                        img = Image.open(fpath)
                        st.image(img, caption=fname, use_container_width=True)
                        ts = fname.replace('alert_','').replace('.jpg','')
                        try:
                            dt = datetime.fromtimestamp(int(ts))
                            st.caption(f"📅 {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                        except Exception:
                            pass
                        with open(fpath, 'rb') as f:
                            col.download_button("⬇️ Download", f.read(), fname,
                                                "image/jpeg", use_container_width=True)
    else:
        st.info("Koi screenshot nahi — Live Detection tab mein detection shuru karo!")

    st.divider()
    st.subheader("📧 Email Logs")
    if st.session_state.email_log:
        df_email = pd.DataFrame(st.session_state.email_log)
        st.dataframe(df_email, use_container_width=True,
                     column_config={'time':'Time','to':'Sent To',
                                    'status':'Status','file':'Screenshot'})
    else:
        st.info("Koi email log nahi abhi!")

# ============================================================
# TAB 5 — DATABASE
# ============================================================
with tab5:
    st.subheader("🗄️ PostgreSQL Database")

    db_tab1, db_tab2 = st.tabs(["👤 Employees", "🚨 Alerts Log"])

    with db_tab1:
        st.markdown("#### ➕ Naya Employee Register")

        # ── Session state for live capture ──────────────────
        if 'reg_captured_video' not in st.session_state:
            st.session_state.reg_captured_video = None
        if 'reg_capturing' not in st.session_state:
            st.session_state.reg_capturing = False

        reg_col1, reg_col2 = st.columns([1, 1])

        with reg_col1:
            reg_name    = st.text_input("👤 Naam", key="reg_name_input")
            reg_user_id = st.text_input("🪪 Employee ID", key="reg_uid_input")

        with reg_col2:
            st.markdown("**📹 Video — Upload ya Live Capture**")
            video_method = st.radio("Method:", ["📁 File Upload", "📷 Live Webcam Capture"],
                                    horizontal=True, key="video_method")

        if video_method == "📁 File Upload":
            reg_video = st.file_uploader("📹 Registration Video (max 15 sec)",
                                         type=["mp4","avi","mov"], key="reg_video_upload")
            video_bytes_to_send = reg_video.read() if reg_video else None
            video_name_to_send  = reg_video.name   if reg_video else "video.mp4"

        else:  # Live Webcam Capture
            st.info("📷 **Start** dabao — 10 seconds ki video capture hogi automatically")
            cap_col1, cap_col2 = st.columns(2)
            capture_btn = cap_col1.button("🔴 Start Capture", type="primary",
                                          key="start_capture", use_container_width=True)
            clear_btn   = cap_col2.button("🗑️ Clear",         type="secondary",
                                          key="clear_capture", use_container_width=True)

            if clear_btn:
                st.session_state.reg_captured_video = None
                st.rerun()

            cap_preview = st.empty()
            cap_status  = st.empty()

            if capture_btn:
                st.session_state.reg_captured_video = None
                cap_source_val = int(cam_source) if cam_source.isdigit() else cam_source
                cap_reg = cv2.VideoCapture(cap_source_val)
                cap_reg.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap_reg.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap_reg.set(cv2.CAP_PROP_FPS, 20)

                RECORD_SEC = 10
                fps_rec    = 20
                fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
                tmp_out    = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_path   = tmp_out.name
                tmp_out.close()
                writer = cv2.VideoWriter(tmp_path, fourcc, fps_rec, (640, 480))

                start_t    = time.time()
                frame_c    = 0

                while True:
                    ret, frm = cap_reg.read()
                    if not ret:
                        cap_status.error("❌ Camera nahi mila!")
                        break
                    frm    = cv2.resize(frm, (640, 480))
                    elapsed_r = time.time() - start_t
                    remaining_r = RECORD_SEC - elapsed_r
                    if remaining_r <= 0:
                        break

                    # Countdown overlay
                    cv2.rectangle(frm, (0, 0), (640, 50), (200, 0, 0), -1)
                    cv2.putText(frm, f"Recording... {remaining_r:.1f}s remaining",
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    bar_w = int(640 * (elapsed_r / RECORD_SEC))
                    cv2.rectangle(frm, (0, 465), (bar_w, 480), (0, 0, 255), -1)

                    writer.write(frm)
                    frame_c += 1
                    cap_preview.image(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB),
                                      channels="RGB", use_container_width=True)
                    cap_status.markdown(f"⏺️ **Recording** — `{remaining_r:.1f}` sec baki")

                writer.release()
                cap_reg.release()
                cap_preview.empty()

                with open(tmp_path, "rb") as f:
                    st.session_state.reg_captured_video = f.read()
                os.remove(tmp_path)
                cap_status.success(f"✅ 10 sec video capture ho gayi! Neeche Register karo.")

            if st.session_state.reg_captured_video:
                st.video(st.session_state.reg_captured_video)
                st.success("✅ Video ready hai — Register Karo button dabao")

            video_bytes_to_send = st.session_state.reg_captured_video
            video_name_to_send  = "live_capture.mp4"
            reg_video = video_bytes_to_send  # just for validation check below

        st.divider()
        if st.button("📤 Register Karo", type="primary", use_container_width=True,
                     key="final_register_btn"):
            if not reg_name or not reg_user_id or not video_bytes_to_send:
                st.error("❌ Naam, Employee ID aur Video — teeno zaroori hain!")
            else:
                try:
                    res = requests.post(
                        f"{API_BASE}/upload",
                        data={"name": reg_name, "user_id": reg_user_id},
                        files={"video": (video_name_to_send, video_bytes_to_send, "video/mp4")},
                        timeout=120
                    )
                    if res.status_code == 200:
                        st.success(f"✅ {reg_name} register ho gaya! Ab `python -m app.services.extract_faces` chalao.")
                        st.session_state.reg_captured_video = None
                    else:
                        st.error(f"❌ {res.json().get('detail','Error!')}")
                except Exception as e:
                    st.error(f"❌ API se connect nahi hua: {e}\n\nPehle `uvicorn app.main:app --reload` chalao.")

        st.divider()
        st.markdown("#### 📋 Registered Employees")

        if st.button("🔄 Refresh List"):
            st.rerun()

        try:
            res = requests.get(f"{API_BASE}/videos", timeout=30)
            if res.status_code == 200:
                employees = res.json()
                if employees:
                    df_emp = pd.DataFrame(employees)[['user_id','name','created_at']]
                    df_emp.columns = ['Employee ID', 'Naam', 'Register Time']
                    st.dataframe(df_emp, use_container_width=True)

                    st.markdown("#### 🗑️ Employee Delete Karo")
                    del_id = st.text_input("Employee ID daalo jise delete karna hai")
                    if st.button("Delete", type="secondary"):
                        if del_id:
                            r2 = requests.delete(f"{API_BASE}/video/{del_id}", timeout=30)
                            if r2.status_code == 200:
                                st.success("✅ Deleted!")
                                st.rerun()
                            else:
                                st.error(r2.json().get('detail','Error!'))
                else:
                    st.info("Koi employee registered nahi. Upar form se register karo!")
            else:
                st.error("API error!")
        except Exception:
            st.info("ℹ️ FastAPI offline — `uvicorn app.main:app --reload` chalao pehle.")

        st.divider()
        st.markdown("#### 🔄 Face Extraction")
        st.info("Employees register karne ke baad, **extract_faces.py** script chalao taake face recognition kaam kare:\n```\npython extract_faces.py\n```")

    with db_tab2:
        st.markdown("#### 🚨 Distraction Alerts (PostgreSQL)")
        col_r, col_c = st.columns([3,1])

        with col_c:
            if st.button("🔄 Refresh", key="refresh_alerts"):
                st.rerun()
            if st.button("🗑️ Clear All Alerts", type="secondary"):
                try:
                    requests.delete(f"{API_BASE}/alerts", timeout=30)
                    st.success("Cleared!")
                    st.rerun()
                except Exception:
                    st.error("API offline!")

        try:
            res = requests.get(f"{API_BASE}/alerts?limit=200", timeout=30)
            if res.status_code == 200:
                alerts = res.json()
                if alerts:
                    df_alerts = pd.DataFrame(alerts)[[
                        'id','timestamp','duration_sec','email_sent',
                        'email_to','face_recognized','screenshot_path'
                    ]]
                    df_alerts.columns = ['ID','Timestamp','Duration(s)','Email Sent',
                                          'Email To','Face Recognized','Screenshot']
                    with col_r:
                        st.dataframe(df_alerts, use_container_width=True)
                else:
                    st.info("Koi alerts nahi abhi!")
            else:
                st.error("API error!")
        except Exception:
            st.info("ℹ️ FastAPI offline — `uvicorn app.main:app --reload` chalao.")