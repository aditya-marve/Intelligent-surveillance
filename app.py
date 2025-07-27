import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pickle
from scipy.spatial.distance import cosine

# Load known face embeddings
with open("face_embeddings.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Load models
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval()
yolo_model = YOLO("runs/detect/yolov8s_weapon2/weights/best.pt")

CONFIDENCE_THRESHOLD = 0.5
THRESHOLD = 0.6

# Utility functions
def detect_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x, y, x2, y2 = map(int, box)
            face = rgb_frame[y:y2, x:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (160, 160))
            face = (face / 255.0 - 0.5) / 0.5
            face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()

            with torch.no_grad():
                face_embedding = resnet(face_tensor).numpy().flatten()

            best_match = "Unknown"
            best_score = float("inf")

            for name, embedding in known_faces.items():
                embedding = np.array(embedding).flatten()
                score = cosine(face_embedding, embedding)

                if score < THRESHOLD and score < best_score:
                    best_match = name
                    best_score = score

            color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, best_match, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame

def detect_weapons(frame):
    results = yolo_model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"Weapon {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

# Streamlit UI
st.set_page_config(page_title="🔐 Surveillance App", layout="centered")
st.title("🔐 Real-time Surveillance App")
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
        border-radius: 12px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

option = st.radio("Choose Detection Mode:", ["Face Recognition", "Weapon Detection", "Both"], horizontal=True)
start = st.button("▶️ Start Camera")
stop = st.button("⏹ Stop Camera")
frame_placeholder = st.empty()

run = start and not stop

if run:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture frame from camera.")
            break

        if option == "Face Recognition":
            frame = detect_faces(frame)
        elif option == "Weapon Detection":
            frame = detect_weapons(frame)
        elif option == "Both":
            frame = detect_faces(frame)
            frame = detect_weapons(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

        if stop:
            break

    cap.release()
