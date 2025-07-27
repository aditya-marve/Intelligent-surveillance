import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import pickle

# Load known face embeddings
with open("face_embeddings.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Initialize face detection (MTCNN) and feature extraction (FaceNet)
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

# Start video capture
cap = cv2.VideoCapture(0)

THRESHOLD = 0.6  # Adjusted threshold for better recognition

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)  # Convert coordinates to integers

            if y >= h or x >= w:
                continue

            face = rgb_frame[y:h, x:w]
            if face.size == 0:
                continue

            try:
                # Resize and preprocess face
                face = cv2.resize(face, (160, 160))
                face = (face / 255.0 - 0.5) / 0.5  # Normalize input
                face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()

                # Extract face embedding
                with torch.no_grad():
                    face_embedding = resnet(face_tensor).numpy().flatten()

                best_match = "Unknown"
                best_score = float("inf")

                # Compare with known faces
                for name, embedding in known_faces.items():
                    embedding = np.array(embedding).flatten()  # Ensure stored embeddings are also 1D
                    score = cosine(face_embedding, embedding)

                    if score < THRESHOLD and score < best_score:
                        best_match = name
                        best_score = score

                # Draw bounding box and label
                color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (w, h), color, 2)
                cv2.putText(frame, best_match, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Debugging: Print the recognized name & cosine score
                print(f"Detected: {best_match}, Score: {best_score}")

            except Exception as e:
                print(f"Error processing face: {e}")

    # Show output
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()