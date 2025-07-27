import os
import cv2
import torch
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN

# Define dataset path and embeddings file
DATASET_PATH = os.path.abspath("dataset/known_faces")
SAVE_PATH = "face_embeddings.pkl"

# Load FaceNet Model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load MTCNN for face detection
mtcnn = MTCNN(keep_all=False)

# Dictionary to store face embeddings
face_embeddings = {}

# Iterate over each person in dataset
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue  # Skip if not a folder

    embeddings = []
    
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        # Debugging: Check if file exists
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue

        # Read image using OpenCV
        image = cv2.imread(image_path)

        # Debugging: Check if OpenCV read the image
        if image is None:
            print(f"❌ OpenCV could not read image: {image_path}")
            continue

        # Convert BGR (OpenCV) to RGB (Facenet)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face
        face = mtcnn(image)

        if face is None:
            print(f"⚠ No face detected in {image_path}")
            continue

        # Convert face to tensor and get embedding
        face_embedding = model(face.unsqueeze(0))
        embeddings.append(face_embedding.detach().numpy())

    # Save embeddings for the person
    if embeddings:
        face_embeddings[person_name] = np.mean(embeddings, axis=0)  # Averaging multiple images
    
# Save embeddings to a file
with open(SAVE_PATH, "wb") as f:
    pickle.dump(face_embeddings, f)

print("✅ Face embeddings saved successfully!")