import cv2
import time
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.5  # Only show predictions above this confidence

def draw_boxes(results, frame):
    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"Weapon {conf:.2f}"

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

def detect_weapons(model_path="runs//detect//yolov8s_weapon2//weights//best.pt"):
    # Load YOLO model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam.")
        return

    print("🚀 Press 'q' to quit the webcam window.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 detection
        results = model(frame, verbose=False)

        # Draw detections
        frame = draw_boxes(results, frame)

        # FPS Counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("🔫 Weapon Detection", frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_weapons() 
