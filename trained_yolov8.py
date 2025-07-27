from ultralytics import YOLO

if __name__ == "__main__":
    # Load YOLOv8 model (nano or small recommended for 1650 Ti)
    model = YOLO("yolov8s.pt")  # Use "yolov8s.pt" instead of "yolov8n.pt" for better accuracy

    # Train on weapon dataset
    model.train(
        data=r"C:\\Users\\YASH UMATE\\OneDrive\\Desktop\\face_recognition\\weapon-detection\\data.yaml",  # Ensure proper labels in dataset
        epochs=50,         # More training cycles for better learning
        imgsz=640,          # Increased image size for accuracy (512 also works)
        batch=4,            # Adjust to `4` if VRAM allows, else keep `2`
        device="cuda",      # Ensure GPU usage
        workers=0,          # Avoids PyTorch multiprocessing issues on Windows
        amp=True,           # Faster training with mixed precision
        save=True,          # Save model after training
        name="yolov8s_weapon",  # Save dir: runs/train/yolov8s_weapon   
        verbose=True,       # Print logs
        seed=42,            # Ensures reproducibility
        augment=True,       # Use data augmentation
        patience=20         # Stops early if no improvement in 20 epochs 
    )
