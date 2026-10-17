
import sys
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "india_vru_best.pt"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

model = YOLO(str(MODEL_PATH))

def detect_objects(image_path):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(
        source=str(path),
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )
    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "label": result.names[class_id],
                "confidence": round(float(box.conf.item()), 4),
                "bbox": [round(float(v), 2) for v in box.xyxy[0].tolist()]
            })

    return detections

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detector.py <image_path>")
        sys.exit(1)

    try:
        detections = detect_objects(sys.argv[1])
        if not detections:
            print("No objects detected.")
        else:
            for detection in detections:
                print(detection)
    except Exception as error:
        print(f"Detection failed: {error}")
        sys.exit(1)
