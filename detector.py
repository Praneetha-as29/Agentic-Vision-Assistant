
import sys
from pathlib import Path

import gdown
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "india_vru_best.pt"
MODEL_URL_ID = "19Srglx8G6BI6i6wmjnPzIaLsvUqcgDZJ"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45


def load_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
        downloaded_path = gdown.download(
            id=MODEL_URL_ID,
            output=str(MODEL_PATH),
            quiet=False
        )

        if not downloaded_path or not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
            MODEL_PATH.unlink(missing_ok=True)
            raise RuntimeError(
                "Failed to download the custom model. Check Google Drive sharing permissions."
            )

    return YOLO(str(MODEL_PATH))


model = load_model()


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
