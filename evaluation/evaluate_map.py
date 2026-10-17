
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "evaluation" / "images"
LABEL_DIR = ROOT / "evaluation" / "labels"
MODEL_PATH = ROOT / "yolov8n.pt"

CLASS_MAP = {
    0: 0,
    1: 2,
    2: 5,
    3: 3,
    4: 7,
}

CLASS_NAMES = {
    0: "person",
    1: "car",
    2: "bus",
    3: "motorcycle",
    4: "truck",
}

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
CONFIDENCE_THRESHOLD = 0.001

model = YOLO(str(MODEL_PATH))

tp_records = []
confidence_records = []
predicted_classes = []
target_classes = []

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

for image_path in image_paths:
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(f"Missing annotation: {label_path}")

    image = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )[0]

    height, width = image.orig_shape

    targets = []

    for line in label_path.read_text().splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(values[0])

        if class_id not in CLASS_MAP:
            continue

        x, y, w, h = map(float, values[1:])

        targets.append([
            CLASS_MAP[class_id],
            (x - w / 2) * width,
            (y - h / 2) * height,
            (x + w / 2) * width,
            (y + h / 2) * height
        ])

    targets = np.array(targets, dtype=float).reshape(-1, 5)

    boxes = image.boxes

    if boxes is None or len(boxes) == 0:
        predictions = np.empty((0, 6))
    else:
        predictions = np.column_stack((
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy()
        ))

    supported = np.isin(predictions[:, 5], list(CLASS_MAP.values()))
    predictions = predictions[supported]

    target_classes.extend(targets[:, 0].astype(int).tolist())

    matched = np.zeros(
        (len(predictions), len(IOU_THRESHOLDS)),
        dtype=bool
    )

    if len(predictions) and len(targets):
        ious = box_iou(
            __import__("torch").tensor(predictions[:, :4]),
            __import__("torch").tensor(targets[:, 1:5])
        ).numpy()

        same_class = (
            predictions[:, 5, None] == targets[None, :, 0]
        )
        ious *= same_class

        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            pairs = np.argwhere(ious >= threshold)
            pairs = sorted(
                pairs,
                key=lambda pair: ious[pair[0], pair[1]],
                reverse=True
            )

            used_predictions = set()
            used_targets = set()

            for pred_index, target_index in pairs:
                if (
                    pred_index not in used_predictions
                    and target_index not in used_targets
                ):
                    matched[pred_index, threshold_index] = True
                    used_predictions.add(pred_index)
                    used_targets.add(target_index)

    tp_records.extend(matched.tolist())
    confidence_records.extend(predictions[:, 4].tolist())
    predicted_classes.extend(predictions[:, 5].astype(int).tolist())

if not target_classes:
    raise RuntimeError("No supported ground-truth objects were found.")

tp = np.asarray(tp_records, dtype=bool).reshape(-1, len(IOU_THRESHOLDS))
conf = np.asarray(confidence_records)
pred_cls = np.asarray(predicted_classes, dtype=int)
target_cls = np.asarray(target_classes, dtype=int)

results = ap_per_class(
    tp,
    conf,
    pred_cls,
    target_cls,
    plot=False
)

tp_counts, fp_counts, precision, recall, f1, ap, unique_classes = results[:7]

print("\nYOLOv8n pilot evaluation")
print(f"Images: {len(image_paths)}")
print(f"Ground-truth objects evaluated: {len(target_cls)}")
print("Auto-rickshaw: excluded from COCO-class metrics")
print()

for index, class_id in enumerate(unique_classes):
    name = model.names[int(class_id)]
    print(
        f"{name:15} "
        f"Precision={precision[index]:.4f} "
        f"Recall={recall[index]:.4f} "
        f"mAP50={ap[index, 0]:.4f} "
        f"mAP50-95={ap[index].mean():.4f}"
    )

print("\nAggregate metrics")
print(f"mAP@0.5: {ap[:, 0].mean():.4f}")
print(f"mAP@0.5:0.95: {ap.mean():.4f}")
