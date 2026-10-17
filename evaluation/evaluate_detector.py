
from pathlib import Path
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"
CLASSES = (ROOT / "classes.txt").read_text().splitlines()

COCO_MAPPING = {
    0: 0,
    1: 2,
    2: 5,
    3: 3,
    4: 7,
}

MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.5

model = YOLO(MODEL_PATH)

ground_truth = defaultdict(list)
predictions = defaultdict(list)
class_counts = defaultdict(int)

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

for image_path in image_paths:
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(f"Missing annotation: {label_path}")

    image_results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        verbose=False
    )[0]

    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) != 5:
            continue

        class_id = int(values[0])
        x, y, w, h = map(float, values[1:])

        if class_id == 5:
            continue

        if class_id not in COCO_MAPPING:
            continue

        image_width, image_height = image_results.orig_shape[1], image_results.orig_shape[0]

        box = [
            (x - w / 2) * image_width,
            (y - h / 2) * image_height,
            (x + w / 2) * image_width,
            (y + h / 2) * image_height,
        ]

        ground_truth[(image_path.name, class_id)].append(box)
        class_counts[class_id] += 1

    if image_results.boxes is not None:
        for box in image_results.boxes:
            coco_id = int(box.cls.item())
            confidence = float(box.conf.item())

            matching_class = next(
                (custom_id for custom_id, mapped_id in COCO_MAPPING.items()
                 if mapped_id == coco_id),
                None
            )

            if matching_class is None:
                continue

            predictions[(image_path.name, matching_class)].append(
                (box.xyxy[0].cpu().numpy(), confidence)
            )

total_tp = 0
total_fp = 0
total_fn = 0

print("\nPilot evaluation — YOLOv8n")
print(f"Images evaluated: {len(image_paths)}")
print("IoU threshold:", IOU_THRESHOLD)
print("Auto-rickshaws excluded: unsupported by pretrained COCO model")
print()

for class_id, class_name in enumerate(CLASSES):
    if class_id not in COCO_MAPPING:
        continue

    tp = 0
    fp = 0
    fn = 0

    for image_path in image_paths:
        key = (image_path.name, class_id)
        gt_boxes = ground_truth[key]
        pred_boxes = sorted(
            predictions[key],
            key=lambda item: item[1],
            reverse=True
        )

        matched = set()

        for pred_box, confidence in pred_boxes:
            best_iou = 0
            best_index = None

            for index, gt_box in enumerate(gt_boxes):
                if index in matched:
                    continue

                x1 = max(pred_box[0], gt_box[0])
                y1 = max(pred_box[1], gt_box[1])
                x2 = min(pred_box[2], gt_box[2])
                y2 = min(pred_box[3], gt_box[3])

                intersection = max(0, x2 - x1) * max(0, y2 - y1)

                pred_area = max(0, pred_box[2] - pred_box[0]) * max(0, pred_box[3] - pred_box[1])
                gt_area = max(0, gt_box[2] - gt_box[0]) * max(0, gt_box[3] - gt_box[1])

                union = pred_area + gt_area - intersection
                iou = intersection / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index is not None and best_iou >= IOU_THRESHOLD:
                tp += 1
                matched.add(best_index)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched)

    total_tp += tp
    total_fp += fp
    total_fn += fn

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(
        f"{class_name:15} "
        f"GT={class_counts[class_id]:3} "
        f"TP={tp:3} FP={fp:3} FN={fn:3} "
        f"Precision={precision:.3f} Recall={recall:.3f}"
    )

precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0

print("\nOverall micro-averaged results")
print(f"TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
