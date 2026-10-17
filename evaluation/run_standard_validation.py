
from pathlib import Path
import yaml
from ultralytics import YOLO

root = Path(__file__).resolve().parent.parent
evaluation = root / "evaluation"
dataset = evaluation / "coco_dataset"
model = YOLO(str(root / "yolov8n.pt"))

data = {
    "path": str(dataset.resolve()),
    "train": "images",
    "val": "images",
    "names": {int(key): value for key, value in model.names.items()}
}

yaml_path = evaluation / "coco_dataset_full.yaml"

with yaml_path.open("w", encoding="utf-8") as file:
    yaml.safe_dump(data, file, sort_keys=False)

print("Running Ultralytics validation...")
print("Images:", dataset / "images")
print("Labels:", dataset / "labels")
print("Auto-rickshaw annotations are excluded from this COCO baseline.")

metrics = model.val(
    data=str(yaml_path),
    split="val",
    imgsz=640,
    conf=0.001,
    iou=0.6,
    plots=True,
    verbose=True
)

print("\nStandard Ultralytics validation results")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print("Results directory:", metrics.save_dir)
