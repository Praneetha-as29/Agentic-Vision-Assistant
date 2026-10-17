
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent.parent
evaluation = root / "evaluation"
source_images = evaluation / "images"
source_labels = evaluation / "labels"
dataset = evaluation / "coco_dataset"
target_images = dataset / "images"
target_labels = dataset / "labels"

class_mapping = {
    0: 0,
    1: 2,
    2: 5,
    3: 3,
    4: 7
}

if not source_images.is_dir() or not source_labels.is_dir():
    raise FileNotFoundError("Original evaluation images or labels are missing.")

images = sorted(
    path for path in source_images.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    raise FileNotFoundError("No evaluation images were found.")

target_images.mkdir(parents=True, exist_ok=True)
target_labels.mkdir(parents=True, exist_ok=True)

for image_path in images:
    label_path = source_labels / f"{image_path.stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(f"Missing labels for {image_path.name}")

    shutil.copy2(image_path, target_images / image_path.name)
    output_lines = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(f"Invalid annotation in {label_path.name}: {line}")

        class_id = int(values[0])

        if class_id in class_mapping:
            values[0] = str(class_mapping[class_id])
            output_lines.append(" ".join(values))

    target = target_labels / label_path.name
    target.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""),
        encoding="utf-8"
    )

    print(f"{image_path.name}: {len(output_lines)} COCO-supported annotations")

print(f"\nDataset prepared at: {dataset}")
print(f"Images copied: {len(images)}")
print("Original images and labels were left unchanged.")
