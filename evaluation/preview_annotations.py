from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

image_dir = Path("evaluation/images")
label_dir = Path("evaluation/labels")
output_dir = Path("evaluation/preview")
classes = Path("evaluation/classes.txt").read_text().splitlines()

output_dir.mkdir(parents=True, exist_ok=True)

for image_path in sorted(image_dir.iterdir()):
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        print(f"Missing labels: {label_path}")
        continue

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) != 5:
            continue

        cls_id = int(values[0])
        x, y, w, h = map(float, values[1:])

        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        label = classes[cls_id] if 0 <= cls_id < len(classes) else str(cls_id)
        draw.text((x1, max(0, y1 - 18)), label, fill="yellow")

    output_path = output_dir / f"{image_path.stem}_preview.jpg"
    image.save(output_path)
    print(f"Saved: {output_path}")

