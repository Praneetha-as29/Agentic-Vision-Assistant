from pathlib import Path

source_dir = Path("evaluation/labels")
target_dir = Path("evaluation/coco_labels")
target_dir.mkdir(parents=True, exist_ok=True)

class_mapping = {
    0: 0,
    1: 2,
    2: 5,
    3: 3,
    4: 7
}

for source in source_dir.glob("*.txt"):
    output_lines = []

    for line in source.read_text().splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(values[0])

        if class_id not in class_mapping:
            continue

        values[0] = str(class_mapping[class_id])
        output_lines.append(" ".join(values))

    target = target_dir / source.name
    target.write_text("\n".join(output_lines) + ("\n" if output_lines else ""))
    print(f"Prepared {target}")
