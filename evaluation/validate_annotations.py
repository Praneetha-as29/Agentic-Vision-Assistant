from pathlib import Path

classes = Path("evaluation/classes.txt").read_text().splitlines()
label_dir = Path("evaluation/labels")

for path in sorted(label_dir.glob("*.txt")):
    errors = []
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]

    for i, row in enumerate(rows, 1):
        if len(row) != 5:
            errors.append(f"Line {i}: expected 5 values")
            continue

        try:
            cls = int(row[0])
            x, y, w, h = map(float, row[1:])
        except ValueError:
            errors.append(f"Line {i}: invalid number")
            continue

        if not 0 <= cls < len(classes):
            errors.append(f"Line {i}: invalid class ID {cls}")
        if not all(0 <= v <= 1 for v in (x, y, w, h)):
            errors.append(f"Line {i}: coordinates outside [0, 1]")
        if w <= 0 or h <= 0:
            errors.append(f"Line {i}: non-positive width or height")
        if not (0 <= x-w/2 and x+w/2 <= 1 and 0 <= y-h/2 and y+h/2 <= 1):
            errors.append(f"Line {i}: box extends outside image")

    print(f"\n{path.name}: {len(rows)} annotations")
    print("PASS" if not errors else "\n".join(errors))
