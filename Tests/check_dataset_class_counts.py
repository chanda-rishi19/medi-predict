import os
from collections import Counter

DATASET_ROOT = "Dataset"
SPLITS = ["train", "val", "test"]
GRADES = ["0", "1", "2", "3", "4"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

for split in SPLITS:
    split_dir = os.path.join(DATASET_ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"\n[{split}] Folder not found — skipping")
        continue

    counts = Counter()
    for grade in GRADES:
        grade_dir = os.path.join(split_dir, grade)
        if not os.path.isdir(grade_dir):
            print(f"Warning: missing folder {grade_dir}")
            counts[grade] = 0
            continue

        counts[grade] = sum(
            1 for name in os.listdir(grade_dir)
            if os.path.isfile(os.path.join(grade_dir, name))
            and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        )

    total = sum(counts.values())
    print(f"\n{split.upper()} class distribution (total: {total})")
    print("-" * 36)
    for grade in GRADES:
        count = counts[grade]
        percentage = (count / total * 100) if total else 0
        print(f"Grade {grade}: {count:5d} images ({percentage:6.2f}%)")
