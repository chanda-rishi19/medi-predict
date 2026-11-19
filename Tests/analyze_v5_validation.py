"""
MediPredict V5 validation confusion matrix
Run from your project root:
    python analyze_v5_validation.py

Expected layout:
    Dataset/val/{0,1,2,3,4}/image files
    models/best_model_v5.pth

Uses the same DenseNet121 ordinal architecture, 320px resize, ImageNet
normalization, and >= 0.5 cumulative-logit decoding as V5.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    balanced_accuracy_score, f1_score, cohen_kappa_score
)

DATA_DIR = "Dataset"
VAL_DIR = os.path.join(DATA_DIR, "val")
CHECKPOINT_PATH = os.path.join("models", "best_model_v5.pth")
IMAGE_SIZE = 320
BATCH_SIZE = 8
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = [0, 1, 2, 3, 4]


def build_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(model.classifier.in_features, 4)
    )
    return model


def main():
    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    if dataset.classes != ["0", "1", "2", "3", "4"]:
        raise ValueError(
            f"Expected folders/classes ['0','1','2','3','4'], found {dataset.classes}"
        )

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = build_model().to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    all_true, all_pred, all_probs, all_paths = [], [], [], []
    with torch.no_grad():
        offset = 0
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            cumulative_probs = torch.sigmoid(logits)
            predictions = (cumulative_probs >= 0.5).sum(dim=1).long()

            batch_n = len(labels)
            paths = dataset.samples[offset:offset + batch_n]
            all_paths.extend([item[0] for item in paths])
            offset += batch_n

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(predictions.cpu().numpy().tolist())
            all_probs.extend(cumulative_probs.cpu().numpy().tolist())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print("\\n" + "=" * 60)
    print("          MEDIPREDICT V5 — VALIDATION ANALYSIS")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'not stored')}")
    print(f"Validation images: {len(dataset)}")
    print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:          {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Quadratic QWK:     {cohen_kappa_score(y_true, y_pred, weights='quadratic'):.4f}")

    print("\\nConfusion matrix (rows = ACTUAL, columns = PREDICTED; grades 0..4):")
    print("       Pred 0  Pred 1  Pred 2  Pred 3  Pred 4")
    for grade, row in enumerate(cm):
        print(f"Actual {grade}  " + "  ".join(f"{v:6d}" for v in row))

    print("\\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=LABELS,
        target_names=[f"Grade {i}" for i in LABELS],
        zero_division=0
    ))

    print("\\nGrade 2 / Grade 3 error focus:")
    print(f"Actual Grade 2 predicted Grade 3: {cm[2, 3]}")
    print(f"Actual Grade 3 predicted Grade 2: {cm[3, 2]}")
    print(f"Actual Grade 2 total: {cm[2].sum()}")
    print(f"Actual Grade 3 total: {cm[3].sum()}")

    output_csv = "v5_validation_predictions.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "actual_grade", "predicted_grade",
            "p_grade_gt_0", "p_grade_gt_1", "p_grade_gt_2", "p_grade_gt_3"
        ])
        for path, actual, pred, probs in zip(all_paths, y_true, y_pred, all_probs):
            writer.writerow([path, int(actual), int(pred), *[round(float(x), 6) for x in probs]])

    print(f"\\nPer-image predictions saved to: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
