"""
MediPredict V5 — final test evaluation: baseline vs frozen calibrated cutoffs.
Run from project root:
    python evaluate_v5_test_calibrated.py

Uses Dataset/test/{0,1,2,3,4} and models/best_model_v5.pth.
Do not tune thresholds using test results.
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

TEST_DIR = os.path.join("Dataset", "test")
CHECKPOINT_PATH = os.path.join("models", "best_model_v5.pth")
IMAGE_SIZE = 320
BATCH_SIZE = 8
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = [0, 1, 2, 3, 4]

BASELINE_THRESHOLDS = [0.5, 0.5, 0.5, 0.5]
CALIBRATED_THRESHOLDS = [0.675, 0.700, 0.700, 0.700]


def build_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(model.classifier.in_features, 4)
    )
    return model


def decode_grades(cumulative_probs, thresholds):
    threshold_tensor = torch.tensor(
        thresholds, dtype=cumulative_probs.dtype, device=cumulative_probs.device
    ).unsqueeze(0)
    return (cumulative_probs >= threshold_tensor).sum(dim=1).long()


def print_metrics(title, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
    print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:          {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Quadratic QWK:     {cohen_kappa_score(y_true, y_pred, weights='quadratic'):.4f}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("       Pred 0  Pred 1  Pred 2  Pred 3  Pred 4")
    for grade, row in enumerate(cm):
        print(f"Actual {grade}  " + "  ".join(f"{v:6d}" for v in row))

    print("\nPer-grade classification report:")
    print(classification_report(
        y_true, y_pred, labels=LABELS,
        target_names=[f"Grade {i}" for i in LABELS],
        zero_division=0
    ))
    return cm


def main():
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    if dataset.classes != ["0", "1", "2", "3", "4"]:
        raise ValueError(
            f"Expected folders/classes ['0','1','2','3','4'], found {dataset.classes}"
        )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = build_model().to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    all_true, all_probs, all_paths = [], [], []
    with torch.no_grad():
        offset = 0
        for images, labels in loader:
            probs = torch.sigmoid(model(images.to(DEVICE)))
            n = len(labels)
            all_paths.extend([item[0] for item in dataset.samples[offset:offset + n]])
            offset += n
            all_true.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.asarray(all_true, dtype=np.int64)
    probs = torch.tensor(np.asarray(all_probs), dtype=torch.float32, device=DEVICE)
    baseline_pred = decode_grades(probs, BASELINE_THRESHOLDS).cpu().numpy()
    calibrated_pred = decode_grades(probs, CALIBRATED_THRESHOLDS).cpu().numpy()

    print("\nMEDIPREDICT V5 — UNTOUCHED TEST SET EVALUATION")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'not stored')}")
    print(f"Test images: {len(dataset)}")
    print(f"Baseline cutoffs: {BASELINE_THRESHOLDS}")
    baseline_cm = print_metrics("BASELINE (all thresholds = 0.5)", y_true, baseline_pred)
    print(f"\nCalibrated cutoffs: {CALIBRATED_THRESHOLDS}")
    calibrated_cm = print_metrics("CALIBRATED (frozen validation cutoffs)", y_true, calibrated_pred)

    # Save per-image predictions to allow transparent error analysis.
    output_csv = "v5_test_baseline_vs_calibrated_predictions.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "actual_grade", "baseline_predicted_grade",
            "calibrated_predicted_grade",
            "p_grade_gt_0", "p_grade_gt_1", "p_grade_gt_2", "p_grade_gt_3"
        ])
        for path, actual, base_pred, cal_pred, p in zip(
            all_paths, y_true, baseline_pred, calibrated_pred, all_probs
        ):
            writer.writerow([path, int(actual), int(base_pred), int(cal_pred),
                             *[round(float(x), 6) for x in p]])

    print(f"\nPer-image results saved to: {output_csv}")
    print("\nMetric deltas (calibrated minus baseline):")
    print(f"Accuracy:          {accuracy_score(y_true, calibrated_pred) - accuracy_score(y_true, baseline_pred):+.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, calibrated_pred) - balanced_accuracy_score(y_true, baseline_pred):+.4f}")
    print(f"Macro F1:          {f1_score(y_true, calibrated_pred, average='macro', zero_division=0) - f1_score(y_true, baseline_pred, average='macro', zero_division=0):+.4f}")
    print(f"Quadratic QWK:     {cohen_kappa_score(y_true, calibrated_pred, weights='quadratic') - cohen_kappa_score(y_true, baseline_pred, weights='quadratic'):+.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
