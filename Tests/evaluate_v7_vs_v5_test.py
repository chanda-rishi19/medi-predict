"""
MediPredict: compare V5 baseline checkpoint with V7 Grade-1-focused checkpoint
on Dataset/test. No threshold tuning; both use the fixed 0.5 ordinal cutoffs.

Run from project root:
    python evaluate_v7_vs_v5_test.py
"""
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report
)

TEST_DIR = os.path.join("Dataset", "test")
V5_PATH = os.path.join("models", "best_model_v5.pth")
V7_PATH = os.path.join("models", "best_model_v7_grade1_focus.pth")
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


def decode(logits):
    return (torch.sigmoid(logits) >= 0.5).sum(dim=1).long()


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = build_model().to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def evaluate_model(model, loader):
    ys, preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE, non_blocking=True))
            predictions = decode(logits).cpu().numpy().tolist()
            ys.extend(labels.numpy().tolist())
            preds.extend(predictions)
    return np.asarray(ys), np.asarray(preds)


def report(title, y, pred):
    cm = confusion_matrix(y, pred, labels=LABELS)
    grade_recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    print("\\n" + "=" * 68)
    print(title)
    print("=" * 68)
    print(f"Accuracy:          {accuracy_score(y, pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, pred):.4f}")
    print(f"Macro F1:          {f1_score(y, pred, average='macro', zero_division=0):.4f}")
    print(f"Quadratic QWK:     {cohen_kappa_score(y, pred, weights='quadratic'):.4f}")
    print(f"Grade 1 recall:    {grade_recalls[1]:.4f}")
    print("\\nConfusion matrix (rows=actual, columns=predicted):")
    print("       Pred 0  Pred 1  Pred 2  Pred 3  Pred 4")
    for g, row in enumerate(cm):
        print(f"Actual {g}  " + "  ".join(f"{v:6d}" for v in row))
    print("\\nClassification report:")
    print(classification_report(
        y, pred, labels=LABELS,
        target_names=[f"Grade {g}" for g in LABELS], zero_division=0
    ))
    return cm, grade_recalls


def main():
    for path in [TEST_DIR, V5_PATH, V7_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=transform)
    if ds.classes != ["0", "1", "2", "3", "4"]:
        raise ValueError(f"Expected classes ['0','1','2','3','4'], got {ds.classes}")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    v5, v5_ckpt = load_checkpoint(V5_PATH)
    v7, v7_ckpt = load_checkpoint(V7_PATH)
    y5, pred5 = evaluate_model(v5, loader)
    y7, pred7 = evaluate_model(v7, loader)
    if not np.array_equal(y5, y7):
        raise RuntimeError("Ground truth order mismatch between model runs.")

    print("MEDIPREDICT TEST COMPARISON — V5 vs V7")
    print(f"Device: {DEVICE} | Test images: {len(ds)}")
    print(f"V5 checkpoint epoch: {v5_ckpt.get('epoch', 'not stored')}")
    print(f"V7 checkpoint epoch: {v7_ckpt.get('epoch', 'not stored')}")
    print("Both models use thresholds [0.5, 0.5, 0.5, 0.5].")

    cm5, recall5 = report("V5 BASELINE", y5, pred5)
    cm7, recall7 = report("V7 GRADE-1-FOCUSED", y7, pred7)

    def values(y, p):
        return {
            "accuracy": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0),
            "qwk": cohen_kappa_score(y, p, weights="quadratic"),
            "grade1_recall": recall1(y, p)
        }

    def recall1(y, p):
        cm = confusion_matrix(y, p, labels=LABELS)
        return cm[1, 1] / max(cm[1].sum(), 1)

    m5, m7 = values(y5, pred5), values(y7, pred7)
    print("\\n" + "=" * 68)
    print("METRIC DELTAS (V7 minus V5)")
    print("=" * 68)
    for key in ["accuracy", "balanced_accuracy", "macro_f1", "qwk", "grade1_recall"]:
        print(f"{key:18s}: {m7[key] - m5[key]:+.4f}")

    out = "v5_vs_v7_test_predictions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "actual_grade", "v5_predicted_grade", "v7_predicted_grade"])
        for (path, _), actual, p5, p7 in zip(ds.samples, y5, pred5, pred7):
            writer.writerow([path, int(actual), int(p5), int(p7)])
    print(f"\\nSaved per-image comparison to: {out}")
    print("Note: this test comparison is descriptive; don't tune thresholds from these results.")


if __name__ == "__main__":
    main()
