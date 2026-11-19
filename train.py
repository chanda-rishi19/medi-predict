
import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    recall_score
)

# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = "Dataset"
MODEL_DIR = "models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model_v5.pth")

IMAGE_SIZE = 320
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 7
NUM_WORKERS = 0

SEED = 42

# ==========================================
# REPRODUCIBILITY
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 55)
print("MEDIPREDICT - KNEE OA TRAINING")
print("=" * 55)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ==========================================
# DATA TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=7),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# LOAD DATASETS
# ==========================================

train_path = os.path.join(DATA_DIR, "train")
val_path = os.path.join(DATA_DIR, "val")
test_path = os.path.join(DATA_DIR, "test")

for path in [train_path, val_path, test_path]:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Dataset folder not found: {path}"
        )

train_dataset = datasets.ImageFolder(
    train_path,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_path,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    test_path,
    transform=eval_transform
)

expected_classes = ["0", "1", "2", "3", "4"]

if train_dataset.classes != expected_classes:
    raise ValueError(
        f"Unexpected class mapping: {train_dataset.classes}"
    )

if val_dataset.classes != expected_classes:
    raise ValueError("Validation class mapping does not match training.")

if test_dataset.classes != expected_classes:
    raise ValueError("Test class mapping does not match training.")

print("\nDataset information")
print("-" * 40)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))
print("Class mapping:", train_dataset.class_to_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

# ==========================================
# MODEL
# MUST MATCH gradcam_v5_fixed.py
# ==========================================

def build_model():
    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(model.classifier.in_features, 4)
    )

    return model.to(DEVICE)

model = build_model()

# ==========================================
# ORDINAL TARGETS
# Grade 0 -> [0, 0, 0, 0]
# Grade 1 -> [1, 0, 0, 0]
# Grade 2 -> [1, 1, 0, 0]
# Grade 3 -> [1, 1, 1, 0]
# Grade 4 -> [1, 1, 1, 1]
# ==========================================

def make_ordinal_targets(labels):
    thresholds = torch.arange(
        4,
        device=labels.device
    )

    return (
        labels.unsqueeze(1) > thresholds.unsqueeze(0)
    ).float()

# ==========================================
# ORDINAL LOSS
# ==========================================

class_counts = np.bincount(
    [label for _, label in train_dataset.samples],
    minlength=5
)

threshold_positive = np.array([
    class_counts[1:].sum(),
    class_counts[2:].sum(),
    class_counts[3:].sum(),
    class_counts[4:].sum()
], dtype=np.float32)

threshold_negative = len(train_dataset) - threshold_positive

# Moderate weighting for less frequent higher-grade thresholds
pos_weights = np.sqrt(
    threshold_negative / np.maximum(threshold_positive, 1)
)

pos_weights = np.clip(pos_weights, 1.0, 3.0)

pos_weights = torch.tensor(
    pos_weights,
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weights
)

print("\nOrdinal threshold weights:", pos_weights.cpu().numpy())

# ==========================================
# OPTIMIZER AND SCHEDULER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# ==========================================
# PREDICTION AND METRICS
# ==========================================

def logits_to_grades(logits):
    probabilities = torch.sigmoid(logits)

    # Number of ordinal thresholds predicted as positive
    predictions = (probabilities >= 0.5).sum(dim=1)

    return predictions.cpu().numpy()


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "qwk": cohen_kappa_score(
            y_true, y_pred,
            weights="quadratic"
        ),
        "recall": recall_score(
            y_true, y_pred,
            labels=[0, 1, 2, 3, 4],
            average=None,
            zero_division=0
        )
    }


# ==========================================
# TRAIN ONE EPOCH
# ==========================================

def train_one_epoch():
    model.train()

    total_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        targets = make_ordinal_targets(labels)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = criterion(logits, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(train_dataset)


# ==========================================
# VALIDATION
# ==========================================

@torch.no_grad()
def evaluate(loader):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        targets = make_ordinal_targets(labels)

        logits = model(images)

        loss = criterion(logits, targets)

        predictions = logits_to_grades(logits)

        total_loss += loss.item() * images.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions)

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions)
    )

    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


# ==========================================
# TRAINING LOOP
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)

best_qwk = -float("inf")
best_epoch = 0
epochs_without_improvement = 0

print("\nStarting DenseNet121 ordinal training")
print("=" * 55)

for epoch in range(EPOCHS):

    train_loss = train_one_epoch()

    val_metrics = evaluate(val_loader)

    scheduler.step(val_metrics["qwk"])

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss       : {train_loss:.4f}")
    print(f"Validation Loss     : {val_metrics['loss']:.4f}")
    print(f"Validation Accuracy : {val_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy   : {val_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1            : {val_metrics['macro_f1']:.4f}")
    print(f"Validation QWK      : {val_metrics['qwk']:.4f}")
    print(f"Learning Rate       : {current_lr:.7f}")

    print("Grade Recall:")
    for grade, recall in enumerate(val_metrics["recall"]):
        print(f"  Grade {grade}: {recall:.4f}")

    if val_metrics["qwk"] > best_qwk:

        best_qwk = val_metrics["qwk"]
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": best_epoch,
            "best_val_qwk": best_qwk,
            "class_to_idx": train_dataset.class_to_idx,
            "image_size": IMAGE_SIZE
        }

        torch.save(checkpoint, CHECKPOINT_PATH)

        print("\nBest model saved:", CHECKPOINT_PATH)

    else:
        epochs_without_improvement += 1
        print(
            "No improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

# ==========================================
# LOAD BEST CHECKPOINT
# ==========================================

print("\nLoading best validation checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

print("Best epoch:", best_epoch)
print("Best validation QWK:", best_qwk)

# ==========================================
# FINAL TEST EVALUATION
# ==========================================

print("\nEvaluating on held-out test dataset...")
print("=" * 55)

test_metrics = evaluate(test_loader)

print("\nFINAL TEST RESULTS")
print("-" * 40)
print(f"Accuracy          : {test_metrics['accuracy']:.4f}")
print(f"Balanced Accuracy : {test_metrics['balanced_accuracy']:.4f}")
print(f"Macro F1          : {test_metrics['macro_f1']:.4f}")
print(f"Quadratic Kappa   : {test_metrics['qwk']:.4f}")

print("\nPer-grade recall:")

for grade, recall in enumerate(test_metrics["recall"]):
    print(f"Grade {grade}: {recall:.4f}")

print("\nTraining completed successfully.")
print("Checkpoint:", CHECKPOINT_PATH)
