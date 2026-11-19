"""
MediPredict V7 — targeted Grade 1 recall experiment.

Strategy:
- Fine-tune from the strong V5 ordinal checkpoint (not the V3 backbone).
- Apply a moderate per-sample loss multiplier to Grade 1 images.
- Select checkpoint on validation using a balanced composite:
      selection_score = 0.70 * validation_QWK + 0.30 * Grade1_recall
- Only train and validate. TEST and AUTO_TEST are never loaded/evaluated.

Expected:
Dataset/{train,val}/{0,1,2,3,4}
models/best_model_v5.pth

Run:
    python train_v7_grade1_focus.py
"""
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report
)

SEED = 42
DATA_DIR = "Dataset"
INIT_PATH = os.path.join("models", "best_model_v5.pth")
SAVE_PATH = os.path.join("models", "best_model_v7_grade1_focus.pth")
HISTORY_PATH = os.path.join("models", "history_v7_grade1_focus.csv")

IMAGE_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 20
PATIENCE = 5
LR = 5e-6
WEIGHT_DECAY = 1e-4
GRADE1_LOSS_MULTIPLIER = 1.5
QWK_SCORE_WEIGHT = 0.70
GRADE1_RECALL_SCORE_WEIGHT = 0.30
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = [0, 1, 2, 3, 4]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transforms_for(split):
    norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=.08, contrast=.08),
            transforms.ToTensor(),
            norm
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        norm
    ])


def make_loaders():
    datasets_by_split, loaders = {}, {}
    for split in ["train", "val"]:
        ds = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=transforms_for(split))
        if ds.classes != ["0", "1", "2", "3", "4"]:
            raise ValueError(f"{split}: expected class folders 0-4, got {ds.classes}")
        loaders[split] = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=(split == "train"),
            num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
        )
        datasets_by_split[split] = ds
    return loaders, datasets_by_split


def make_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(.35),
        nn.Linear(model.classifier.in_features, 4)
    )
    return model.to(DEVICE)


def ordinal_targets(labels):
    return torch.stack([(labels > k).float() for k in range(4)], dim=1)


def decode_logits(logits):
    return (torch.sigmoid(logits) >= 0.5).sum(dim=1).long()


def metrics_for(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    recall_by_grade = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    grade1_recall = float(recall_by_grade[1])
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    score = QWK_SCORE_WEIGHT * qwk + GRADE1_RECALL_SCORE_WEIGHT * grade1_recall
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "qwk": qwk,
        "grade1_recall": grade1_recall,
        "selection_score": float(score),
        "confusion_matrix": cm
    }


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_y, all_pred = [], []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images)
        preds = decode_logits(logits)
        all_y.extend(labels.numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
    y, p = np.asarray(all_y), np.asarray(all_pred)
    return metrics_for(y, p), y, p


def main():
    seed_everything(SEED)
    os.makedirs("models", exist_ok=True)
    if not os.path.isfile(INIT_PATH):
        raise FileNotFoundError(f"V5 checkpoint not found: {INIT_PATH}")

    loaders, dsets = make_loaders()
    model = make_model()
    checkpoint = torch.load(INIT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    print("Initialized V7 from V5 checkpoint.")
    print(f"Device: {DEVICE}")
    print(f"Train images: {len(dsets['train'])} | Validation images: {len(dsets['val'])}")
    print("Train class counts (0..4):", np.bincount(dsets["train"].targets, minlength=5).tolist())
    print(f"Grade 1 per-sample loss multiplier: {GRADE1_LOSS_MULTIPLIER}")
    print(f"Checkpoint score = {QWK_SCORE_WEIGHT:.2f}*QWK + "
          f"{GRADE1_RECALL_SCORE_WEIGHT:.2f}*Grade1Recall")
    print("Test and auto_test folders will not be loaded in this experiment.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=.5, patience=2
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best_score = -float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for images, labels in loaders["train"]:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            targets = ordinal_targets(labels)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            # Average the four ordinal BCE losses per image, then modestly upweight
            # Grade 1 examples to focus optimization on the current recall weakness.
            per_image_loss = bce(logits, targets).mean(dim=1)
            sample_weights = torch.ones_like(per_image_loss)
            sample_weights[labels == 1] = GRADE1_LOSS_MULTIPLIER
            loss = (per_image_loss * sample_weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        train_loss = total_loss / max(batches, 1)
        val_metrics, y_val, p_val = evaluate(model, loaders["val"])
        scheduler.step(val_metrics["selection_score"])
        lr_now = optimizer.param_groups[0]["lr"]

        row = {k: v for k, v in val_metrics.items() if k != "confusion_matrix"}
        row.update(epoch=epoch, train_loss=train_loss, lr=lr_now)
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | TrainLoss {train_loss:.4f} | "
            f"ValAcc {val_metrics['accuracy']:.4f} | "
            f"ValBalAcc {val_metrics['balanced_accuracy']:.4f} | "
            f"ValMacroF1 {val_metrics['macro_f1']:.4f} | "
            f"ValQWK {val_metrics['qwk']:.4f} | "
            f"Grade1Recall {val_metrics['grade1_recall']:.4f} | "
            f"Score {val_metrics['selection_score']:.4f} | LR {lr_now:.2e}"
        )

        if val_metrics["selection_score"] > best_score + 1e-5:
            best_score = val_metrics["selection_score"]
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_state,
                "epoch": best_epoch,
                "best_val_selection_score": best_score,
                "best_val_qwk": val_metrics["qwk"],
                "best_val_grade1_recall": val_metrics["grade1_recall"],
                "architecture": "densenet121_ordinal4",
                "image_size": IMAGE_SIZE,
                "experiment": "V7_grade1_focus",
                "grade1_loss_multiplier": GRADE1_LOSS_MULTIPLIER
            }, SAVE_PATH)
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch {best_epoch}, "
                  f"selection score {best_score:.4f}")
            break

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    model.load_state_dict(best_state)
    final_metrics, y_val, p_val = evaluate(model, loaders["val"])

    print("\n" + "=" * 62)
    print(f"V7 BEST VALIDATION CHECKPOINT — EPOCH {best_epoch}")
    print("=" * 62)
    print(f"Accuracy:          {final_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {final_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {final_metrics['macro_f1']:.4f}")
    print(f"Quadratic QWK:     {final_metrics['qwk']:.4f}")
    print(f"Grade 1 recall:    {final_metrics['grade1_recall']:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(final_metrics["confusion_matrix"])
    print("\nClassification report:")
    print(classification_report(
        y_val, p_val, labels=LABELS,
        target_names=[f"Grade {i}" for i in LABELS], zero_division=0
    ))
    print(f"\nSaved checkpoint: {SAVE_PATH}")
    print(f"Saved history: {HISTORY_PATH}")
    print("Do not run test/auto_test evaluation yet; first inspect V7 validation results.")


if __name__ == "__main__":
    main()
