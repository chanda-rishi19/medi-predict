"""
MediPredict V6 — DenseNet121 ordinal classification (CORAL-style cumulative targets).
Starts from the V3 DenseNet121 backbone weights, replaces the 5-class head with 4
ordinal logits, trains on train/, selects checkpoint by validation QWK, then evaluates
test/ and auto_test/ once.

Expected folders:
Dataset/{train,val,test,auto_test}/{0,1,2,3,4}
V3 checkpoint: models/best_model_v3.pth
"""
import os, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report

SEED = 42
DATA_DIR = "Dataset"
INIT_PATH = "models/best_model_v3.pth"
SAVE_PATH = "models/best_model_v6.pth"
HISTORY_PATH = "models/history_v6.csv"
IMAGE_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 35
PATIENCE = 8
LR = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def transforms_for(split):
    norm = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=.08, contrast=.08),
            transforms.ToTensor(), norm
        ])
    return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), norm])

def make_loaders():
    loaders, datasets_by_split = {}, {}
    for split in ["train", "val", "test", "auto_test"]:
        ds = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=transforms_for(split))
        if ds.classes != ["0","1","2","3","4"]:
            raise ValueError(f"{split}: expected class folders 0-4, got {ds.classes}")
        loaders[split] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split=="train"),
                                    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
        datasets_by_split[split] = ds
    return loaders, datasets_by_split

def make_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(.35), nn.Linear(model.classifier.in_features, 4))
    return model.to(DEVICE)

def load_v3_backbone(model):
    if not os.path.isfile(INIT_PATH):
        raise FileNotFoundError(f"V3 checkpoint missing: {INIT_PATH}")
    checkpoint = torch.load(INIT_PATH, map_location=DEVICE, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    own = model.state_dict()
    compatible = {k:v for k,v in state.items() if k in own and own[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    # Classifier weights do not match (V3 has 5 outputs; V5 has 4 ordinal outputs).
    print(f"Initialized from V3 backbone: loaded {len(compatible)} compatible tensors.")
    print("Ordinal classifier head initialized fresh.")

def ordinal_targets(labels):
    # Each output k learns whether true grade is > k (k=0..3).
    return torch.stack([(labels > k).float() for k in range(4)], dim=1)

def compute_mild_ordinal_pos_weights(train_dataset):
    """
    Mild per-threshold balancing for cumulative ordinal BCE.
    pos_weight[k] = sqrt(negative_count / positive_count) for target grade > k.
    Square-root scaling is deliberately less aggressive than full inverse ratio.
    """
    class_counts = np.bincount(train_dataset.targets, minlength=5).astype(np.float64)
    weights = []
    for k in range(4):
        positive = class_counts[k + 1:].sum()  # grades > k
        negative = class_counts[:k + 1].sum()  # grades <= k
        if positive <= 0 or negative <= 0:
            weights.append(1.0)
        else:
            weights.append(float(np.sqrt(negative / positive)))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE), class_counts

def decode_logits(logits):
    # Count cumulative decisions; output in 0..4. This naturally maps ordered tasks.
    return (torch.sigmoid(logits) >= 0.5).sum(dim=1).long()

def qwk(y, pred):
    return cohen_kappa_score(y, pred, weights="quadratic")

@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    losses, ys, preds = [], [], []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        targets = ordinal_targets(labels)
        if criterion is not None:
            losses.append(criterion(logits, targets).item())
        pred = decode_logits(logits)
        ys.extend(labels.cpu().numpy().tolist())
        preds.extend(pred.cpu().numpy().tolist())
    y, p = np.asarray(ys), np.asarray(preds)
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": accuracy_score(y,p),
        "balanced_accuracy": balanced_accuracy_score(y,p),
        "macro_f1": f1_score(y,p,average="macro",zero_division=0),
        "qwk": qwk(y,p)
    }
    return metrics, y, p

def report(split, y, p, metrics):
    print("\\n" + "="*58 + f"\\nV6 {split.upper()}\\n" + "="*58)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"QWK: {metrics['qwk']:.4f}")
    print("Confusion matrix:\\n", confusion_matrix(y,p,labels=[0,1,2,3,4]))
    print("Classification report:\\n", classification_report(y,p,labels=[0,1,2,3,4],
          target_names=[f"Grade {i}" for i in range(5)],zero_division=0))

def main():
    seed_everything(SEED)
    os.makedirs("models", exist_ok=True)
    loaders, dsets = make_loaders()
    model = make_model()
    load_v3_backbone(model)

    pos_weight, class_counts = compute_mild_ordinal_pos_weights(dsets["train"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("Training class counts (grades 0..4):", class_counts.astype(int).tolist())
    print("Mild ordinal pos_weight (thresholds >0..>3):", [round(x, 4) for x in pos_weight.detach().cpu().tolist()])
    print("V6 uses mild ordinal loss balancing; validation QWK remains checkpoint-selection metric.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=3)

    best_qwk, best_epoch, no_improve = -1.0, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    print(f"Device: {DEVICE} | Train: {len(dsets['train'])} | Val: {len(dsets['val'])}")
    print("Training V6 ordinal classifier...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for images, labels in loaders["train"]:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            targets = ordinal_targets(labels)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item(); n_batches += 1

        train_loss = running_loss / max(n_batches,1)
        val_metrics, _, _ = evaluate(model, loaders["val"], criterion)
        scheduler.step(val_metrics["qwk"])
        lr_now = optimizer.param_groups[0]["lr"]
        row = {"epoch":epoch, "train_loss":train_loss, **{f"val_{k}":v for k,v in val_metrics.items()}, "lr":lr_now}
        history.append(row)
        print(f"Epoch {epoch:02d}/{EPOCHS} | TrainLoss {train_loss:.4f} | "
              f"ValAcc {val_metrics['accuracy']:.4f} | ValBalAcc {val_metrics['balanced_accuracy']:.4f} | "
              f"ValMacroF1 {val_metrics['macro_f1']:.4f} | ValQWK {val_metrics['qwk']:.4f} | LR {lr_now:.2e}")

        if val_metrics["qwk"] > best_qwk + 1e-5:
            best_qwk, best_epoch, no_improve = val_metrics["qwk"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict":best_state, "epoch":epoch, "best_val_qwk":best_qwk,
                        "architecture":"densenet121_ordinal4", "image_size":IMAGE_SIZE}, SAVE_PATH)
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch {best_epoch}, Val QWK {best_qwk:.4f}")
            break

    pd.DataFrame(history).to_csv(HISTORY_PATH,index=False)
    model.load_state_dict(best_state)
    print(f"\\nBest checkpoint epoch {best_epoch} | Validation QWK {best_qwk:.4f}")
    print("\\nTest and auto_test are intentionally not evaluated in this V6 training run.")
    print("Keep those sets untouched until the V6 experiment is finalized.")
    print(f"\\nSaved model: {SAVE_PATH}")
    print(f"Saved history: {HISTORY_PATH}")

if __name__ == "__main__":
    main()
