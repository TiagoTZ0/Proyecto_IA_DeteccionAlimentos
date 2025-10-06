import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision import models
from tqdm import tqdm

# Rutas y archivos (ajusta a tu config.py si lo prefieres)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT   = PROJECT_DIR / "data" / "food-101"
IMAGES_DIR  = DATA_ROOT / "images"
META_DIR    = DATA_ROOT / "meta"

MODELS_DIR   = PROJECT_DIR / "models"
MODEL_PATH   = MODELS_DIR / "food101_torch.pth"
CLASSES_PATH = MODELS_DIR / "food101_classes.npy"
CALORIES_JSON= MODELS_DIR / "calories.json"

# Hiperparámetros rápidos
EPOCHS = 6
BATCH_SIZE = 16
LR = 3e-4
VAL_SPLIT = 0.10
SEED = 42

# Utils rápidos
from src.utils import (
    set_seed, get_device, build_dataloaders, ensure_model_dirs,
    save_labels, topk_accuracy, bootstrap_calories_json
)

def build_model(n_classes: int, device):
    # MobileNetV3 Small (más veloz en CPU que V2)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_feat = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feat, n_classes)
    return model.to(device)

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, n = 0.0, 0
    acc1_sum, acc5_sum = 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
            a1, a5 = topk_accuracy(logits, y, ks=(1,5))
            acc1_sum += a1 * x.size(0) / 100.0
            acc5_sum += a5 * x.size(0) / 100.0
    return loss_sum / n, (acc1_sum / n) * 100.0, (acc5_sum / n) * 100.0

def train(args):
    set_seed(SEED)
    device = get_device()
    ensure_model_dirs(MODELS_DIR)

    dl_train, dl_val, dl_test, labels = build_dataloaders(
        images_dir=IMAGES_DIR, meta_dir=META_DIR,
        batch_size=args.batch_size, val_ratio=args.val_split
    )
    n_classes = len(labels)
    save_labels(labels, CLASSES_PATH)
    bootstrap_calories_json(labels, CALORIES_JSON)

    model = build_model(n_classes, device)

    # Congelar base al inicio para acelerar; luego puedes descongelar si quieres
    for p in model.features.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl_train, ncols=100, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_acc1, val_acc5 = evaluate(model, dl_val, criterion, device)
        print(f"Val: loss={val_loss:.4f} | top1={val_acc1:.2f}% | top5={val_acc5:.2f}%")

        if val_acc1 > best_val:
            best_val = val_acc1
            torch.save({"state_dict": model.state_dict(), "labels": labels}, MODEL_PATH)
            print(f"✔ Guardado mejor modelo en {MODEL_PATH}")

    # Test final
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_acc1, test_acc5 = evaluate(model, dl_test, criterion, device)
    print(f"Test: loss={test_loss:.4f} | top1={test_acc1:.2f}% | top5={test_acc5:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--val-split", type=float, default=VAL_SPLIT)
    args = ap.parse_args()
    train(args)
