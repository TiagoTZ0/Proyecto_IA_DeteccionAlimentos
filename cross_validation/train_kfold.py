"""
train_kfold.py
---------------------------------
Validación cruzada (K-Fold) con MobileNetV3 Small.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision import models

# =========================================
# Agregar raíz del proyecto al sys.path
# =========================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

# =========================================
# Importar funciones del proyecto (src/utils)
# =========================================
from src.utils import (
    set_seed, get_device, build_dataloaders, ensure_model_dirs,
    save_labels, topk_accuracy, bootstrap_calories_json
)

# =========================================
# Configuración de rutas del proyecto
# =========================================
DATA_DIR    = PROJECT_DIR / "data" / "food-101"
IMAGES_DIR  = DATA_DIR / "images"
META_DIR    = DATA_DIR / "meta"

MODELS_DIR  = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "cross_validation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================
# Hiperparámetros (ajustados a CPU)
# =========================================
EPOCHS = 2          # Reduce a 1 si quieres probar más rápido
BATCH_SIZE = 8
LR = 3e-4
NUM_FOLDS = 3       # Aumenta si tienes más potencia
SEED = 42

# =========================================
# Funciones auxiliares
# =========================================
def build_model(n_classes: int, device):
    """Construye MobileNetV3 Small adaptada al número de clases."""
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    in_feat = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feat, n_classes)
    return model.to(device)


def evaluate(model, loader, criterion, device):
    """Evalúa el modelo en validación."""
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

# =========================================
# Entrenamiento principal K-Fold
# =========================================
def train_kfold():
    set_seed(SEED)
    device = get_device()
    ensure_model_dirs(MODELS_DIR)

    print(f"🧠 Dispositivo: {device}")
    print(f"📂 Dataset: {IMAGES_DIR}")

    # Cargar dataset usando tus utilidades
    dl_train, _, dl_test, labels = build_dataloaders(
        images_dir=IMAGES_DIR, meta_dir=META_DIR,
        batch_size=BATCH_SIZE, val_ratio=0.1  # 10% temporal, requerido por la función
    )

    dataset = dl_train.dataset
    n_classes = len(labels)
    print(f"🍱 Clases detectadas: {n_classes}")

    save_labels(labels, MODELS_DIR / "food101_classes.npy")
    bootstrap_calories_json(labels, MODELS_DIR / "calories.json")

    # Configurar K-Fold
    indices = np.arange(len(dataset))
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_accs = []

    # === Ciclo principal por fold ===
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n🟩 Fold {fold + 1}/{NUM_FOLDS}")
        print(f"→ Train: {len(train_idx)} imágenes | Val: {len(val_idx)} imágenes")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # Modelo
        model = build_model(n_classes, device)
        for p in model.features.parameters():
            p.requires_grad = False  # Transfer learning

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        best_val = 0.0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)

            val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch}/{EPOCHS} | Loss={avg_loss:.4f} | Val top1={val_acc1:.2f}% | top5={val_acc5:.2f}%")

            if val_acc1 > best_val:
                best_val = val_acc1
                torch.save(model.state_dict(), RESULTS_DIR / f"fold{fold+1}_best.pth")

        print(f"✅ Fold {fold + 1} completado | Mejor Top-1 = {best_val:.2f}%")
        fold_accs.append(best_val)

    # === Resultados finales ===
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print("\n📊 Resultados finales:")
    print(f"Top-1 promedio: {mean_acc:.2f}% ± {std_acc:.2f}%")

    results = {"fold_accuracies": fold_accs, "mean": mean_acc, "std": std_acc}
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n📁 Resultados guardados en: {RESULTS_DIR}")

# =========================================
# Punto de entrada
# =========================================
if __name__ == "__main__":
    train_kfold()
