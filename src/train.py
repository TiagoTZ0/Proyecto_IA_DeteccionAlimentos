import argparse
import sys
from pathlib import Path

# ------------------------------------------------------
# 1. EL ENTRENAMIENTO FUNCIONE "NORMAL" (Ejecución directa)
# ------------------------------------------------------
# Esto permite ejecutar "python src/train.py"
file_path = Path(__file__).resolve()
root_path = file_path.parents[1]  # Sube un nivel para encontrar la carpeta raiz
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
# ------------------------------------------------------

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm

# ------------------------------------------------------
# 2. CONFIGURACIÓN DE RUTAS
# ------------------------------------------------------
PROJECT_DIR = root_path
DATA_ROOT   = PROJECT_DIR / "data" / "food-101" 
IMAGES_DIR  = DATA_ROOT / "images"
META_DIR    = DATA_ROOT / "meta" # (Se ignorará en utils, pero lo dejamos definido)

MODELS_DIR   = PROJECT_DIR / "models"
MODEL_PATH   = MODELS_DIR / "food101_torch.pth"
CLASSES_PATH = MODELS_DIR / "food101_classes.npy"
CALORIES_JSON= MODELS_DIR / "calories.json"

# ------------------------------------------------------
# 3. HIPERPARÁMETROS
# ------------------------------------------------------
EPOCHS = 25        
BATCH_SIZE = 16    # Si tu laptop sufre, bájalo a 8
LR = 1e-4          
VAL_SPLIT = 0.10
SEED = 42

# Utils (Ahora sí funcionará el import)
from src.utils import (
    set_seed, get_device, build_dataloaders, ensure_model_dirs,
    save_labels, topk_accuracy, bootstrap_calories_json
)

def build_model(n_classes: int, device):
    print(f"🏗️ Construyendo MobileNet V2 para {n_classes} clases...")
    
    # MobileNet V2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Ajustar capa final
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, n_classes)
    
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
    print(f"🚀 Iniciando entrenamiento en: {device} usando MobileNetV2")
    ensure_model_dirs(MODELS_DIR)

    # Carga de datos
    dl_train, dl_val, dl_test, labels = build_dataloaders(
        images_dir=IMAGES_DIR, meta_dir=META_DIR,
        batch_size=args.batch_size, val_ratio=args.val_split
    )
    n_classes = len(labels)
    print(f"🍎 Clases detectadas: {n_classes}")
    
    save_labels(labels, CLASSES_PATH)
    bootstrap_calories_json(labels, CALORIES_JSON)

    model = build_model(n_classes, device)

    # Descongelar todo para aprender bien las frutas
    for p in model.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler corregido (sin verbose)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0
    
    print(f"⏳ Entrenando por {args.epochs} épocas...")
    
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl_train, ncols=100, desc=f"Epoch {epoch}/{args.epochs}")
        
        train_loss_sum = 0.0
        n_train = 0
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * x.size(0)
            n_train += x.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validación
        val_loss, val_acc1, val_acc5 = evaluate(model, dl_val, criterion, device)
        
        scheduler.step(val_acc1)
        
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            torch.save({"state_dict": model.state_dict(), "labels": labels}, MODEL_PATH)
            print(f"⭐ ¡Mejorado! Val Acc: {val_acc1:.2f}% -> Guardado.")
        else:
            print(f"   Val Acc: {val_acc1:.2f}% (Best: {best_val_acc:.2f}%)")

    print("\n🏁 Entrenamiento finalizado. Evaluando en Test set...")
    
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_acc1, test_acc5 = evaluate(model, dl_test, criterion, device)
    print(f"📊 RESULTADO FINAL -> Test Acc Top-1: {test_acc1:.2f}% | Top-5: {test_acc5:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--val-split", type=float, default=VAL_SPLIT)
    args = ap.parse_args()
    train(args)