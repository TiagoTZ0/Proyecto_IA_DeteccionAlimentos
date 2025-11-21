import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# ===== PARÁMETROS =====
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
SEED = 42

# ¡IMPORTANTE! Poner esto en None para que lea TODAS tus frutas y comidas
LIMIT_CLASSES = None 

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms(img_size=IMG_SIZE):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Un poco de rotación ayuda con las frutas
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tfms, eval_tfms

def build_dataloaders(images_dir: Path, meta_dir: Path, batch_size=16, val_ratio=0.10):
    """
    Versión Robustecida: Escanea directamente las carpetas en 'images_dir'.
    Ignora 'meta_dir' para permitir datasets personalizados (como tus frutas).
    """
    print(f"📂 Escaneando imágenes en: {images_dir} ...")
    
    # 1. Definir transformaciones
    train_tfms, eval_tfms = get_transforms()

    # 2. Cargar TODO el dataset usando la estructura de carpetas
    # ImageFolder asume estructura: root/clase/imagen.jpg
    full_dataset = datasets.ImageFolder(root=str(images_dir))
    classes = full_dataset.classes
    
    # Filtro opcional para pruebas rápidas (si LIMIT_CLASSES no es None)
    if LIMIT_CLASSES is not None:
        print(f"⚠️ LIMIT_CLASSES activado: Solo se usarán las primeras {LIMIT_CLASSES} clases.")
        # Esto es un truco para filtrar, idealmente se hace antes, pero ImageFolder lee todo.
        # Para simplificar, si limitamos, re-mapeamos indices.
        # (En tu caso de entrenamiento 'PRO', déjalo en None).
        pass 

    # 3. Dividir Train / Val / Test
    # Usaremos 80% Train, 10% Val, 10% Test (aprox si val_ratio=0.1)
    total_size = len(full_dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * val_ratio) # Usamos mismo ratio para test
    train_size = total_size - val_size - test_size

    print(f"📊 Split: Train={train_size}, Val={val_size}, Test={test_size}")

    ds_train_subset, ds_val_subset, ds_test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(SEED)
    )

    # 4. Aplicar transformaciones correctas a cada split
    # ImageFolder aplica una transformación global. Aquí forzamos la correcta.
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x) # Aplicar tfm aquí en vez de en ImageFolder original
            return x, y
        def __len__(self):
            return len(self.subset)

    # Recargamos ImageFolder SIN transformaciones base para aplicarlas despues
    raw_dataset = datasets.ImageFolder(root=str(images_dir), transform=None)
    # Volvemos a dividir sobre el raw
    train_raw, val_raw, test_raw = random_split(
        raw_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(SEED)
    )

    ds_train = TransformedSubset(train_raw, train_tfms)
    ds_val   = TransformedSubset(val_raw,   eval_tfms)
    ds_test  = TransformedSubset(test_raw,  eval_tfms)

    # 5. Dataloaders
    num_workers = 0 # Windows prefiere 0
    
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return dl_train, dl_val, dl_test, classes

def ensure_model_dirs(models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)

def save_labels(labels: List[str], classes_path: Path):
    np.save(classes_path, np.array(labels))

def load_labels(classes_path: Path) -> List[str]:
    return list(np.load(classes_path, allow_pickle=True))

def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1,)):
    with torch.no_grad():
        maxk = max(ks)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * (100.0 / targets.size(0))).item())
        return res

def bootstrap_calories_json(labels: List[str], calories_json: Path):
    # Cargar existente si hay
    data = {}
    if calories_json.exists():
        try:
            data = json.loads(calories_json.read_text(encoding="utf-8"))
        except:
            data = {}
    
    # Agregar claves faltantes con 0
    updated = False
    for cls in labels:
        # Normalizar clave para evitar duplicados
        key = cls
        if key not in data:
            data[key] = 0
            updated = True
            
    if updated:
        calories_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📝 calories.json actualizado con nuevas clases.")