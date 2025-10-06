import json, random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ===== Parámetros rápidos por defecto (puedes ajustar) =====
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

SEED = 42

# Subconjunto (activar para acelerar mucho)
# Si no quieres limitar, pon LIMIT_CLASSES = None
LIMIT_CLASSES = 20          # toma solo N clases (None = todas)
MAX_TRAIN_PER_CLASS = 300   # imágenes por clase en train
MAX_VAL_PER_CLASS   = 50
MAX_TEST_PER_CLASS  = 80

def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms(img_size=IMG_SIZE):
    # Augmentations livianas para CPU
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tfms, eval_tfms

def read_classes(meta_dir: Path) -> List[str]:
    classes = (meta_dir / "classes.txt").read_text(encoding="utf-8").strip().splitlines()
    return classes

def read_split_paths(images_dir: Path, meta_dir: Path, split: str) -> List[Tuple[str, str]]:
    lines = (meta_dir / f"{split}.txt").read_text(encoding="utf-8").strip().splitlines()
    pairs = []
    for rel in lines:
        cls, _ = rel.split("/", 1)
        full = images_dir / f"{rel}.jpg"
        pairs.append((str(full), cls))
    return pairs

class Food101FileDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]], class_to_idx: Dict[str,int], tfm):
        self.items = items
        self.class_to_idx = class_to_idx
        self.tfm = tfm
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, cls = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        y = self.class_to_idx[cls]
        return x, y

def stratified_val_split(train_pairs: List[Tuple[str,str]], val_ratio: float) -> Tuple[list, list]:
    X = [p[0] for p in train_pairs]
    y = [p[1] for p in train_pairs]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_ratio, random_state=SEED, stratify=y)
    return list(zip(X_tr, y_tr)), list(zip(X_val, y_val))

def _limit_by_class(pairs: List[Tuple[str,str]], cap: int) -> List[Tuple[str,str]]:
    from collections import defaultdict
    buckets = defaultdict(list)
    out = []
    for p,c in pairs:
        if len(buckets[c]) < cap:
            buckets[c].append((p,c))
            out.append((p,c))
    return out

def build_dataloaders(images_dir: Path, meta_dir: Path, batch_size=16, val_ratio=0.10):
    classes = read_classes(meta_dir)
    class_to_idx = {c:i for i,c in enumerate(classes)}

    train_pairs = read_split_paths(images_dir, meta_dir, "train")
    test_pairs  = read_split_paths(images_dir, meta_dir, "test")

    # Split de validación estratificado
    train_pairs, val_pairs = stratified_val_split(train_pairs, val_ratio)

    # ----- ACELERADOR: limitar clases y muestras por clase -----
    if LIMIT_CLASSES is not None:
        keep = set(classes[:LIMIT_CLASSES])  # toma las primeras N clases
        classes = [c for c in classes if c in keep]
        class_to_idx = {c:i for i,c in enumerate(classes)}
        train_pairs = [(p,c) for (p,c) in train_pairs if c in keep]
        val_pairs   = [(p,c) for (p,c) in val_pairs   if c in keep]
        test_pairs  = [(p,c) for (p,c) in test_pairs  if c in keep]

        train_pairs = _limit_by_class(train_pairs, MAX_TRAIN_PER_CLASS)
        val_pairs   = _limit_by_class(val_pairs,   MAX_VAL_PER_CLASS)
        test_pairs  = _limit_by_class(test_pairs,  MAX_TEST_PER_CLASS)
    # -----------------------------------------------------------

    train_tfms, eval_tfms = get_transforms()

    ds_train = Food101FileDataset(train_pairs, class_to_idx, train_tfms)
    ds_val   = Food101FileDataset(val_pairs,   class_to_idx, eval_tfms)
    ds_test  = Food101FileDataset(test_pairs,  class_to_idx, eval_tfms)

    # En CPU, num_workers=0 y pin_memory=False suelen ser más rápidos/estables (especialmente en Windows)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return dl_train, dl_val, dl_test, classes

def ensure_model_dirs(models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)

def save_labels(labels: List[str], classes_path: Path):
    np.save(classes_path, np.array(labels))

def load_labels(classes_path: Path) -> List[str]:
    return list(np.load(classes_path, allow_pickle=True))

def softmax_np(x: np.ndarray):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

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
    if calories_json.exists():
        return
    d = {cls: 0 for cls in labels}
    calories_json.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
