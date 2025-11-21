import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# Aseguramos que el script encuentre config y utils
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import load_labels

def get_eval_tfms():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

def load_model(n_classes: int, device):
    print(f"⏳ Cargando MobileNetV2 para {n_classes} clases...")
    model = models.mobilenet_v2(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, n_classes)
    
    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
        
    model.eval().to(device)
    return model

def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def _norm(s):
    return str(s).strip().lower().replace('_', ' ').replace('-', ' ')

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Cargar Labels
    labels = load_labels(config.CLASSES_PATH)
    
    # 2. Cargar Modelo
    model = load_model(len(labels), device)
    tfm = get_eval_tfms()

    # 3. Predecir
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"❌ Error: No existe la imagen {img_path}")
        return

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x).cpu().numpy().squeeze()
    
    probs = softmax_np(logits)
    top3 = np.argsort(-probs)[:3]

    # 4. Cargar Calorías
    cal_map = {}
    if Path(config.CALORIES_JSON).exists():
        raw = json.loads(Path(config.CALORIES_JSON).read_text(encoding="utf-8"))
        cal_map = {_norm(k): float(v) for k, v in raw.items()}

    print(f"\n🔍 Resultado para: {img_path.name}")
    print("-" * 30)
    
    top_idx = top3[0]
    top_class = labels[top_idx]
    
    # Buscar calorías
    kcal_val = cal_map.get(_norm(top_class), 0.0)
    
    print(f"🏆 PREDICCIÓN: {top_class} ({probs[top_idx]*100:.1f}%)")
    
    if kcal_val > 0:
        print(f"🔥 Calorías base: {kcal_val:.0f} kcal / 100g")
        if args.grams:
            total = (args.grams / 100) * kcal_val
            print(f"⚖️  Para {args.grams}g: {total:.0f} kcal")
    else:
        print("⚠️  Calorías no disponibles en JSON.")

    print("-" * 30)
    print("Otras opciones probables:")
    for i, idx in enumerate(top3[1:], 2):
        print(f"{i}. {labels[idx]} ({probs[idx]*100:.1f}%)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="Ruta a la imagen")
    ap.add_argument("--grams", type=float, default=None, help="Peso en gramos para calcular kcal")
    args = ap.parse_args()
    main(args)