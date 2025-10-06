import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import config
from src.utils import load_labels, softmax_np

def get_eval_tfms():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

def load_model(n_classes: int, device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, n_classes)
    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels()
    model = load_model(len(labels), device)
    tfm = get_eval_tfms()

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).cpu().numpy().squeeze()
    probs = softmax_np(logits)
    top3 = np.argsort(-probs)[:3]

    print("Top-3 predicciones:")
    for i, idx in enumerate(top3, 1):
        print(f"{i}. {labels[idx]} — {probs[idx]*100:.2f}%")

    # Calorías
    cal_map = json.loads(Path(config.CALORIES_JSON).read_text(encoding="utf-8"))
    top_class = labels[int(top3[0])]
    kcal_100 = float(cal_map.get(top_class, 0))
    if args.grams and kcal_100 > 0:
        kcal = (args.grams / 100.0) * kcal_100
        print(f"\nClase: {top_class} | kcal/100g: {kcal_100:.0f} | gramos: {args.grams} → kcal estimadas: {kcal:.0f}")
    else:
        print(f"\nClase: {top_class} | kcal/100g: {kcal_100:.0f} (edita {config.CALORIES_JSON} para mejorar)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--grams", type=float, default=None)
    args = ap.parse_args()
    main(args)
