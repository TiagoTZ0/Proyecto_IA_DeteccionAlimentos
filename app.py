import os, sys, io, json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
from torchvision import models, transforms

import config

# -----------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------
sys.path.append(str((Path(__file__).parent / "src").resolve()))
st.set_page_config(page_title="Food & Fruits AI", page_icon="🍎", layout="centered")

# -----------------------------------------------------------
# FUNCIONES
# -----------------------------------------------------------

@st.cache_resource
def load_labels():
    if not Path(config.CLASSES_PATH).exists():
        return []
    return list(np.load(config.CLASSES_PATH, allow_pickle=True))

def _norm(s: str) -> str:
    return str(s).strip().lower().replace('_', ' ').replace('-', ' ')

@st.cache_data(show_spinner=False)
def load_calories() -> dict:
    p = Path(config.CALORIES_JSON)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {_norm(k): float(v) for k, v in raw.items()}

def kcal_lookup(label: str, cal_map: dict) -> float:
    # Busqueda robusta
    cands = [
        label,
        label.lower(),
        label.replace('_', ' '),
        label.replace('_', '-'),
        _norm(label)
    ]
    for c in cands:
        v = cal_map.get(c) # Buscar tal cual
        if v is None: 
            v = cal_map.get(_norm(c)) # Buscar normalizado
        
        if v is not None and v > 0:
            return float(v)
    return 0.0

@st.cache_resource
def load_model(n_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CAMBIO A MOBILENET V2 ---
    # No cargamos pesos de internet, cargaremos los nuestros
    model = models.mobilenet_v2(weights=None) 
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, n_classes)

    if Path(config.MODEL_PATH).exists():
        ckpt = torch.load(config.MODEL_PATH, map_location=device)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        st.error("⚠️ No se encontró el modelo entrenado (.pth). Ejecuta train.py primero.")

    model.eval().to(device)
    return model, device

def get_eval_tfm():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

# -----------------------------------------------------------
# CARGA
# -----------------------------------------------------------
labels = load_labels()
cal_map = load_calories()

if len(labels) > 0:
    model, device = load_model(len(labels))
    tfm = get_eval_tfm()
else:
    st.warning("⚠️ No se encontraron etiquetas. Entrena el modelo primero.")
    st.stop()

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
menu = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "🍎 Clasificador", "📋 Lista de Alimentos"]
)

if menu == "🏠 Inicio":
    st.title("🍎 Food & Fruits AI Recognition")
    st.write(f"""
    Bienvenido. Este modelo ha sido actualizado para reconocer **{len(labels)} clases** diferentes, 
    incluyendo platos peruanos, internacionales y una gran variedad de frutas.

    ### 🚀 Novedades MobileNetV2
    - Arquitectura más robusta.
    - Dataset expandido (Fruits-262 + Food-101).
    - Sistema de calorías integrado.
    """)

elif menu == "🍎 Clasificador":
    st.title("📸 Clasificador Inteligente")
    uploaded = st.file_uploader("Sube una foto...", type=["jpg", "jpeg", "png"])
    
    c1, c2 = st.columns([1, 1])
    
    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        c1.image(image, caption="Tu foto", use_container_width=True)

        # Predicción
        x = tfm(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x).cpu().numpy().squeeze()
        probs = softmax_np(logits)
        
        # Top 1
        top_idx = int(np.argmax(probs))
        top_class = labels[top_idx]
        top_prob = float(probs[top_idx])
        
        kcal_100 = kcal_lookup(top_class, cal_map)

        with c2:
            st.subheader("Resultado")
            st.success(f"**{top_class.replace('_', ' ').title()}**")
            st.write(f"Confianza: **{top_prob:.2%}**")
            
            if kcal_100 > 0:
                st.info(f"🔥 **{kcal_100:.0f} kcal** / 100g")
                gramos = st.slider("Peso (g):", 10, 500, 150, 10)
                total_kcal = (gramos/100)*kcal_100
                st.metric("Total Calorías", f"{total_kcal:.0f}")
            else:
                st.warning("⚠️ Calorías no registradas en calories.json")

            st.write("---")
            st.write("**Otras posibilidades:**")
            top3 = np.argsort(-probs)[:3]
            for i, idx in enumerate(top3):
                p_val = probs[idx]
                if p_val > 0.01: # Solo mostrar si tiene > 1%
                    st.caption(f"{labels[idx]} ({p_val:.1%})")

elif menu == "📋 Lista de Alimentos":
    st.title(f"📋 Catálogo ({len(labels)} clases)")
    st.write("Lista completa de lo que el modelo puede detectar actualmente:")
    
    # Mostrar en una tabla limpia
    data_rows = []
    for lab in sorted(labels):
        k = kcal_lookup(lab, cal_map)
        val_str = f"{k:.0f}" if k > 0 else "-"
        data_rows.append({"Alimento": lab.replace("_", " ").title(), "Kcal/100g": val_str})
    
    st.dataframe(data_rows, use_container_width=True)