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
# CONFIGURACIÓN INICIAL
# -----------------------------------------------------------
sys.path.append(str((Path(__file__).parent / "src").resolve()))

st.set_page_config(page_title="Food-101 Calories", page_icon="🍽️", layout="centered")

# -----------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------

@st.cache_resource
def load_labels():
    return list(np.load(config.CLASSES_PATH, allow_pickle=True))

@st.cache_resource
def load_calories():
    if not Path(config.CALORIES_JSON).exists():
        Path(config.CALORIES_JSON).write_text(json.dumps({}, indent=2), encoding="utf-8")
    return json.loads(Path(config.CALORIES_JSON).read_text(encoding="utf-8"))

@st.cache_resource
def load_model(n_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modelo ligero para CPU
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_feat = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feat, n_classes)

    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

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
# CARGA DE MODELO Y DATOS
# -----------------------------------------------------------
labels = load_labels()
cal_map = load_calories()
model, device = load_model(len(labels))
tfm = get_eval_tfm()

# -----------------------------------------------------------
# INTERFAZ PRINCIPAL CON SIDEBAR
# -----------------------------------------------------------
menu = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "🍲 Clasificador de Alimentos", "📋 Alimentos Detectables"]
)

# -----------------------------------------------------------
# SECCIÓN 1: INICIO
# -----------------------------------------------------------
if menu == "🏠 Inicio":
    st.title("🍽️ Food-101 Calories Recognition")
    st.write("""
    Bienvenido a **Food-101 Calories Recognition**, una aplicación desarrollada con **Python, PyTorch y Streamlit**
    que permite reconocer alimentos e inferir su valor calórico estimado.

    ### 🔍 ¿Qué hace esta IA?
    - Detecta el tipo de alimento a partir de una imagen.
    - Calcula la probabilidad de acierto.
    - Muestra las **calorías estimadas por porción (100 g)** y permite ajustar el peso.
    
    ### 🧠 Tecnología
    - **Modelo:** MobileNetV3 Small (versión ligera para CPU)
    - **Dataset:** Subconjunto del *Food-101* (20 clases)
    - **Método:** *Transfer Learning* sobre arquitectura preentrenada en ImageNet.

    Puedes comenzar la detección en la pestaña **🍲 Clasificador de Alimentos**.
    """)

# -----------------------------------------------------------
# SECCIÓN 2: CLASIFICADOR
# -----------------------------------------------------------
elif menu == "🍲 Clasificador de Alimentos":
    st.title("🍲 Clasificador de Alimentos + Calorías (Food-101)")
    st.write("Sube una imagen y obtendrás la predicción del alimento y su valor calórico estimado.")

    uploaded = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
    c1, c2 = st.columns([1, 1])

    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        c1.image(image, caption="Imagen cargada", use_container_width=True)

        x = tfm(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x).cpu().numpy().squeeze()
        probs = softmax_np(logits)
        top_idx = int(np.argmax(probs))
        top_class = labels[top_idx]
        top_prob = float(probs[top_idx])

        kcal_100 = float(cal_map.get(top_class, 0))
        with c2:
            st.subheader("Resultado")
            st.write(f"**Clase predicha:** {top_class}")
            st.write(f"**Confianza:** {top_prob:.2%}")
            st.write(f"**kcal por 100 g (tabla):** {kcal_100:.0f}")

            gramos = st.slider("Gramos consumidos", min_value=10, max_value=500, value=150, step=10)
            kcal = (gramos / 100.0) * kcal_100 if kcal_100 > 0 else 0.0
            st.metric("Calorías estimadas", f"{kcal:.0f} kcal")

            st.write("**Top-3 predicciones:**")
            top3 = np.argsort(-probs)[:3]
            for i, idx in enumerate(top3, 1):
                st.write(f"{i}. {labels[idx]} — {probs[idx]:.2%}")

            st.info(f"Si {top_class} muestra 0 kcal, edita {config.CALORIES_JSON} para agregar su valor correspondiente.")
    else:
        st.info("Carga una imagen para comenzar.")

# -----------------------------------------------------------
# SECCIÓN 3: ALIMENTOS DETECTABLES
# -----------------------------------------------------------
elif menu == "📋 Alimentos Detectables":
    st.title("📋 Alimentos que puede reconocer el modelo")
    st.write("""
    Este modelo ha sido entrenado con un **subconjunto de 20 clases** del dataset *Food-101* 
    para permitir entrenamientos rápidos en CPU.  
    En la versión final con MobileNetV2, se incluirán las 101 clases completas.
    """)

    subset_classes = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
        "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
        "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
        "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla"
    ]

    st.write("### 🍱 Clases detectables actualmente:")
    for food in subset_classes:
        kcal = cal_map.get(food, 0)
        st.write(f"- **{food.replace('_', ' ').title()}** — {kcal if kcal > 0 else '⚠️ sin datos'} kcal/100g")

    st.info("⚙️ Puedes actualizar los valores calóricos editando el archivo `calories.json` en la carpeta de tu proyecto.")

# -----------------------------------------------------------
# NOTA FINAL PARA EL DESARROLLO
# -----------------------------------------------------------
# Si se usa MobileNetV2 (versión final):
# model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# in_feat = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(in_feat, n_classes)
