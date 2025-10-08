# 🍱 Food-101 – Detección de Alimentos y Calorías

Clasificador de alimentos basado en **MobileNetV2** con estimación de calorías por porción.  
El modelo se entrena con el dataset **Food-101**, aplicando *Transfer Learning* y técnicas de normalización para el reconocimiento visual de alimentos y el cálculo nutricional estimado.

Actualmente, se utiliza **MobileNetV3 Small** para entrenamientos rápidos en CPU, mientras que **MobileNetV2** se empleará en la versión final por su mayor precisión y estabilidad.

---

## 🧠 Descripción General

El objetivo del proyecto es desarrollar un sistema de Inteligencia Artificial capaz de **reconocer alimentos a partir de imágenes** y **estimar su valor calórico promedio por porción**.  
El modelo fue implementado en **Python (PyTorch)** para el entrenamiento y **Streamlit** para la interfaz de usuario.

---

## 🍽️ Subconjunto de Clases (versión actual)

Durante la fase de validación en CPU, se trabajó con un subconjunto de **20 clases representativas** del dataset Food-101 para optimizar los tiempos de entrenamiento y pruebas:

apple_pie 🍎🥧  caesar_salad 🥬  
baby_back_ribs 🍖  cannoli 🍰  
baklava 🍯  caprese_salad 🍅🧀  
beef_carpaccio 🥩  carrot_cake 🎂  
beef_tartare 🥩  ceviche 🐟🍋  
beet_salad 🥗  cheese_plate 🧀  
beignets 🍩  cheesecake 🍰  
bibimbap 🍚  chicken_curry 🍛  
bread_pudding 🍞  chicken_quesadilla 🌮  
breakfast_burrito 🌯  bruschetta 🍅🍞  

Cada clase contiene aproximadamente **750 imágenes de entrenamiento** y **250 de prueba**.

---

## 📁 Estructura del Proyecto

Food101-Calories/
│
├── data/
│ ├── food-101/
│ │ ├── images/
│ │ └── meta/
│ └── imagenes_propias/
│
├── models/
│ ├── calories.json       ##Se generan a partir del entrenamiento
│ ├── food101_classes.npy
│ └── food101_torch.pth
│
├── src/ # Scripts de entrenamiento y utilidades
│ ├── predict.py # Clasificación de imágenes alimenticias
│ ├── train.py # El script principal de entrenamiento
│ └── utils.py # Funciones auxiliares del proyecto
│
├── cross_validation/ # Carpeta de validación cruzada
│ ├── notes.md # Notas relacionadas a la validación cruzada
│ └── train_kfold.py # Script de entrenamiento específico para K-Fold
│
├── app.py # Interfaz Streamlit
├── config.py # Configuración general del proyecto
├── README.md # Documentación del proyecto
├── .gitignore # Archivos a ignorar por Git
└── requirements.txt # Dependencias Necesarias
---


## ⚙️ Requisitos del Sistema

- **Python 3.12 o superior**  
- **Visual Studio Code** con extensiones:
  - Python  
  - Streamlit  

Instalación de librerías necesarias:


*(Para CPU no se necesita CUDA; MobileNetV3 Small está optimizada para entrenamientos ligeros.)*

---

## 🧩 Entrenamiento del Modelo

1. **Descargar y extraer el dataset Food-101** dentro de la carpeta del proyecto:

Food101-Calories/data/food-101/
├── images/ # 101 carpetas de clases
└── meta/ # Archivos train.txt, test.txt, classes.txt

2. **Ejecutar el entrenamiento desde la terminal:**

python src/train.py --root "data/food-101" --epochs 10 --batch-size 16 --freeze-base

3. **El modelo entrenado se guardará automáticamente en:**

models/food101_torch.pth

---

## ⚡ Entrenamiento Rápido (para pruebas)

Si deseas entrenar más rápido en CPU:

python src/train.py --epochs 3 --batch-size 8 --limit-classes 20 --freeze-base

*(Esto entrena solo con 20 clases y menos imágenes por clase para validar el pipeline de entrenamiento.)*

---

## 🔁 Validación Cruzada (Cross-Validation)

Para evaluar la capacidad de generalización, se utilizó **K-Fold Cross-Validation (K = 3)**.  
Cada fold se entrenó durante **2 épocas** con **batch size = 8**, optimizador **AdamW** y **CrossEntropyLoss**.

Los modelos generados se almacenan en:

cross_validation/results/
├── fold_1_best.pth
├── fold_2_best.pth
├── fold_3_best.pth
└── summary.json

**Resultados obtenidos:**

| Fold | Pérdida de Validación | Precisión Top-1 (%) | Precisión Top-5 (%) |
|------|------------------------|---------------------|---------------------|
| 1    | 1.6924                 | 53.80              | 88.25              |
| 2    | 1.6725                 | 50.85              | 85.35              |
| 3    | 1.6949                 | 55.95              | 86.50              |
| **Promedio ± Desv.Est.** | — | **53.53 ± 2.09** | — |

📊 Los resultados demuestran un comportamiento estable del modelo entre los diferentes folds, validando su robustez incluso en CPU.

---

## 🔍 Predicción por Consola

Ejemplo de inferencia:

python src/predict.py --image "data/imagenes_propias/pasta.jpg"

**Salida esperada:**

Predicción: spaghetti_bolognese (Top-1)
Probabilidad: 0.89
Calorías estimadas: 435 kcal por porción

---

## 💻 Interfaz con Streamlit

Ejecuta la interfaz gráfica con:

streamlit run app.py


La aplicación permite:
- Subir una imagen de un alimento  
- Ver el nombre de la comida detectada  
- Mostrar las calorías estimadas por 100 g  
- Ajustar los gramos para calcular el valor total  
- Visualizar la probabilidad de clasificación  

---

## 📊 Modelo

- **Arquitectura principal:** MobileNetV2 (preentrenada en ImageNet)  
- **Versión de prueba:** MobileNetV3 Small (para CPU y pruebas rápidas)  
- **Método:** Transfer Learning  
- **Capas finales:** Linear (1280 → N clases)  
- **Optimización:** AdamW (lr = 3e-4, weight_decay = 1e-4)  
- **Pérdida:** CrossEntropyLoss  
- **Transformaciones:** Normalización y *data augmentation*  

---

## 🚀 Pasos para Ejecutar el Proyecto Completo

1. **Instalar dependencias**

pip install -r requirements.txt

2. **Descargar y extraer el dataset Food-101** en `data/food-101/`

3. **Entrenar el modelo**

- Entrenamiento rápido (20 clases):

python src/train.py --epochs 3 --batch-size 8 --limit-classes 20 --freeze-base

- Entrenamiento completo (101 clases):
python src/train.py --epochs 10 --batch-size 16 --freeze-base

4. **Validar el modelo (K-Fold Cross Validation)**  
python cross_validation/train_kfold.py

5. **Ejecutar la aplicación**
streamlit run app.py


6. **Subir una imagen y visualizar resultados:**
- Clase detectada  
- Probabilidad  
- Calorías estimadas  

---

## 🧠 Créditos

Proyecto desarrollado como parte del curso de **Inteligencia Artificial: Principios y Técnicas**  
**Universidad Privada Antenor Orrego (UPAO) – Facultad de Ingeniería**

**Autores:**  
- Trigoso Zárate, Tiago André  
- Velásquez Góngora, Bruno Martín  
- Correa Asencio, Damer  

---

## 🛡️ Licencia

Uso educativo y de investigación.  
Basado parcialmente en el dataset público **Food-101**.
