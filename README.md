# Food-101 Calories Recognition

Clasificador de alimentos basado en **MobileNetV2** con estimación de calorías por porción.  
El modelo se entrena con el dataset **Food-101**, aplicando *Transfer Learning* y técnicas de normalización para reconocimiento visual de alimentos y cálculo nutricional estimado.

Por el momento, se utiliza **MobileNetV3 Small** para entrenamiento rápido en CPU.

---

## 🍱 Funciona con los siguientes alimentos (subconjunto actual)

apple_pie 🍎🥧           caesar_salad 🥬  
baby_back_ribs 🍖        cannoli 🍰  
baklava 🍯               caprese_salad 🍅🧀  
beef_carpaccio 🥩        carrot_cake 🎂  
beef_tartare 🥩          ceviche 🐟🍋  
beet_salad 🥗            cheese_plate 🧀  
beignets 🍩              cheesecake 🍰  
bibimbap 🍚              chicken_curry 🍛  
bread_pudding 🍞         chicken_quesadilla 🌮  
breakfast_burrito 🌯     bruschetta 🍅🍞  

---

## 📁 Estructura del Proyecto

Food101-Calories/  
│  
├── src/  
│   ├── train.py                # Entrenamiento del modelo (MobileNetV2 o MobileNetV3 Small)  
│   ├── predict.py              # Predicción e inferencia  
│   ├── utils.py                # Funciones auxiliares y carga del dataset  
│   ├── model/                  # Carpeta para guardar modelos .pth  
│   └── __init__.py  
│  
├── data/  
│   ├── food-101/               # Dataset con imágenes e índices meta/  
│   └── nutrition_info.csv      # Información nutricional por clase (kcal)  
│  
├── app.py                      # Interfaz Streamlit  
├── requirements.txt  
├── README.md  
└── .gitignore  

---

## ⚙️ Requisitos del Sistema

- Python 3.12 o superior  
- Visual Studio Code con extensiones:
  - Python  
  - Streamlit  

Instala las librerías necesarias con:

pip install torch torchvision pillow numpy pandas streamlit

*(Para CPU, no necesitas CUDA; el modelo está optimizado con MobileNetV2 o MobileNetV3 Small.)*

---

## 🧠 Entrenamiento del Modelo

1. Descarga y extrae **Food-101** en la ruta:
D:\Datasets\food-101
├── images\ (101 carpetas de clases)
└── meta\train.txt, test.txt, classes.txt

2. Ejecuta el entrenamiento desde la terminal de VS Code:
python src/train.py --root "D:\Datasets" --epochs 10 --batch-size 16 --freeze-base

3. El modelo entrenado se guardará como:
src/model/food101_mobilenetv2.pth

---

## 🔍 Predicción por Consola

Ejemplo de inferencia:

python src/predict.py --image "data/test/pasta.jpg"

Salida esperada:

Predicción: spaghetti_bolognese (Top-1)
Probabilidad: 0.89
Calorías estimadas: 435 kcal por porción

---

## 💻 Interfaz con Streamlit

Ejecuta la interfaz gráfica con:

streamlit run app.py

La aplicación permite:
- Subir una imagen de un alimento.  
- Ver el nombre de la comida detectada.  
- Mostrar las calorías estimadas por 100 g.  
- Ajustar los gramos para calcular el valor total.  
- Visualizar la probabilidad de clasificación.  

---

## ⚡ Entrenamiento Rápido (para pruebas)

Si deseas entrenar más rápido en CPU:
python src/train.py --epochs 3 --batch-size 8 --limit-classes 20 --freeze-base

(Esto entrena solo con 20 clases y menos imágenes por clase para validar el pipeline de entrenamiento.)

---

## 📊 Modelo

- **Arquitectura:** MobileNetV2 (preentrenada en ImageNet)  
- **Versión rápida:** MobileNetV3 Small (para CPU o pruebas cortas)  
- **Método:** Transfer Learning  
- **Capas finales ajustadas:** Linear (1280 → 101 clases)  
- **Optimización:** Adam (lr = 1e-4)  
- **Normalización:** Transformaciones y escalado [0, 1]  
- **Pérdida:** CrossEntropyLoss  

---

## 🚀 Pasos para Ejecutar el Proyecto Completo

1. **Instalar dependencias**
pip install -r requirements.txt

2. **Descargar y extraer el dataset Food-101** en la carpeta `data/`.

3. **Entrenar el modelo**
- Entrenamiento rápido (20 clases):
  python src/train.py --epochs 3 --batch-size 8 --limit-classes 20 --freeze-base
- Entrenamiento completo (101 clases):
  python src/train.py --epochs 10 --batch-size 16 --freeze-base

4. **Verificar que el modelo entrenado (.pth)** esté en `src/model/`.

5. **Ejecutar la aplicación**
streamlit run app.py

6. **Subir una imagen de un alimento** (por ahora, alguno de los 20 entrenados).  
Verás la predicción, la probabilidad y las calorías estimadas.

---

## 🧩 Créditos

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
