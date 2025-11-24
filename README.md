# 🍎 Food & Fruits AI – Clasificador de Alimentos y estimación de Calorías

> **Sistema inteligente de reconocimiento visual de platos de comida y frutas, con estimación nutricional en tiempo real.**

Este proyecto implementa un modelo de **Deep Learning** basado en la arquitectura **MobileNetV2**, entrenado mediante *Transfer Learning* sobre un dataset híbrido personalizado. La aplicación final permite a los usuarios subir fotos de sus comidas, identificar qué son y calcular las calorías totales según el peso de la porción.

---

## 🧠 Descripción Técnica

El núcleo del proyecto es una Red Neuronal Convolucional (CNN) optimizada para inferencia rápida (incluso en CPU).

* **Modelo Base:** `MobileNetV2` (Preentrenado en ImageNet).
* **Técnica:** *Full Fine-Tuning* (Reentrenamiento de capas profundas y clasificador).
* **Dataset:** Fusión personalizada de **Food-101** (Platos preparados) + **Fruits-262** (Selección de 51 frutas y verduras).
* **Frameworks:** PyTorch (Entrenamiento) y Streamlit (Despliegue Web).

---

## 🍽️ Dataset Híbrido (Food + Fruits)

El modelo ha sido entrenado para reconocer aproximadamente **152 clases distintas**, combinando una amplia variedad de platos cocinados con una selección de frutas frescas.

### 1. Platos Preparados (Food-101)
Incluye 101 categorías de comida internacional, tales como:
* `Pizza`, `Sushi`, `Hamburguesa`, `Tacos`, `Ramen`, `Paella`, `Lasagna`, `Ceviche`, `Steak`, `Risotto`, entre otros.
De los cuales vamos a usar 51 categorias.

### 2. Frutas y Verduras (Subconjunto Fruits-262)
Se integraron **51 clases específicas** seleccionadas del dataset Fruits-262, abarcando desde frutas de consumo diario hasta variedades exóticas y verduras comunes en la cocina.

**Algunas de las clases incluidas:**
* **Frutas Comunes:** Manzana, Plátano, Naranja, Mandarina, Fresa, Uva, Piña, Sandía, Durazno, Limón.
* **Frutas Exóticas/Tropicales:** Maracuyá, Pitahaya (Dragonfruit), Lúcuma, Aguaje, Chirimoya, Granadilla, Carambola, Coco.
* **Vegetales/Frutos:** Tomate, Palta (Avocado), Pimiento, Maíz, Zapallo, Berenjena.

---

## 📁 Estructura del Proyecto

Food101-Calories/
│
├── data/
│   └── food-101_fruits-262/
│       └── images/          # Carpeta UNIFICADA con las 152 clases
│
├── models/
│   ├── calories.json        # Base de datos nutricional (kcal/100g)
│   ├── food101_classes.npy  # Archivo generado con los nombres de las clases
│   └── food101_torch.pth    # Pesos del modelo entrenado (MobileNetV2)
│
├── src/
│   ├── app.py               # Aplicación Web (Streamlit)
│   ├── train.py             # Script de entrenamiento principal
│   ├── predict.py           # Script para pruebas rápidas por consola
│   └── utils.py             # Procesamiento de datos y carga dinámica
│
├── config.py                # Variables globales
├── requirements.txt         # Librerías necesarias
└── README.md                # Documentación

---

## ⚙️ Instalación y Requisitos

1.  **Clonar el repositorio o descargar el código.**
2.  **Crear un entorno virtual (opcional pero recomendado):**
    # En Windows:
    python -m venv venv
    venv\Scripts\activate
    
    # En Mac/Linux:
    python3 -m venv venv
    source venv/bin/activate

3.  **Instalar dependencias:**
    pip install -r requirements.txt

---

## 🚀 Entrenamiento del Modelo

El sistema escanea automáticamente la carpeta `data/food-101_fruits-262/images` y se adapta a la cantidad de clases que encuentre.

**Para iniciar el entrenamiento:**

python src/train.py --epochs 25 --batch-size 16

> **Nota:** Si tu equipo no tiene GPU dedicada, el script detectará CPU automáticamente. Si tienes poca memoria RAM, reduce el batch size a 8.

Al finalizar, se generarán automáticamente en la carpeta `models/`:
* `food101_torch.pth` (El cerebro de la IA).
* `food101_classes.npy` (La lista de etiquetas).
* `calories.json` (Plantilla de calorías actualizada).

---

## 💻 Ejecución de la Aplicación (Demo)

Una vez entrenado el modelo, lanza la interfaz gráfica:

streamlit run src/app.py

### Funcionalidades de la App:
1.  **📸 Reconocimiento Visual:** Sube cualquier imagen (JPG/PNG).
2.  **📊 Probabilidades:** Muestra la confianza del modelo y el Top-3 de posibles resultados.
3.  **🔥 Calculadora Nutricional:**
    * Detecta el alimento.
    * Consulta la base de datos `calories.json`.
    * Permite ajustar el peso con un *slider* para calcular el total calórico estimado de la porción.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.9+
* **Deep Learning:** PyTorch, Torchvision.
* **Arquitectura:** MobileNetV2 (Eficiente y liviana).
* **Interfaz:** Streamlit.
* **Procesamiento de Datos:** Pandas, NumPy, Pillow.

---

## 🧠 Créditos

Proyecto desarrollado para el curso de **Inteligencia Artificial: Principios y Técnicas**.  
**Universidad Privada Antenor Orrego (UPAO) – Facultad de Ingeniería**

**Equipo de Desarrollo:**
* Trigoso Zárate, Tiago André
* Velásquez Góngora, Bruno Martín
* Correa Asencio, Damer
* Chavez, Jhon
* Vergara Lopez, Junior
---

## 🛡️ Licencia

Este proyecto utiliza subconjuntos de los datasets públicos **Food-101** y **Fruits-262** con fines académicos y de investigación.