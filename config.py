from pathlib import Path

# Rutas base
PROJECT_DIR = Path(__file__).parent
DATA_ROOT   = PROJECT_DIR / "data" / "food-101"      # .../data/food-101
IMAGES_DIR  = DATA_ROOT / "images"
META_DIR    = DATA_ROOT / "meta"

MODELS_DIR  = PROJECT_DIR / "models"
MODEL_PATH  = MODELS_DIR / "food101_torch.pth"
CLASSES_PATH = MODELS_DIR / "food101_classes.npy"
CALORIES_JSON = MODELS_DIR / "calories.json"

# Parámetros de entrenamiento
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 12
LR = 3e-4
VAL_SPLIT = 0.10   
NUM_WORKERS = 2
SEED = 42

# Normalización tipo ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
