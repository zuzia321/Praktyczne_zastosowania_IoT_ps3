
from pathlib import Path
import random
import shutil
import yaml

BASE_DIR = Path(__file__).resolve().parent

IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
OUTPUT_DIR = BASE_DIR / "dataset"

VAL_RATIO = 0.2
SEED = 42

with open(BASE_DIR / "data.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

names = config["names"]

if isinstance(names, dict):
    names = [names[i] for i in range(len(names))]

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = sorted(
    p for p in IMAGES_DIR.iterdir()
    if p.suffix.lower() in extensions
)

if len(images) < 2:
    raise ValueError("Potrzebne są co najmniej 2 obrazy.")

for img in images:
    label = LABELS_DIR / (img.stem + ".txt")
    if not label.exists():
        raise FileNotFoundError(f"Brak etykiety: {label}")

random.seed(SEED)
random.shuffle(images)

val_count = max(1, round(len(images) * VAL_RATIO))

val_images = images[:val_count]
train_images = images[val_count:]

def copy_files(files, subset):
    img_dest = OUTPUT_DIR / "images" / subset
    lbl_dest = OUTPUT_DIR / "labels" / subset

    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)

    for img in files:
        label = LABELS_DIR / (img.stem + ".txt")

        shutil.copy2(img, img_dest / img.name)
        shutil.copy2(label, lbl_dest / label.name)

copy_files(train_images, "train")
copy_files(val_images, "val")

dataset_config = {
    "path": OUTPUT_DIR.as_posix(),
    "train": "images/train",
    "val": "images/val",
    "nc": len(names),
    "names": names
}

with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(
        dataset_config,
        f,
        allow_unicode=True,
        sort_keys=False
    )

print("Przygotowanie zakończone!")
print(f"Trening: {len(train_images)} obrazów")
print(f"Walidacja: {len(val_images)} obrazów")
print(f"Liczba klas: {len(names)}")
print(f"Klasy: {names}")
print(f"Konfiguracja: {OUTPUT_DIR / 'data.yaml'}")
