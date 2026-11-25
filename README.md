# Praktyczne_zastosowania_IoT_ps3

## Uruchomienie gotowego modelu
Wszystkie poniższe polecenia należy wykonywać w terminalu (np. Anaconda Prompt, PowerShell) po przejściu do katalogu `YOLO/`:

```bash
cd YOLO
```

Zainstaluj wymagane biblioteki:

```bash
python -m pip install -r requirements.txt
```

Uruchom wykrywanie obiektów z kamery:

```bash
python camera.py
```

Program wykorzystuje gotowy model `best.pt`. Naciśnij `Q`, aby zakończyć. Jeżeli wyłączenie poprzez `Q` nie działa po prostu wyłączyć `ctrl+c`

## Ponowne trenowanie modelu

Zainstaluj zależności treningowe (obejmują również biblioteki z `requirements.txt`):

```bash
python -m pip install -r requirements-train.txt
```

Przygotuj dane (80% treningowe, 20% walidacyjne):

```bash
python prepare_dataset.py
```

Rozpocznij trening:

```bash
python train.py
```

Trening wykorzystuje **YOLOv8n**, maksymalnie **100 epok** oraz mechanizm **Early Stopping** (`patience=20`).

Najlepszy wytrenowany model jest automatycznie kopiowany do pliku `best.pt` w głównym katalogu projektu.

Do trenowania na GPU NVIDIA wymagany jest PyTorch z obsługą CUDA. W przeciwnym razie trening odbywa się na CPU.