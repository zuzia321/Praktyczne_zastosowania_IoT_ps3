
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "dataset" / "data.yaml"

    if torch.cuda.is_available():
        device = 0
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("GPU CUDA niedostępne - trening na CPU")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_path),
        epochs=100,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        patience=20,
        project=str(base_dir / "runs"),
        name="my_yolov8",
        exist_ok=True
    )

    best_model_path = (
        base_dir / "runs" / "my_yolov8"
        / "weights" / "best.pt"
    )

    output_path = base_dir / "best.pt"

    if not best_model_path.is_file():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {best_model_path}"
        )

    shutil.copy2(best_model_path, output_path)

    print(f"\nModel skopiowany do: {output_path}")

    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(data=str(data_path), device=device)

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    print("\nTrening zakończony!")
    print("Model zapisany w:", best_model_path)

if __name__ == "__main__":
    main()
