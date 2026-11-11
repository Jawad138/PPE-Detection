"""Train a YOLO model on the Ultralytics Construction-PPE dataset locally.

Usage:
    python train.py                     # defaults tuned for a 4 GB GPU
    python train.py --epochs 100 --batch 16 --model yolov8n.pt
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Train PPE detection model")
    p.add_argument("--model", default="yolo26n.pt", help="pretrained weights (yolo26n.pt, yolov8n.pt, ...)")
    p.add_argument("--data", default="construction-ppe.yaml", help="dataset yaml (auto-downloaded by Ultralytics)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="8 fits a 4 GB GPU; use -1 for auto-batch")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--workers", type=int, default=2, help="dataloader workers (keep low on Windows)")
    p.add_argument("--name", default="ppe_detection")
    return p.parse_args()


def main():
    args = parse_args()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
    )

    # Evaluate the best checkpoint on the validation split
    run_dir = ROOT / "runs" / args.name
    best = run_dir / "weights" / "best.pt"
    metrics = YOLO(best).val(data=args.data, imgsz=args.imgsz, device=device)
    print("\n=== Validation metrics (best.pt) ===")
    print(f"Precision : {metrics.box.mp:.3f}")
    print(f"Recall    : {metrics.box.mr:.3f}")
    print(f"mAP50     : {metrics.box.map50:.3f}")
    print(f"mAP50-95  : {metrics.box.map:.3f}")

    # Copy portfolio artifacts to the project root
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    for name in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
        src = run_dir / name
        if src.exists():
            shutil.copy(src, results_dir / name)
    shutil.copy(best, ROOT / "best.pt")
    print(f"\nSaved best.pt and plots to {ROOT}")


if __name__ == "__main__":
    main()
