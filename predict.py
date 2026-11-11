"""Run PPE detection on an image or a folder of images.

Usage:
    python predict.py sample_images/
    python predict.py path/to/image.jpg --conf 0.3
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from ppe_utils import analyze, draw

ROOT = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="image file or folder")
    p.add_argument("--weights", default=str(ROOT / "best.pt"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default=str(ROOT / "outputs"))
    args = p.parse_args()

    src = Path(args.source)
    images = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    model = YOLO(args.weights)
    for img_path in images:
        result = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        persons, dets = analyze(result)
        annotated = draw(result.orig_img, persons, dets)
        cv2.imwrite(str(out_dir / img_path.name), annotated)

        ok = sum(p.compliant for p in persons)
        print(f"{img_path.name}: {len(persons)} person(s), {ok} compliant, "
              f"{len(persons) - ok} non-compliant")
    print(f"\nAnnotated images saved to {out_dir}")


if __name__ == "__main__":
    main()
