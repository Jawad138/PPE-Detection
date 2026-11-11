"""Live PPE detection from a webcam or a video file.

Usage:
    python webcam.py                         # laptop camera
    python webcam.py --camera 1              # another camera
    python webcam.py --source site.mp4 --save outputs/site_annotated.mp4

Keys:  Q / Esc = quit,  S = save a snapshot to outputs/
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from ppe_utils import analyze, draw

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Live PPE detection")
    p.add_argument("--camera", type=int, default=0, help="camera index (0 = built-in webcam)")
    p.add_argument("--source", help="video file to use instead of the camera")
    p.add_argument("--weights", default=str(ROOT / "best.pt"))
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--save", help="optional path to save the annotated video (.mp4)")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.weights).exists():
        raise SystemExit(f"{args.weights} not found. Train the model first with `python train.py`.")

    model = YOLO(args.weights)
    if args.source:
        cap = cv2.VideoCapture(args.source)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # DirectShow opens faster on Windows
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera/video. Try a different --camera index.")

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, size)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    window = "PPE Detection  (Q = quit, S = snapshot)"
    fps, last = 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video or camera disconnected

        result = model.predict(frame, conf=args.conf, verbose=False)[0]
        persons, dets = analyze(result)
        annotated = draw(frame, persons, dets)

        # FPS (smoothed) and compliance summary banner
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-6))
        last = now
        bad = sum(not p.compliant for p in persons)
        banner = f"FPS {fps:4.1f} | People {len(persons)} | Non-compliant {bad}"
        color = (40, 40, 220) if bad else (60, 180, 75)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 32), color, -1)
        cv2.putText(annotated, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

        if writer:
            writer.write(annotated)
        cv2.imshow(window, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            path = out_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(path), annotated)
            print(f"Saved {path}")
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed with the X button

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
