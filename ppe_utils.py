"""Shared helpers: run detection and decide PPE compliance per person."""
from dataclasses import dataclass, field

import cv2
import numpy as np

# Construction-PPE classes: helmet, gloves, vest, boots, goggles, Person,
# no_helmet, no_goggle, no_gloves, no_boots, forklift, no_vest. We match by
# lowercase name so it keeps working if the class order differs.
REQUIRED_PPE = {"helmet", "vest"}
PPE_ITEMS = {"helmet", "gloves", "vest", "boots", "goggles"}
MISSING_CLASSES = {"no_helmet": "helmet", "no_vest": "vest", "no_gloves": "gloves",
                   "no_goggle": "goggles", "no_boots": "boots"}

GREEN, RED, ORANGE, WHITE = (60, 180, 75), (40, 40, 220), (0, 140, 255), (255, 255, 255)


@dataclass
class PersonStatus:
    box: tuple
    wearing: set = field(default_factory=set)
    missing: set = field(default_factory=set)

    @property
    def compliant(self) -> bool:
        return not (REQUIRED_PPE - self.wearing) and not (self.missing & REQUIRED_PPE)


def _overlap(inner, outer) -> float:
    """Fraction of `inner` box area that lies inside `outer`."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def analyze(result, overlap_thresh: float = 0.5):
    """Assign detected equipment to each person and return (persons, detections)."""
    names = result.names
    dets = []
    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                              result.boxes.cls.cpu().numpy().astype(int),
                              result.boxes.conf.cpu().numpy()):
        dets.append((tuple(map(float, box)), names[cls].lower(), float(conf)))

    persons = [PersonStatus(box=b) for b, n, _ in dets if n == "person"]
    for box, name, _ in dets:
        if name not in PPE_ITEMS and name not in MISSING_CLASSES:
            continue  # skip people, forklifts, etc.
        # attach to the person box that contains most of this item
        best = max(persons, key=lambda p: _overlap(box, p.box), default=None)
        if best is None or _overlap(box, best.box) < overlap_thresh:
            continue
        if name in MISSING_CLASSES:
            best.missing.add(MISSING_CLASSES[name])
        else:
            best.wearing.add(name)
    return persons, dets


def draw(image: np.ndarray, persons, dets) -> np.ndarray:
    """Draw equipment boxes and a per-person compliance label."""
    img = image.copy()
    s = max(img.shape[:2]) / 1000  # scale text/lines with image size
    for box, name, conf in dets:
        if name == "person":
            continue
        x1, y1, x2, y2 = map(int, box)
        color = RED if name.startswith("no_") else ORANGE if name in PPE_ITEMS else (200, 120, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, round(2 * s)))
        _label(img, f"{name} {conf:.2f}", x1, y1, color, 0.32 * s)

    for p in persons:
        x1, y1, x2, y2 = map(int, p.box)
        color = GREEN if p.compliant else RED
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, round(3 * s)))
        missing = (REQUIRED_PPE - p.wearing) | p.missing
        text = "OK" if p.compliant else "MISSING: " + ", ".join(sorted(missing))
        _label(img, text, x1, y1, color, 0.7 * s, below=True)
    return img


def _label(img, text, x, y, color, scale=0.5, below=False):
    """Draw a filled text label at the top-left corner of a box, kept inside the image."""
    scale = max(scale, 0.4)
    thick = max(1, round(scale * 2))
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = max(3, h // 3)
    # person labels go just inside the top of the box; equipment labels sit above it
    top = y if below else y - h - 2 * pad
    top = min(max(top, 0), img.shape[0] - h - 2 * pad)
    x = min(max(x, 0), max(0, img.shape[1] - w - 2 * pad))
    cv2.rectangle(img, (x, top), (x + w + 2 * pad, top + h + 2 * pad), color, -1)
    cv2.putText(img, text, (x + pad, top + h + pad), cv2.FONT_HERSHEY_SIMPLEX, scale,
                WHITE, thick, cv2.LINE_AA)
