"""
detection_script.py — run the fine-tuned gloved_hand / bare_hand detector over
a folder of images, save annotated copies, and log detections as one JSON
file per image.

Usage:
    python detection_script.py --input samples --output output --confidence 0.5
    python detection_script.py --input samples --output output --confidence 0.5 \
        --weights models/best.pt --logs logs --imgsz 416 --iou 0.45
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# BGR colors for annotation, keyed by class name.
COLORS = {
    "gloved_hand": (0, 180, 0),   # green
    "bare_hand": (0, 0, 220),     # red
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list  # [x1, y1, x2, y2], ints, absolute pixel coords

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": self.confidence, "bbox": self.bbox}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Folder of .jpg/.jpeg/.png images to run detection on")
    p.add_argument("--output", required=True, help="Folder to save annotated images into")
    p.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (default 0.5)")
    p.add_argument("--weights", default="models/best.pt", help="Path to trained weights")
    p.add_argument("--logs", default="logs", help="Folder to save per-image JSON logs into")
    p.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    return p.parse_args()


def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def iter_images(input_dir: Path):
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def read_image(path: Path) -> np.ndarray:
    # cv2.imread fails silently on non-ASCII Windows paths; imdecode is robust.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"could not encode image for: {path}")
    buf.tofile(str(path))


def run_inference(model: YOLO, img: np.ndarray, conf: float, imgsz: int, iou: float) -> list[Detection]:
    results = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device="cpu", verbose=False)
    result = results[0]
    h, w = img.shape[:2]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        label = result.names[cls_id]
        confidence = round(float(box.conf.item()), 2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = [
            int(np.clip(x1, 0, w - 1)),
            int(np.clip(y1, 0, h - 1)),
            int(np.clip(x2, 0, w - 1)),
            int(np.clip(y2, 0, h - 1)),
        ]
        detections.append(Detection(label=label, confidence=confidence, bbox=bbox))
    return detections


def annotate(image: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = image.copy()
    for d in dets:
        color = COLORS.get(d.label, (255, 255, 255))
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text = f"{d.label} {d.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        chip_y = max(th + 4, y1)  # clamp so the label chip stays inside the frame
        cv2.rectangle(out, (x1, chip_y - th - 4), (x1 + tw + 4, chip_y), color, -1)
        cv2.putText(out, text, (x1 + 2, chip_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_log(log_dir: Path, filename: str, dets: list[Detection]) -> None:
    log_path = log_dir / f"{Path(filename).stem}.json"
    payload = {"filename": filename, "detections": [d.to_dict() for d in dets]}
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    log_dir = Path(args.logs)

    if not input_dir.is_dir():
        print(f"error: input folder does not exist: {input_dir}", file=sys.stderr)
        return 1

    images = list(iter_images(input_dir))
    if not images:
        print(f"error: no .jpg/.jpeg/.png images found in {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)

    total_dets, gloved, bare = 0, 0, 0
    t_start = time.time()

    for img_path in images:
        img = read_image(img_path)
        t0 = time.time()
        dets = run_inference(model, img, args.confidence, args.imgsz, args.iou)
        elapsed_ms = (time.time() - t0) * 1000

        annotated = annotate(img, dets)
        write_image(output_dir / img_path.name, annotated)
        write_log(log_dir, img_path.name, dets)

        for d in dets:
            if d.label == "gloved_hand":
                gloved += 1
            elif d.label == "bare_hand":
                bare += 1
        total_dets += len(dets)

        print(f"{img_path.name}: {len(dets)} detection(s) in {elapsed_ms:.0f} ms")

    total_s = time.time() - t_start
    print(f"\n{len(images)} images processed, {total_dets} detections "
          f"({gloved} gloved_hand, {bare} bare_hand), {total_s:.1f}s elapsed")
    print(f"Annotated images -> {output_dir}")
    print(f"Detection logs    -> {log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
