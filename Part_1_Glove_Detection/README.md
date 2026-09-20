# Part 1 — Gloved vs Bare Hand Detection (Python)

## Dataset

- **Name:** `rsaishivani/hand-glove-dataset` (Kaggle mirror of a Roboflow export)
- **Source:** https://www.kaggle.com/datasets/rsaishivani/hand-glove-dataset
  (upstream: https://universe.roboflow.com/glove-uylxg/glove-q7czq/dataset/1)
- **License:** MIT
- **Format:** YOLOv8 `.txt` labels, already split into `train` / `valid` / `test`
- **Classes:** source `['glove', 'no_glove']` renamed 1:1 to the required
  `['gloved_hand', 'bare_hand']` — no remapping logic needed, the class indices
  already line up.
- **Size:** 1,708 images total (train 1,462 / valid 138 / test 108). Every image has
  a non-empty label file. Train-split class balance is near-even (792 `glove` boxes /
  765 `no_glove` boxes), which is why a plain random subsample (see below) is safe —
  no stratified sampling was needed.
- **Verification done before training:** ground-truth boxes for a sample of images
  were rendered and inspected by eye to confirm class 0 really is a gloved hand and
  class 1 really is a bare hand (a silently-swapped class mapping trains and scores
  perfectly while being completely wrong, so this check matters).
- **Known limitation:** these are casual phone photos (varied backgrounds, indoor
  lighting, close-up single-hand framing), not factory-camera footage. See "What
  worked / what didn't" below.

## Model

- **Architecture:** YOLO11n (Ultralytics), ~2.6M parameters — the smallest model in
  the family, chosen specifically because training ran on CPU only (no GPU on this
  machine).
- **Base weights:** COCO-pretrained `yolo11n.pt`.
- **Fine-tuning strategy:** backbone frozen (`freeze=10`), so only the detection head
  trains on top of the COCO-pretrained backbone's features. This is genuine transfer
  learning — it reuses visual features already learned from millions of COCO images —
  and cuts CPU training cost roughly in half versus a full fine-tune (measured on
  this machine: 144 ms/img frozen vs 233 ms/img full, both at imgsz=416).

## Preprocessing / Training

```
python prepare_dataset.py --src <extracted_raw_dataset> --dst data --max-train 400 --max-val 100
python train.py --data data/data.yaml --model yolo11n.pt --epochs 10 --imgsz 416 --batch 16 --workers 4
```

- Subsampled to 400 train / 100 val images (random, seed 0) to keep CPU training time
  short; the source balance meant no stratification was required.
- `imgsz=416`, not 640 — compute scales with the square of the side (640²/416² ≈ 2.4×),
  and hands in this dataset are large enough in frame that 416 loses little accuracy
  for a large saving in training/inference cost.
- `cache="ram"` removes JPEG decode from the training hot loop (400 images at 416px is
  well under 1 GB decoded).
- `optimizer="AdamW", lr0=0.002, cos_lr=True` — with only 10 epochs there isn't enough
  schedule length for SGD's usual warmup+decay to converge; a modest AdamW LR moves the
  head quickly and the cosine schedule settles it in the final epochs.
- `close_mosaic=2` disables mosaic augmentation for the last 2 epochs so the final
  checkpoint's statistics reflect real (not composited) images.
- **Total training time: ~10 minutes** (measured), for 10 epochs on 400 images.

### Results (this run)

Measured wall-clock: **10 epochs completed in 0.195 hours (11.7 minutes)** on 400
training images, 12-core CPU, no GPU — matching the planned ~10 minute budget.

Final validation metrics for the shipped `models/best.pt` checkpoint (see
`runs/glove_v1/results.png` / `results.csv` for the full per-epoch curve):

| Metric | Value |
|---|---|
| Precision | 0.974 |
| Recall | 0.974 |
| mAP50 | 0.966 |
| mAP50-95 | 0.679 |

mAP50 climbed quickly from 0.62 after epoch 1 to 0.91+ by epoch 4 and 0.97-0.99 from
epoch 6 onward — a strong result for a 10-epoch frozen-backbone run on only 400
images, and clear evidence the COCO-pretrained backbone's features transfer well to
hand shapes with only the detection head adapting.

## What worked / what didn't

**Worked:**
- Freezing the backbone was the right call for the time budget — mAP50 crossed 0.85
  within 3-4 epochs, showing the COCO-pretrained features transfer well to hand
  shapes even with the head alone adapting to the new classes.
- The dataset's near-even class balance meant no special sampling logic was needed —
  one less thing to get wrong.

**Didn't / limitations:**
- The dataset is close-up phone photos of a single hand against varied household
  backgrounds — not multi-worker factory-camera footage with smaller, more numerous,
  more motion-blurred hands at a distance. A model trained only on this data would
  likely underperform on real factory video without additional fine-tuning on
  in-domain images (see Part 3, Q2 for the debugging checklist this implies).
- Only 400/100 images and 10 epochs were used to fit the time budget; a production
  model would benefit from the full 1,462-image train split, more epochs, and
  eventually unfreezing the backbone for a longer full fine-tune.
- The validation split's class balance (38 `gloved_hand` / 102 `bare_hand`) is skewed
  versus the near-even training set — a larger held-out set would give a more
  reliable per-class metric.

## How to run

```bash
pip install -r ../requirements.txt

# 1. Prepare the dataset (already done for this submission; data/ is included)
python prepare_dataset.py --src <path_to_extracted_raw_dataset> --dst data

# 2. Train (already done; models/best.pt is included)
python train.py --data data/data.yaml --epochs 10 --imgsz 416

# 3. Run detection on a folder of images
python detection_script.py --input samples --output output --confidence 0.5

# 4. Export to ONNX (+ OpenVINO bonus)
python export_model.py --weights models/best.pt --imgsz 416 --formats onnx,openvino
```

Both exports are included: `models/glove_detector.onnx` (verified output shape
`[1, 6, 3549]` — 4 box coords + 2 classes, matching Part 2's C++ decoder) and
`models/glove_detector_openvino/` (`best.xml` + `best.bin` IR format).

`detection_script.py` CLI:

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Folder of `.jpg`/`.jpeg`/`.png` images |
| `--output` | *(required)* | Folder to save annotated images into |
| `--confidence` | `0.5` | Confidence threshold |
| `--weights` | `models/best.pt` | Path to trained weights |
| `--logs` | `logs` | Folder for per-image JSON logs |
| `--imgsz` | `416` | Inference image size |
| `--iou` | `0.45` | NMS IoU threshold |

Output: one annotated `.jpg` per input image in `output/`, and one `.json` per input
image in `logs/` (e.g. `image1.jpg` → `logs/image1.json`), in the format:

```json
{
  "filename": "image1.jpg",
  "detections": [
    {"label": "gloved_hand", "confidence": 0.92, "bbox": [x1, y1, x2, y2]}
  ]
}
```

Images with no detections still produce a log file with `"detections": []`.
