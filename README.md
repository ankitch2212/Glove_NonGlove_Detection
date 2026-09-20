# Gloved vs Bare Hand Detection — Submission

A safety-compliance object detector for two classes: `gloved_hand` and `bare_hand`,
delivered as a Python pipeline (Part 1), a C++ port (Part 2), and a written reasoning
section (Part 3).

**Start here if you want the plain-language version:** [`EXPLANATION.md`](EXPLANATION.md)
walks through what was built, how the pipeline works end to end, and why each
decision was made, without assuming CV background.

## Structure

```
submission/
├── EXPLANATION.md              <- plain-language walkthrough of the whole test
├── README.md                   <- this file
├── requirements.txt            <- Python dependencies (Part 1)
├── Part_1_Glove_Detection/     <- Python detection pipeline + fine-tuned model
│   ├── detection_script.py     <- the required CLI script
│   ├── prepare_dataset.py      <- dataset normalization
│   ├── train.py                <- training recipe
│   ├── export_model.py         <- ONNX + OpenVINO export
│   ├── models/                 <- best.pt, glove_detector.onnx, classes.txt
│   ├── samples/, output/, logs/
│   └── README.md                <- dataset, model, training, results, how to run
├── Part_2_Cpp/                 <- C++ port using ONNX Runtime + OpenCV
│   ├── CMakeLists.txt, include/, src/
│   ├── models/, config/, input/, output/, logs/
│   └── README.md                <- build + run instructions
└── Part_3_Answers.md           <- reasoning write-up (4 questions)
```

## Quick start

**Part 1 (Python):**
```bash
cd Part_1_Glove_Detection
pip install -r ../requirements.txt
python detection_script.py --input samples --output output --confidence 0.5
```

**Part 2 (C++):** see [`Part_2_Cpp/README.md`](Part_2_Cpp/README.md) for the full
build steps (requires a C++ toolchain, OpenCV, and ONNX Runtime).

## Notes

- The Python pipeline (Part 1) needed no external downloads beyond the dataset and
  pretrained base weights — the ML stack (PyTorch, Ultralytics, OpenCV, ONNX Runtime)
  was already available.
- The C++ pipeline (Part 2) required installing a full toolchain (MSVC via Visual
  Studio Build Tools, CMake) plus the OpenCV and ONNX Runtime C++ SDKs, none of which
  ship with a plain Python environment.
- See each part's own README for details, and `Part_3_Answers.md` for the reasoning
  write-up.
