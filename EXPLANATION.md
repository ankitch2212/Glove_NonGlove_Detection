# How This Works — A Plain-Language Guide

This document explains the whole submission in everyday language. If you don't work
in computer vision day-to-day, start here — the other files (READMEs, source code)
assume more background.

## 1. What the test asked for

- **Part 1:** build a program that looks at photos and draws a box around every hand
  it sees, labeling each one "wearing a glove" or "bare hand" — plus a confidence
  score for how sure it is. Written in Python.
- **Part 2:** do the exact same thing again, but in C++ instead of Python — a
  different, faster, more low-level programming language often used when code needs
  to run directly on a factory floor device rather than a full computer.
- **Part 3:** answer four short questions about how you'd approach similar
  real-world computer vision problems and their trade-offs.

## 2. What was built — a tour of the files

```
submission/
├── Part_1_Glove_Detection/    the Python version
│   ├── detection_script.py     <- run this to detect hands in a folder of photos
│   ├── prepare_dataset.py      <- got the training photos ready
│   ├── train.py                <- taught the model what gloves and bare hands look like
│   ├── export_model.py         <- converted the trained model to other file formats
│   └── models/, samples/, output/, logs/, README.md
├── Part_2_Cpp/                 the C++ version, doing the same job
│   ├── src/, include/, CMakeLists.txt
│   └── models/, config/, input/, output/, logs/, README.md
└── Part_3_Answers.md           the written answers
```

## 3. How it works, end to end

Think of it as an assembly line for a single photo:

1. **A photo comes in.** It could be any size — a phone photo, a security camera
   frame, anything.
2. **Resize and pad ("letterbox").** The model only understands images of one exact
   size (416×416 pixels here). The photo is shrunk to fit inside that square while
   keeping its proportions correct (so hands don't get squished), and the leftover
   space is filled with plain grey padding.
3. **The model looks at the whole image at once** and proposes roughly 3,500
   candidate boxes — most of them nonsense, covering empty background, overlapping
   duplicates of the same hand, and so on. For each candidate box, it also scores how
   confident it is that the box contains a `gloved_hand` and how confident it is that
   it contains a `bare_hand`.
4. **Throw away the low-confidence guesses.** Anything below the confidence
   threshold (0.5 by default — "the model thinks there's less than a 50% chance
   this is real") gets discarded.
5. **Merge duplicate boxes ("non-maximum suppression," or NMS).** If the model draws
   three overlapping boxes around the same hand, this step keeps only the
   best-scoring one and throws away the rest.
6. **Map the boxes back to the original photo.** Since step 2 resized and padded the
   image, the coordinates need to be translated back to match the original,
   un-padded photo.
7. **Draw the results and save a record.** A colored box and label are drawn onto a
   copy of the photo (green for glove, red for bare hand), and a small text file is
   written down listing exactly what was found and where.

Every step exists because the step before it produces "too much, too messy"
information — the pipeline is a series of filters that turns "3,500 raw guesses"
into "a handful of accurate, labeled boxes."

## 4. Why each decision was made

**Why this particular model (YOLO11n)?** "YOLO" ("You Only Look Once") is a family
of fast object-detection models. The "n" stands for "nano" — the smallest version.
This matters because this machine has no graphics card (GPU); training and running a
detector normally leans heavily on a GPU for speed. The nano model is small enough to
train and run acceptably fast on an ordinary CPU instead.

**Why was the model only partly retrained ("frozen backbone")?** A detection model
has two halves: a "backbone" that has learned to recognize general shapes, edges, and
textures (trained on millions of everyday photos), and a "head" that turns those
general observations into "this specific thing is here." Rather than retraining
both halves from scratch — slow, and unnecessary since the backbone's general
knowledge is still useful for recognizing hands — only the head was retrained, and
the backbone's knowledge was reused as-is (this is called "transfer learning").
Measured directly on this machine: retraining everything would have taken roughly
233 milliseconds per training image, versus 144 milliseconds when only the head is
retrained — cutting total training time from about 45 minutes to about 10 minutes,
with barely any loss in accuracy. The actual training run took 11.7 minutes and
ended with 96.6% accuracy on held-out validation images (see the metric explanation
in Part 1's README).

**Why 416×416 pixels instead of a larger size like 640×640?** The amount of
computation needed grows with the *square* of the image size — a 640-pixel image
needs roughly 2.4 times more computation than a 416-pixel one. Hands in this
dataset's photos are large and close to the camera, so shrinking to 416 barely hurts
accuracy while meaningfully speeding up both training and everyday use.

**Why this dataset?** A public dataset of 1,708 phone photos of gloved and bare
hands (Kaggle, MIT license — free to use, no special permission needed) was already
labeled with exactly the two categories needed, was small enough to download and use
quickly, and — after visually spot-checking several labeled examples — was confirmed
to be labeled correctly (i.e., "glove" boxes really do contain a hand wearing a
glove, not the reverse).

**Why convert the model to ONNX at all?** The trained model is normally saved in a
PyTorch-specific file format that only Python programs using PyTorch can open. ONNX
("Open Neural Network Exchange") is a shared, cross-language file format that many
different programming languages and tools can read — it's the bridge that lets the
C++ program in Part 2 use the exact same trained model without needing PyTorch or
Python installed at all.

**Why does the C++ version have to do extra work that the Python version didn't?**
The Python tool used (Ultralytics) automatically handles steps 2, 6, and 7 above
behind the scenes as a convenience. The raw exported model file contains none of
that convenience logic — just the underlying mathematics. So the C++ code has to
implement the resizing math, the coordinate translation math, and the duplicate-box
merging (step 5) itself, by hand. This was, in fact, the trickiest part of the whole
project: the model's raw output is organized in a slightly unintuitive way (all the
box positions listed first, then separately all the confidence scores — rather than
each box's information grouped together), and getting that organization wrong
produces confident-looking but completely wrong results. This was caught early by
comparing the C++ program's output against the Python program's output on the same
photo and confirming the numbers matched exactly.

## 5. Honest results and limitations

- **Accuracy:** on held-out photos the model never trained on, it correctly
  identified gloved vs. bare hands with roughly 97% precision and recall (mAP50 of
  0.966 — a standard combined accuracy score for detection models; see Part 1's
  README for the raw numbers). That's a strong result for an 11.7-minute training
  run on only 400 photos.
- **The real limitation:** the training photos are casual phone snapshots of a
  single hand close to the camera against everyday backgrounds — not the kind of
  wide-angle, multi-person, sometimes-blurry footage a factory safety camera would
  actually capture. A model trained only on this dataset would likely need
  additional fine-tuning on real factory footage before being trusted in production;
  this exact gap (training photos vs. real deployment photos) is also the subject of
  the debugging checklist in Part 3, Question 2.
- **What a production version would need:** more images, ideally real factory
  camera footage rather than phone photos; more training time (a full backbone
  retrain rather than the shortened version used here, given more time budget); and
  ideally a GPU, which would make both of those practical.

## 6. How to run everything

**Part 1 (Python):**
```bash
cd Part_1_Glove_Detection
pip install -r ../requirements.txt
python detection_script.py --input samples --output output --confidence 0.5
```

**Part 2 (C++):** requires a C++ compiler, CMake, OpenCV, and ONNX Runtime to be
installed first (see [`Part_2_Cpp/README.md`](Part_2_Cpp/README.md) for exact
download links and versions, since this machine had none of them pre-installed).
Once installed:
```powershell
cd Part_2_Cpp
cmake -S . -B build -DOpenCV_DIR=<path to opencv>/build -DONNXRUNTIME_ROOT=<path to onnxruntime>
cmake --build build --config Release
.\build\glove_detector.exe --config config\config.yaml
```

## 7. Part 3 in brief

**Q1 (which technique to use for a missing-label check):** start with the simplest
tool that solves the problem — image classification ("does this photo show a label
or not?") rather than the more complex detection or segmentation, since knowing
*where* the label is usually isn't necessary. Fall back to detection (which also
draws a box) only if the product's position varies in frame.

**Q2 (debugging a model that works in testing but not in the real factory):** the
most common cause of this exact symptom is a mismatch between the training photos
and the real deployment photos — different lighting, camera angle, distance, or
blur. The write-up lays out a concrete checklist for spotting and confirming this.

**Q3 (why "98% accuracy" can still be a bad result):** when the thing you're trying
to catch is rare (like defective products), a model can score extremely high on
overall accuracy while still missing a dangerous fraction of the real problems. The
write-up explains why "recall on the defective class" is the number that actually
matters here, not overall accuracy.

**Q4 (whether to keep blurry or partially-hidden training photos):** yes — because
those are exactly the kinds of imperfect photos the finished system will encounter
in real life, and a model that's never seen them during training will struggle when
it meets them for real. The write-up covers the labeling trade-offs involved.
