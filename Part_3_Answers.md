# Part 3 — Reasoning-Based Questions

## Q1: Choosing the Right Approach

For detecting a missing label on an otherwise visually identical product, I would
start with **classification**, not detection or segmentation. The task as described
is binary (label present / label absent) on a fixed, known region of a visually
consistent product — there's no need to localize *where* the label is if the camera
position and product orientation are controlled on an assembly line, and
classification is simpler to train, faster to run, and needs far less annotation
effort (one label per image instead of bounding boxes). If the product's position or
orientation varies in frame, or if I need to know *where* the label should be to also
catch a misplaced or partially peeled label, I'd fall back to **detection**: train a
model to localize the label itself and flag "no label detected" when confidence falls
below threshold, which also gives a bounding box for a human reviewer to check. I'd
reserve **segmentation** for a further fallback only if the label's shape or partial
coverage matters (e.g. detecting a torn or partially applied label), since it's the
most annotation-expensive and slowest option of the three and isn't justified unless
the simpler approaches demonstrably fail on that specific failure mode.

## Q2: Debugging a Poorly Performing Model

My first check is always a **domain gap** between training and deployment images —
concretely, I'd place training and factory-camera frames side by side and compare
lighting, camera angle, distance/scale, resolution, motion blur, and background
clutter, since a mismatch here is the single most common cause of this symptom (and
is exactly the gap I documented in Part 1 between phone photos and real factory
footage). Next I'd run the model on a batch of the new factory images and visually
inspect failures: are boxes systematically shifted (a preprocessing/letterbox bug),
missing on certain lighting conditions (a data coverage gap), or confidently wrong on
a specific class (a label-quality or class-imbalance issue)? I'd plot the training
loss and mAP curves to rule out under/overfitting — a model that looks great on
training/validation loss but fails in deployment is a domain-gap or data-leakage
problem, not an optimization problem. I'd also check class balance and box-size
distribution in the 1,000 training images against the new images, since a detector
trained mostly on one scale or orientation generalizes poorly to others. Finally I'd
collect a small labeled sample (50-100 images) from the actual factory feed and
measure metrics on it directly — that number, not the original validation set, is the
one that matters for deployment.

## Q3: Accuracy vs Real Risk

Accuracy is close to useless here because the classes are imbalanced (defects are
presumably rare) and, more importantly, the two error types have wildly different
costs — a safety compliance system that misses a defective/unsafe product (a false
negative) is far worse than one that flags a good product for review (a false
positive), yet accuracy weights both errors equally and lets a huge majority of
easy true-negatives mask a bad false-negative rate. Missing 1 in 10 defective
products is a **10% false-negative rate on the class that actually matters**, and
that number is invisible inside a 98% accuracy figure. I'd look instead at
**recall on the defect class** (the fraction of real defects actually caught), the
**precision-recall curve** to see the trade-off at different confidence thresholds,
and the **confusion matrix** to see exactly which defect types are being missed. In a
safety context I'd also weight recall over precision deliberately (tune the
confidence threshold down, accepting more false alarms) since a missed defect can
mean an injury while a false alarm only costs a human a few seconds of review.

## Q4: Annotation Edge Cases

I would keep blurry or partially visible objects in the dataset rather than discard
them, because the deployed model will encounter exactly these cases in real video
streams (motion blur, hands partially out of frame, occlusion by other objects) — a
model that has never seen them during training will fail precisely where it matters
most in production. The trade-off is annotation consistency: a labeler has to decide
a threshold for "too blurry/occluded to label confidently," and inconsistent
judgment calls here introduce label noise that can hurt training if not handled
carefully. My practical rule is to label everything that a human can still
confidently identify, tag genuinely ambiguous or unidentifiable cases with a separate
flag (e.g. `difficult: true`) rather than dropping them outright, and optionally
exclude only the `difficult`-flagged instances from the loss during early training
while still keeping them in the validation set — that way the model is evaluated on
the realistic distribution it will actually face, without being forced to learn from
labels no human could confidently assign either.
