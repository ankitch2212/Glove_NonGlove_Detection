# Part 3 — Reasoning-Based Questions

## Q1: Choosing the Right Approach

I'd start with **classification**, not detection or segmentation, since the task is
binary (label present/absent) on a visually consistent product — classification
needs far less annotation effort and is faster to train and run. If the product's
position or orientation varies in frame, I'd fall back to **detection** to localize
the label region and flag low-confidence detections as "missing." I'd only reach for
**segmentation** if I specifically needed to catch partial/torn labels, since it's
the most annotation-expensive option and isn't justified unless simpler approaches
demonstrably fail on that case.

## Q2: Debugging a Poorly Performing Model

My first check is a **domain gap** between training and deployment images —
lighting, camera angle, distance, and motion blur are the most common causes of this
exact symptom. I'd visually inspect failures on the new images to see if errors are
systematic (shifted boxes suggest a preprocessing bug; missed cases under certain
lighting suggest a data coverage gap), and check the training set's class balance and
box-size distribution against the new images. Finally, I'd collect and label a small
sample (50-100 images) from the actual deployment feed and measure metrics directly
on it — that number, not the original validation set, is what matters.

## Q3: Accuracy vs Real Risk

Accuracy is close to useless here because the classes are imbalanced and the two
error types have very different costs — missing a defective product (false negative)
is worse than flagging a good one for review (false positive), yet accuracy weights
both equally. Missing 1 in 10 defects is a 10% false-negative rate on the class that
actually matters, invisible inside a 98% accuracy figure. I'd look instead at
**recall on the defect class** and the **precision-recall curve**, and in a safety
context I'd deliberately tune the confidence threshold to favor recall over
precision, since a missed defect can mean an injury while a false alarm only costs a
few seconds of review.

## Q4: Annotation Edge Cases

I'd keep blurry or partially visible objects in the dataset rather than discard them,
since the deployed model will encounter exactly these cases in real video streams,
and a model that's never seen them will fail precisely where it matters most. The
trade-off is annotation consistency, so my rule is to label everything a human can
still confidently identify, flag genuinely ambiguous cases separately (e.g.
`difficult: true`) rather than dropping them, and keep them in the validation set so
the model is evaluated on the realistic distribution it will actually face.
