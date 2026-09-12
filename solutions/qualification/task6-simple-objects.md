# FAIO 2025 Qualification — Task 6: Simple Objects

Source: [`faio-2025/qualification/task6_Simple_Objects.md`](../../faio-2025/qualification/task6_Simple_Objects.md) · Yandex.Contest 81908, problem 6

## 1. What the problem asks

Write a program that **counts** geometric figures in a blueprint image. Given a 2048 × 2048 RGB image containing `N` figures (`100 ≤ N ≤ 500`) — triangles, rectangles and circles — output the exact count. Tested on **8000** images; submit `submit.csv` with one integer per row.

Guarantees in the statement: figures are cleanly drawn with solid coloured borders, do not overlap ambiguously, the background is distinguishable, and **each figure is a single connected component when considering edge pixels**. You only need to count, not classify.

Scoring is a customised accuracy on relative error:

```
Error Rate = abs(ŷ − y) / y
Accuracy   = 1 − min(1, Error Rate)
```

with a 0.55 floor below which the verdict is 0.

## 2. Knowledge required

- Image loading and the RGB array layout
- Thresholding / background separation
- **Connected components** or **contour finding** — the core tool
- Contour filtering by area to drop noise, and the external-vs-nested contour distinction (a ring border yields both an outer and an inner contour)
- Optionally shape classification by vertex count, aspect ratio or circularity, if it helps the counting logic
- Batch processing: 8000 images, so per-image cost matters

**Given in the statement:** a fair amount. The Tips section names the approach — "edge detection, contour finding, and shape classification using properties like number of corners, aspect ratio, or circularity" — and the Input section states that each figure is one connected component of edge pixels. That connectivity guarantee is the whole solution. Treat contour finding as **given knowledge**; no ML theory is supplied, and none is needed.

## 3. Languages and libraries

**Explicitly given — the only task in the round that states it.** Quoting the Tips section:

> You may use any programming language and libraries (OpenCV, scikit-image, PIL, etc.)

So:

- **Allowed:** any language, any library. OpenCV, scikit-image and PIL are named by the organisers
- **Forbidden:** nothing is listed

Source code is not required either — only `submit.csv` is graded.

## Solution

Because every figure is one connected component of border pixels, counting external contours is the answer. No training, no model.

```python
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def count_figures(path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # background is the dominant value; anything else is a stroke
    bg = np.bincount(gray.ravel()).argmax()
    mask = (np.abs(gray.astype(int) - int(bg)) > 20).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= 20)

rows = [(i, count_figures(p)) for i, p in enumerate(sorted(Path("test_images").iterdir()))]
pd.DataFrame(rows, columns=["id", "count"]).to_csv("submit.csv", index=False)
```

Key choices:

- `RETR_EXTERNAL` returns only outermost contours, so a hollow shape counts once rather than twice. Using `RETR_LIST` roughly doubles the count — the single biggest scoring mistake available here.
- Derive the background value per image instead of assuming white; the statement allows coloured backgrounds, only promising it is distinguishable.
- Filter tiny contours by area to kill anti-aliasing speckles, but keep the threshold low — genuine small figures exist.

Validate on `train.csv` with its images, which gives ground-truth counts to tune the two constants (colour tolerance, minimum area).

**Traps:**

- **The metric as printed in the archive is inverted:** "`0, if Accuracy > 0.55`" would zero out good solutions and reward bad ones. The intent is clearly the reverse — 0 when accuracy is *below* 0.55, rescaled above it. Assume the intended reading, and do not try to game the text as written.
- Since the error is relative, a fixed off-by-a-few is cheap at `N = 500` and expensive at `N = 100`. Test on the low-count images.
- Touching or nested figures merge into one component. The statement promises this does not happen ambiguously, but check a few training images before trusting it.
- 8000 images at 2048 × 2048: read as grayscale where possible and avoid per-image Python loops over pixels.
