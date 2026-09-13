# T28 · Contours and connected components — Day 4

**Anchor task(s):**
- [`faio-2025/qualification/task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md)
- [`solutions/qualification/task6-simple-objects.md`](../solutions/qualification/task6-simple-objects.md)

Rating 0 · **exam-probability rank 1** · ~50 min

This single topic solves task 6 outright — no model, no training. It is the highest-value 50 minutes in the whole day.

## [concept-first]

**Connected component.** Start from a binary mask: pixels are 1 (stroke) or 0 (background). Two 1-pixels are *connected* if you can walk from one to the other stepping only on 1-pixels. A connected component is a maximal such group. Counting components is counting objects — provided each object is exactly one component.

**4- vs 8-connectivity.** The neighbours of a pixel are either the 4 edge-sharing ones (up/down/left/right) or those plus the 4 diagonals, making 8.

```
4-connectivity          8-connectivity
    . X .                   X X X
    X o X                   X o X
    . X .                   X X X
```

A one-pixel-wide diagonal line is **8-connected but not 4-connected**. Under 4-connectivity a rotated triangle's thin diagonal edge shatters into dozens of components, and a count of 300 figures becomes 4000. For strokes that can be diagonal — every rotated figure in task 6 — use 8-connectivity. OpenCV's `findContours` is 8-connected for the foreground by construction.

**Contour.** The ordered boundary coordinates of a component. `cv2.findContours(mask, mode, method)` returns `(contours, hierarchy)`; `mask` must be single-channel `uint8` with non-zero foreground.

**The retrieval mode is the whole exam.** A hollow figure's border is a *ring* of stroke pixels. A ring has two boundaries: an outer one and an inner one (the hole's edge).

- `RETR_EXTERNAL` — outermost boundaries only. One ring → **1** contour.
- `RETR_LIST` — every boundary, flat, no hierarchy. One ring → **2** contours.
- `RETR_TREE` — every boundary plus the full nesting hierarchy, so you can filter by depth yourself.

So `RETR_LIST` on an image of hollow figures returns roughly **double** the truth. Task 6's error rate is

$$
\text{Error Rate} = \frac{\lvert \hat{y} - y \rvert}{y}
$$

so $\hat{y} = 2y$ gives Error Rate 1.0, Accuracy 0.0 — below the 0.55 floor, verdict 0. One wrong enum, whole task lost. This is the single biggest scoring mistake available in the problem.

**`method`.** `CHAIN_APPROX_NONE` stores every boundary pixel; `CHAIN_APPROX_SIMPLE` keeps only the endpoints of straight runs. For counting they are identical, and `SIMPLE` is smaller and faster — use it by default. (`approxPolyDP` in T29 is a different, lossy simplification; do not confuse the two.)

**Two cleanup tools.** *Area filter:* `cv2.contourArea(c)` is the enclosed pixel area; anti-aliasing and JPEG ringing leave 1-3 pixel speckles that are genuine components, so drop `area < ~20` — but keep the floor **low**, because real small figures exist and each one dropped costs count. *Morphological close:* `cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)` is dilate-then-erode, bridging 1-2 pixel gaps where thresholding broke a thin stroke and split one figure into three components. A $3 \times 3$ kernel repairs hairline breaks; a $9 \times 9$ kernel fuses genuinely separate neighbours and *undercounts*. Small kernel, always.

**The alternative API.** `cv2.connectedComponentsWithStats(mask, connectivity=8)` returns `(n_labels, labels, stats, centroids)`. `n_labels - 1` is the component count (label 0 is background), and `stats[:, cv2.CC_STAT_AREA]` gives areas for filtering in one vectorised pass. It has **no hierarchy**, so a hollow ring is one component and its hole is simply not a component — it behaves like `RETR_EXTERNAL` for free. Use it when you only need counts; use contours when you also need shape.

## [problem-first]

Open [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md). The Input section contains the answer:

> "Figures do not overlap in a way that merges their borders. Each figure is a single connected component when considering edge pixels."

That is a contract. It says: the map from figures to components is one-to-one, so `count(components) == N` exactly. There is nothing left to model. The Tips section confirms the intended route — "edge detection, contour finding, and shape classification" — and explicitly says you "only need to count the figures, not classify them", so T29's descriptors are optional here.

What the guarantee still leaves you to get right:

1. A hollow figure is one component but **two** contours → `RETR_EXTERNAL`, not `RETR_LIST`.
2. The guarantee is about *edge pixels*, so the mask must hold the strokes and nothing else — T27's per-image background mode.
3. It assumes your mask did not break the stroke; a tight tolerance splits one figure into several components → close with a $3 \times 3$ kernel.
4. Relative error makes an off-by-three cost $3/100$ on a low-$N$ image and $3/500$ on a high-$N$ one — validate on the low-count training images.
5. 8000 images at $2048 \times 2048$: read grayscale, one pass, no Python pixel loops.

## [code-first]

```python
import cv2
import numpy as np

gray = cv2.imread("blueprint.png", cv2.IMREAD_GRAYSCALE)
bg = int(np.bincount(gray.ravel()).argmax())          # T27: background is the mode
mask = (np.abs(gray.astype(int) - bg) > 20).astype(np.uint8) * 255

# Repair hairline breaks in thin strokes. 3x3 only: a big kernel fuses distinct figures.
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

# Route A - contours. RETR_EXTERNAL: a hollow border yields an outer AND an inner
# contour, and RETR_LIST would return both, roughly doubling the count.
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count_a = sum(1 for c in cnts if cv2.contourArea(c) >= 20)   # kill anti-aliasing speckles

# Route B - connected components. No hierarchy at all, so holes never count.
n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
areas = stats[1:, cv2.CC_STAT_AREA]                   # label 0 is background
count_b = int((areas >= 20).sum())

# Diagnostics that catch every mode/connectivity mistake before you submit.
flat, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
n4 = cv2.connectedComponentsWithStats(mask, connectivity=4)[0] - 1
print("external", count_a, "| components", count_b, "| LIST", len(flat), "| 4-conn", n4)
# Expect: count_a == count_b, LIST near 2x (hollow figures), 4-conn inflated by diagonals.
```

Tune `TOL` and the area floor against `train.csv`, which carries the true counts — that is what it is for.

## [drill]

1. A figure is an unfilled square drawn with a 3-pixel stroke. How many contours under `RETR_EXTERNAL`, `RETR_LIST`, `RETR_TREE`?
2. Your count is 612 where truth is 310. Name the most likely single cause.
3. Your count is 4180 where truth is 310. Name the most likely single cause.
4. `connectedComponentsWithStats` returns `n_labels = 301`. How many figures?
5. Why does a $9 \times 9$ closing kernel lower your score even though it fixes broken strokes?
6. $\hat{y} = 2y$ on task 6 — compute Error Rate, Accuracy, and the final verdict.

<details><summary>Answers</summary>

1. 1, 2, 2 — `RETR_TREE` returns the same two boundaries as `RETR_LIST` but labels the inner one as a child, so you can filter to depth 0.
2. `RETR_LIST` instead of `RETR_EXTERNAL`: every hollow border counted twice.
3. 4-connectivity on diagonal strokes — rotated edges fragment into hundreds of one-pixel components. (A too-tight tolerance breaking strokes does the same, less dramatically.)
4. 300. Label 0 is the background, so subtract one.
5. It dilates by 4 pixels before eroding, so two nearby but distinct figures merge into one component and you undercount.
6. Error Rate $= \lvert 2y - y \rvert / y = 1$. Accuracy $= 1 - \min(1, 1) = 0$. That is below the 0.55 floor, so the verdict is 0.

</details>

**Rep:** write `count_figures(path)` returning one integer, run it over every training image, and plot predicted against true count. Report mean $\lvert \hat{y} - y \rvert / y$ separately for images with $y < 150$ and $y > 400$ — the relative metric makes the first group the one that decides your score.

## Traps & 60-second recall

- `RETR_EXTERNAL` for counting. `RETR_LIST` roughly doubles the count on hollow shapes and is the one mistake that zeroes the task.
- 8-connectivity, because diagonal strokes are not 4-connected.
- `connectedComponentsWithStats` minus 1 for the background label; it ignores holes for free.
- Close with a $3 \times 3$ kernel to repair breaks; bigger kernels merge distinct figures.
- Filter contours by area, but keep the floor low — genuine small figures exist.
- `findContours` wants single-channel `uint8` with non-zero foreground.
- Cross-check `RETR_EXTERNAL` against `connectedComponentsWithStats`; disagreement means your mask is wrong.
- The statement's connected-component guarantee is why no ML is needed at all.
