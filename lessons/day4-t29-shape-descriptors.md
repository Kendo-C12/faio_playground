# T29 · Shape descriptors: vertices, circularity, moments — Day 4

**Anchor task(s):**
- [`faio-2025/qualification/task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md)
- [`faio-2025/day1/qaz_letters.md`](../faio-2025/day1/qaz_letters.md)

Rating 0 · **exam-probability rank 1** · ~45 min

T28 turned an image into a list of contours. This turns each contour into numbers. Sometimes those numbers are the answer; sometimes, as in task 6, they are a diagnostic you do not have to submit.

## [concept-first]

Every descriptor below is computed from one contour — the point array `cv2.findContours` returned.

**Vertex count.** `cv2.approxPolyDP(c, eps, True)` replaces the contour with a polygon whose vertices deviate from it by at most `eps` pixels (Ramer–Douglas–Peucker). Set `eps` as a *fraction of perimeter*, not an absolute: `eps = 0.02 * cv2.arcLength(c, True)`. Then `len(approx)` is 3 for a triangle, 4 for a rectangle, and ≥ 8 for a circle. Scale-free because `eps` scales with the shape.

`eps` too small → a circle keeps 40 vertices and an anti-aliased triangle reports 5. `eps` too large → a rectangle collapses to a triangle. `0.02`–`0.04` is the usable band; verify on known shapes.

**Circularity** is `4 * pi * A / P^2`, with `A = cv2.contourArea(c)` and `P = cv2.arcLength(c, True)`. A perfect circle gives 1.0, a square ≈ 0.785, an equilateral triangle ≈ 0.605, a long thin sliver → 0. It is the cleanest single number for "circle or not", invariant to translation, rotation and scale — but **not** robust to a jagged boundary, since noise inflates `P` while leaving `A` alone. Smooth the mask before trusting it.

**Aspect ratio and extent.** `x, y, w, h = cv2.boundingRect(c)` is the axis-aligned box; `aspect = w / h`, `extent = A / (w * h)`. Extent ≈ 1 for an axis-aligned rectangle, ≈ 0.785 for a circle, ≈ 0.5 for a triangle. Both are rotation-*dependent*: a square turned 45° has extent 0.5, indistinguishable from a triangle. `cv2.minAreaRect(c)` gives the rotated box and fixes it — extent against *its* area is ≈ 1 for a rectangle at any angle.

**Solidity.** `A / cv2.contourArea(cv2.convexHull(c))`. Convex shapes — all three of task 6's — give ≈ 1, so it separates convex from concave: useless for classifying here, valuable for spotting merges and for letters.

**Image moments.** `M = cv2.moments(c)` gives raw moments `m_pq = Σ x^p y^q`: area `m00`, centroid `(m10/m00, m01/m00)`. Subtract the centroid → *central* moments `mu_pq` (translation-invariant). Divide by a power of `m00` → *normalised* central moments `nu_pq` (+ scale-invariant).

**The 7 Hu moments.** `cv2.HuMoments(M)` combines the `nu_pq` into 7 values that are additionally **rotation**-invariant. So the invariance ladder is: raw → nothing; central → translation; normalised central → + scale; Hu → + rotation. The 7th also flips sign under reflection, so it detects mirroring. They span many orders of magnitude — always compare them as `sign(h) * log10(|h|)`.

Worked example — the whole task 6 shape taxonomy from two numbers: with `v = len(approxPolyDP(c, 0.02 * P, True))` and `circ = 4*pi*A/P^2`, `v == 3` is a triangle, `v == 4` a rectangle, and `circ > 0.8` a circle. Anything left over is a mask defect, not a fourth shape.

## [problem-first]

Open [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md). Its Tips name exactly this toolbox: "shape classification using properties like **number of corners, aspect ratio, or circularity**". And then it disarms it: "you only need to **count** the figures — not classify them by type (unless it helps your counting logic)".

Be honest about that parenthesis. Descriptors earn their place in task 6 only when counting logic needs them:

1. **Diagnosing a merge.** Two touching figures form one contour whose solidity drops well below 1, flagging a component you may want to count as two.
2. **Separating speckle from figure.** An area floor alone also deletes genuinely tiny circles; area *plus* circularity is the safer filter.
3. **Auditing your mask.** If the descriptors over a training image do not form three clean clusters (v=3, v=4, high circularity), your threshold is wrong — before the count is.

Now open [`qaz_letters.md`](../faio-2025/day1/qaz_letters.md). The situation inverts: the jury hands you a tabular dataset where the visual information of each of 42 Kazakh letters "has been distilled into handcrafted numerical features" — Hu moments ("7 invariant moments describing shape, rotation, scale invariant"), contour counts and areas via OpenCV, radial distances from the centroid, Fourier descriptors of contours, symmetry measures, horizontal/vertical projections, 4×4 zoning, DCT, wavelets. No images at all.

So the lesson is dual-use: in task 6 you *compute* descriptors and may discard them; in `qaz_letters` the descriptors **are** the dataset, and the job is reading the column list to know what each can and cannot distinguish. Sharpest consequence: rotation-invariant features cannot separate letters differing only by orientation, and scale-invariant ones cannot separate letters differing only in size — so for the nine Kazakh letters distinguished by a diacritic, the *non*-invariant zoning and projection features carry the signal.

## [code-first]

```python
import collections, cv2, numpy as np

def describe(c):
    A = cv2.contourArea(c)
    P = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    # eps as a FRACTION of perimeter -> vertex count is scale-free.
    v = len(cv2.approxPolyDP(c, 0.02 * P, True))
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    return {
        "area": A,
        "vertices": v,
        "circularity": 4 * np.pi * A / (P * P + 1e-9),   # 1.0 circle, .785 square, .605 triangle
        "aspect": w / (h + 1e-9),                        # rotation-DEPENDENT
        "extent": A / (w * h + 1e-9),                    # vs axis-aligned box
        "extent_rot": A / (cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1] + 1e-9),
        "solidity": A / (cv2.contourArea(hull) + 1e-9),  # ~1 for convex; flags merged figures
        # Hu spans many decades: log-scale it or the model sees only h[0].
        **{f"hu{i}": np.sign(v_) * np.log10(abs(v_) + 1e-30) for i, v_ in enumerate(hu)},
    }

def classify(d):
    if d["vertices"] == 3:            return "triangle"
    if d["vertices"] == 4:            return "rectangle"
    if d["circularity"] > 0.80:       return "circle"
    return "unknown"                  # a non-empty 'unknown' bucket means the mask needs work

gray = cv2.imread("blueprint.png", cv2.IMREAD_GRAYSCALE)
bg = int(np.bincount(gray.ravel()).argmax())
mask = (np.abs(gray.astype(int) - bg) > 20).astype(np.uint8) * 255
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ds = [describe(c) for c in cnts if cv2.contourArea(c) >= 20]
print(len(ds), "figures")                           # this alone is the task 6 answer
print(collections.Counter(classify(d) for d in ds))  # audit only; 'unknown' must be empty
print("merged suspects:", sum(d["solidity"] < 0.9 for d in ds))   # should be ~0
```

## [drill]

1. Give circularity for a perfect circle, a square and an equilateral triangle.
2. `approxPolyDP` reports 6 vertices for a clean rectangle. What is wrong and what do you change?
3. List the four levels of moment invariance and what each one adds.
4. Why does `qaz_letters` include zoning and projection features when it already has Hu moments?
5. One contour has solidity 0.62 in a task 6 blueprint. What probably happened?
6. Does task 6 require shape classification? Quote the governing clause.

<details><summary>Answers</summary>

1. 1.0, ≈ 0.785 (`pi/4`), ≈ 0.605.
2. `eps` is too small, so anti-aliasing steps survive as corners. Raise it toward `0.03`–`0.04 * arcLength`, or smooth/close the mask first.
3. Raw moments: no invariance. Central (centroid-subtracted): translation. Normalised central: + scale. Hu: + rotation, and `hu[6]` changes sign under reflection.
4. Hu moments are rotation- and scale-invariant, which *destroys* exactly the information distinguishing letters that differ by orientation, size or a small diacritic. Zoning and projections are position-sensitive and keep it.
5. Two figures touched and became one contour, so the shape is concave. It is a merge — the one case where the statement's one-component-per-figure guarantee fails.
6. No. "Remember: you only need to **count** the figures — not classify them by type (unless it helps your counting logic)."

</details>

**Rep:** run `describe` over one training blueprint, print the distribution of `vertices` and a histogram of `circularity`, and confirm you see three clusters. Then check that `len(ds)` matches that image's `train.csv` count, and that `classify` leaves no `unknown`.

## Traps & 60-second recall

- `approxPolyDP` eps must be a fraction of perimeter, never a fixed pixel count.
- `circularity = 4*pi*A/P^2`: 1.0 circle, 0.785 square, 0.605 triangle.
- Aspect ratio and axis-aligned extent are rotation-dependent; `minAreaRect` fixes both.
- Solidity ≈ 1 for all of task 6's shapes, so a low value means a merge, not a new shape.
- Log-scale Hu moments before feeding any model; invariance is a cost, not a virtue, so never use a rotation-invariant feature to separate rotations.
- Task 6 never needs classification; compute descriptors as a sanity check on the count, not as the deliverable.
- When the jury already hands you the features (`qaz_letters`), this lesson is for *reading* the column list, not recomputing it.
