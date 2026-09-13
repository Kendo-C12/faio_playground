# T27 · Image basics: arrays, colour spaces, thresholding — Day 4

**Anchor task(s):**
- [`faio-2025/qualification/task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md)

Rating 1 · **exam-probability rank 1** · ~45 min

Day 4 is vision, embeddings and signals. It starts here because every later step — contours, descriptors, CNN input — is an operation on the array this lesson builds.

## [concept-first]

**An image is a numpy array.** `cv2.imread("x.png")` returns shape `(H, W, 3)`, dtype `uint8`, values `0..255`. For task 6 that is `(2048, 2048, 3)` — 4.2 M pixels, 12.6 MB per image, and 8000 test images.

**Channel order is BGR, not RGB.** `cv2.imread` gives blue first; `PIL.Image.open` and `matplotlib.pyplot.imshow` assume RGB. Mix them and your red triangles render blue. Convert explicitly with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`. For *counting* it does not matter — channel order cannot change a pixel's grayscale value — but it matters the moment you debug visually or feed a pretrained network.

**dtype discipline.** `uint8` wraps around: `np.uint8(250) + np.uint8(10)` is `4`, not `260`. Any arithmetic that can leave `0..255` must go through `.astype(int)` or `.astype(np.float32)` first. This is the single most common silent bug in hand-written CV code.

**Grayscale.** `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` collapses 3 channels to 1 with a luminance-weighted sum:

$$
Y \approx 0.299 R + 0.587 G + 0.114 B
$$

It is not the mean. Consequence: a saturated green and a saturated blue of the same "brightness" to the eye map to very different gray values — and a coloured stroke can accidentally land on the background's gray value.

**Histogram.** `np.bincount(gray.ravel(), minlength=256)` gives how many pixels sit at each of the 256 levels. On a blueprint it is extremely lopsided: one enormous spike at the background, thin bumps at the stroke colours. That spike is information, not noise.

**Thresholding** turns gray into a binary mask.

- *Global:* one cut value for the whole image — `cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)`.
- *Otsu:* picks $T$ automatically by maximising the between-class variance $\sigma_b^2(T)$ — `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`. Assumes a roughly **bimodal** histogram.
- *Adaptive:* a different $T$ per neighbourhood — `cv2.adaptiveThreshold(...)`. For uneven lighting (a photo), not for a synthetic blueprint.

Worked example: a blueprint with background 240, a pale gray stroke at 215 and a dark stroke at 30. Otsu sees the mass at 240 versus the mass near 30 and puts $T$ around 130 — which swallows the 215 stroke into "background" and loses every pale figure. Task 6's relative-error metric then punishes you hardest on the low-$N$ images.

## [problem-first]

Open [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md). Read the one sentence that decides this lesson: figures "may have transparent or solid-colored interiors. Background is distinguishable from figure borders."

Derive what the statement does and does not give you:

1. It promises the background is *distinguishable*. It never says the background is white, nor that there are only two brightness levels. So a hard-coded `gray < 128` and a blind Otsu are both unjustified guesses.
2. $100 \le N \le 500$ strokes on $2048 \times 2048$ means the background is overwhelmingly the **most frequent** pixel value in every image. That is a far stronger, and checkable, assumption than bimodality.
3. So derive it per image: `bg = np.bincount(gray.ravel()).argmax()`, then call a pixel "stroke" when it differs from `bg` by more than a tolerance. This is exactly the move in [`solutions/qualification/task6-simple-objects.md`](../solutions/qualification/task6-simple-objects.md).
4. Tolerance is your one tuning knob. Too small and anti-aliased stroke edges fragment; too large and pale strokes vanish. `train.csv` ships ground-truth counts — tune it there, not by eye.
5. "Transparent interiors" means a hollow ring of stroke pixels. Keep that in mind; T28 is about what that does to the count.

The Tips section is also the only place in the whole round that states tooling: "You may use any programming language and libraries (OpenCV, scikit-image, PIL, etc.)". Nothing else in the round says that, so do not generalise it to the other tasks.

## [code-first]

```python
import cv2
import numpy as np

img = cv2.imread("blueprint.png")          # (2048, 2048, 3), uint8, channel order BGR
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # (2048, 2048), luminance-weighted, not a mean

hist = np.bincount(gray.ravel(), minlength=256)
bg = int(hist.argmax())                    # the background IS the mode: 100-500 strokes cannot outvote it
print("background level:", bg, "share:", hist[bg] / gray.size)   # expect > 0.9

# astype(int) first: uint8 subtraction wraps around and |a-b| becomes nonsense.
TOL = 20                                   # the one constant to tune on train.csv
mask = (np.abs(gray.astype(int) - bg) > TOL).astype(np.uint8) * 255

# Compare against the two alternatives before trusting either of them.
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, blockSize=31, C=5)
for name, m in [("mode", mask), ("otsu", otsu), ("adaptive", adapt)]:
    print(name, "stroke pixel fraction:", (m > 0).mean())    # should agree to within a few %

# 8000 images: read grayscale directly, skip the colour array entirely.
gray_fast = cv2.imread("blueprint.png", cv2.IMREAD_GRAYSCALE)
```

## [drill]

1. `cv2.imread` on a $2048 \times 2048$ PNG — give shape, dtype and channel order.
2. Why is `gray.astype(int)` needed before `np.abs(gray - bg)`?
3. When does Otsu fail on a task 6 blueprint, and what replaces it?
4. Your mask has a stroke pixel fraction of 0.97. What happened?
5. The background is pure white but the image was saved as JPEG. What appears in the histogram near 255, and what must `TOL` absorb?
6. Name one reason BGR-vs-RGB cannot change your task 6 count, and one reason you still convert.

<details><summary>Answers</summary>

1. `(2048, 2048, 3)`, `uint8`, **BGR**.
2. `uint8` arithmetic wraps modulo 256, so `10 - 20` becomes `246` and the comparison silently inverts.
3. When more than two brightness populations exist — a mid-gray stroke sits on the wrong side of a single cut. Replace it with the per-image mode, `np.bincount(gray.ravel()).argmax()`.
4. The polarity is flipped: you selected the background, not the strokes. Expect a few percent, not most of the image.
5. A spread of values just below 255 from JPEG ringing around every edge. `TOL` must sit above that spread, or each stroke grows a halo of false positives.
6. Grayscale conversion is a fixed weighted sum either way, so the *mask* is unchanged up to which weight hits which channel — the count is identical. You still convert for correct visual debugging and for any pretrained model, which expects RGB.

</details>

**Rep:** take one training blueprint, print the top 5 histogram peaks with their pixel shares, and confirm the mode accounts for > 90 % of pixels. Then sweep `TOL` over `10, 20, 40, 60` and record the stroke pixel fraction for each.

## Traps & 60-second recall

- `cv2.imread` is BGR; PIL and matplotlib are RGB.
- Cast out of `uint8` before any subtraction or sum.
- The background is the histogram **mode**, derived per image — never assumed white.
- Otsu needs a bimodal histogram; a blueprint with pale and dark strokes is not bimodal.
- Adaptive thresholding is for uneven illumination; synthetic images do not need it.
- Read with `IMREAD_GRAYSCALE` when you never need colour — $8000 \times 2048^2$ adds up.
- Tune the tolerance against `train.csv` counts, because that file is the only ground truth you get.
