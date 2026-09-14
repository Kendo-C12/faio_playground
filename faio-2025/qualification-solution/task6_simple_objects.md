# Task 6 — Simple Objects · why this solution

Runnable notebook: [`task6_simple_objects.ipynb`](./task6_simple_objects.ipynb)
Statement: [`../qualification/task6_Simple_Objects.md`](../qualification/task6_Simple_Objects.md)

## What the task asks

Count the geometric figures in a 2048×2048 RGB blueprint. Each image holds between 100 and 500 triangles, rectangles and circles, drawn with solid coloured borders. You are tested on **8000 images** and submit `submit.csv`.

## Why there is no machine learning here

The statement gives away the answer in its Input section:

> Figures do not overlap in a way that merges their borders. **Each figure is a single connected component when considering edge pixels.**

That sentence converts the problem from computer vision into counting. If every figure is exactly one connected blob of stroke pixels, then *the number of figures equals the number of connected components*. No training data is needed, no model is fitted, and the answer is exact rather than estimated.

Training a detector here would be slower, less accurate, and would ignore a guarantee the jury handed you. The Tips section confirms the intended route — "edge detection, contour finding, and shape classification using properties like number of corners, aspect ratio, or circularity" — and adds that you only need to **count**, not classify.

This is also the one task in the entire round that states its tooling rules: *"You may use any programming language and libraries (OpenCV, scikit-image, PIL, etc.)"*.

## The decision that doubles or halves your score

`cv2.findContours` takes a retrieval mode, and the choice is worth more than everything else in this notebook.

A figure drawn with a **stroke** has two outlines: the outside of the border and the inside of the border. `RETR_LIST` returns both, so every hollow shape counts twice and your answer is roughly double the truth. `RETR_EXTERNAL` returns only outermost outlines, which is exactly one per figure.

With a relative-error metric, doubling the count is catastrophic. If the truth is 300 and you report 600, the error rate is $|600-300|/300 = 1.0$, accuracy is `0`, and the score for that image is `0` — the same as submitting nothing.

## Why the background is measured, not assumed

The statement promises only that the background is *distinguishable*, never that it is white. It also says figures "may have transparent or solid-coloured interiors".

So the notebook finds the background per image as the most common grey level:

```python
bg = int(np.bincount(gray.ravel()).argmax())
mask = (np.abs(gray.astype(np.int16) - bg) > COLOR_TOL) * 255
```

Hard-coding `255` works until one test image has a coloured background, and then it returns nonsense for that image with no warning. Measuring costs one line.

The morphological close that follows repairs single-pixel gaps in a stroke. Without it, one broken border splits into two contours and inflates the count.

## Why the two constants are calibrated, not guessed

`COLOR_TOL` and `MIN_AREA` are the only free parameters, and both are genuinely uncertain: the right tolerance depends on the anti-aliasing in the images, and the right minimum area depends on how small the smallest real figure is.

The notebook grid-searches both against `train.csv`, which ships with true counts, and scores each combination with the competition's own metric rather than with mean error. That matters because the metric is not linear — see below.

Guessing these two numbers is the difference between a good score and a zero, and calibrating them takes a minute.

## Read the metric before optimising

$$
\text{Error Rate} = \frac{|\hat{y} - y|}{y}, \qquad \text{Accuracy} = 1 - \min(1, \text{Error Rate})
$$

Two properties change how you test:

- **The error is relative.** Being off by 5 costs `5/100 = 0.05` on an image with 100 figures, but only `5/500 = 0.01` on one with 500. **Test on the low-count images**, because that is where your score is decided.
- **There is a floor at 0.55.** Below it the verdict is exactly `0`. This is a cliff, not a slope: an image scoring 0.54 earns precisely what submitting nothing earns. Consistency across all 8000 images beats brilliance on some of them.

**The printed rule is inverted.** The statement says `Final Verdict = 0, if Accuracy > 0.55`, which would award zero to every good solution and reward bad ones. The intended reading is `0` when accuracy is *below* 0.55, rescaled above it. The notebook implements the intent. The correct response to an obvious typo in a statement is to solve the intended problem and flag the ambiguity — never to exploit the literal wording.

## Why runtime is estimated before the full run

8000 images at 2048×2048 is not free. The notebook times ten images and extrapolates before committing.

At 0.4 seconds per image, the full job takes 53 minutes — a fifth of the entire four-hour round. Knowing that number early lets you launch the job and work on another task while it runs, instead of discovering at 15:30 that it will not finish.

## The output format is genuinely ambiguous

The statement contradicts itself. The Output section asks for "a CSV file with a single integer in each row", while the very next sentence refers to "the order of `id` values in `submit.csv`" — and a file of bare integers has no `id` column. Its "Important" paragraph is also copy-pasted from task 5 and names `test.csv` and `solution.csv`, neither of which this task ships.

**Copy the shipped `submit.csv` template.** The notebook writes `id,count` and includes a commented single-column variant, so you can match whichever the template turns out to be.

## Where machine learning would help

If the connectivity guarantee fails on the real test images — touching figures, merged borders, heavy overlap — the contour count under-reports and no tuning of these two constants fixes it. At that point the fallback is to estimate figures from total stroke area divided by mean per-figure area, or to train a density-estimation model on the training counts. Check a handful of training images against their true counts before assuming the guarantee holds.
