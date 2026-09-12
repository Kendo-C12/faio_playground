# T33 · Reading the metric for cheap points — Day 5

**Anchor task(s):**
- [`faio-2025/qualification/task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md)
- [`faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)
- [`faio-2025/day2/HearMe_Personalized_Music_Recommender.md`](../faio-2025/day2/HearMe_Personalized_Music_Recommender.md)
- [`faio-2025/day2/Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md)

Rating 0 · **exam-probability rank 1** · ~35 min

The metric tells you which errors cost money. Read it before you model, and half of your tuning decisions are made for you.

## [concept-first]

**Re-implement the metric locally, first.** Before any model, write the scoring function in five lines and run it on a guess. You then have a number that moves, and you learn the metric's shape by poking it. Every decision below comes out of that function, not out of intuition.

**Ask what the metric is insensitive to.** Insensitivity is permission to stop working.

- [`Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md) scores Hit@3 — "a hit is recorded for a query if its true matching HQ painting appears within the retrieved Top-3". Order inside the top 3 is invisible; everything below rank 3 is equally worthless. So margin optimisation is wasted effort and only boundary queries matter.
- It also says embeddings "are expected to be L2-normalized for optimal performance, though non-normalized submissions will still be accepted" — cosine is scale-invariant, so vector magnitude is free information the metric throws away.
- [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md) asks only for a count: "you only need to **count** the figures — not classify them by type". Shape classification is optional work the metric does not pay for.

**Relative error means small-N dominates.** Task 6's metric:

```
Error Rate = |ŷ − y| / y
Accuracy   = 1 − min(1, Error Rate)
```

The denominator is the true count, which ranges over `100 ≤ N ≤ 500`. Miss by 10 figures at `y = 500` and you lose 0.02. Miss by the same 10 at `y = 100` and you lose 0.10 — five times worse for identical absolute performance. **Therefore: validate on the low-count images.** Your tuning set should be skewed toward small `y`, not a uniform sample, because that is where the loss lives. The same arithmetic says systematic bias is worse than noise: a `+5` offset on every image costs you on every low-count image, while random `±5` partially averages out across 8000 of them.

**A floor means careless scores exactly zero.** Task 6 rescales: `(Accuracy − 0.55) / (1 − 0.55)`, with 0 below the floor. Accuracy 0.54 and accuracy 0.01 are worth the same — nothing. So the first job is *clearing the floor*, which needs relative error under 0.45 — i.e. counts within roughly ±45% of truth. A plain contour count clears that easily; the common way to fail it is a structural bug, such as using `RETR_LIST` and double-counting every hollow shape. Check the floor before you tune anything above it.

**Macro-F1 means a small class can sink you.** The shipped reference notebook for task 4, [`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb), scores with `classification_report` and its grid search tunes on `scoring="f1_macro"` — not accuracy. Macro-F1 averages per-class F1 *unweighted*, so a class with 5% of the rows carries the same third of the score as a class with 60%. Under accuracy you would ignore it; under macro-F1, collapsing the rare class costs you 0.33. Practical consequences: look at the per-class recall, not the aggregate; consider `class_weight="balanced"`; and remember that augmenting the weak class (the notebook's `kazakh_to_russian` and `translit` tricks) moves macro-F1 far more than it moves accuracy.

**Ranking metrics only care about the top-k ordering.** HearMe pays `Σ` of listened fractions over exactly 50 slots per user ([`HearMe_Personalized_Music_Recommender.md`](../faio-2025/day2/HearMe_Personalized_Music_Recommender.md)), normalised to 0–1. Nothing outside those 50 is scored, ranks beyond 50 do not exist, and the statement's "replays don't count extra" caps each slot at 1. So a model's calibration over the whole catalogue is irrelevant; only the identity of each user's top 50 matters.

**One boundary.** Where the archive has an obvious typo, solve the intended problem and note the ambiguity. Task 6 prints:

```
Final Verdict = 0,  if Accuracy > 0.55
                (Accuracy − 0.55) / (1 − 0.55),  otherwise
```

Read literally, that zeroes every good solution and rewards bad ones — it is inverted. The intended rule is plainly 0 when accuracy is *below* 0.55. The correct response is to maximise accuracy as intended and, if the contest has a clarification channel, ask. Writing a deliberately bad solution to exploit the printed text is not strategy; it fails the moment the grader implements the intent.

## [problem-first]

Open [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md) and derive your whole plan from the Evaluation section alone.

1. Metric is on the **count** → do not build a classifier. Connected components suffice, and the statement hands you the guarantee: "each figure is a single connected component when considering edge pixels."
2. Error is **relative to `y`** → build a validation split weighted toward `y` near 100 using `train.csv`'s ground-truth counts.
3. Error is **absolute difference**, not squared → a few large misses cost the same as many small ones; chase bias, not variance.
4. There is a **floor at 0.55** → first milestone is a submission provably above it on train, not a good one.
5. `min(1, ·)` **caps the penalty at 1** → a catastrophic image cannot go below 0 accuracy, so a failed read should fall back to a plausible constant (say the training median) rather than crash the batch.
6. **8000 images** → runtime is part of the score in practice, because a run you cannot finish produces no file at all.

Now the same exercise on [`Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md): "You must treat all images equally… You submit embeddings only", exactly 20,000 rows, `image_name` plus `feature_0 … feature_{D−1}`, D recommended 256 or 512, "no filtering, sorting, or skipping allowed". The metric never sees your model, so nothing about architecture is scored — only whether a query and its HQ painting come out near each other under cosine. That tells you to spend the round on *invariance to the domain gap* (blur, glare, crop) and nothing at all on a classifier head.

## [code-first]

```python
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, classification_report

# ---- task 6: the metric, verbatim from the statement, before any modelling ----
def faio_accuracy(y_hat, y):
    err = np.abs(np.asarray(y_hat, float) - np.asarray(y, float)) / np.asarray(y, float)
    return 1.0 - np.minimum(1.0, err)

def final_verdict(y_hat, y, floor=0.55):
    acc = faio_accuracy(y_hat, y)                 # per-image accuracy
    # intended reading: 0 BELOW the floor. The statement prints this inverted.
    return np.where(acc < floor, 0.0, (acc - floor) / (1 - floor)).mean()

train = pd.read_csv("train.csv")                  # ground-truth counts
# Relative error -> small y dominates. Score the low-count slice SEPARATELY.
low = train[train.count_col <= 150]
print("all :", final_verdict(pred_all, train.count_col))
print("lowN:", final_verdict(pred_low, low.count_col))     # the number that matters

# Bias probe: is the error systematic? Systematic is fixable and expensive.
resid = pred_all - train.count_col
print("median bias", np.median(resid), "| IQR", np.subtract(*np.percentile(resid,[75,25])))

# ---- task 4: macro-F1, not accuracy. Watch the weakest class. ----------------
print(classification_report(y_true, y_pred, digits=3))      # per-class recall
print("macro F1", f1_score(y_true, y_pred, average="macro"))
print("accuracy", (y_true == y_pred).mean())   # will flatter you; ignore it

# ---- Hit@3: an indicator, so only boundary queries are worth work ------------
def hit_at_3(Q, G, truth):                     # Q, G L2-normalised
    sims = G @ Q.T                             # (gallery, query)
    top3 = np.argsort(-sims, axis=0)[:3]
    return np.mean([truth[j] in top3[:, j] for j in range(Q.shape[0])])

# Where to spend time: queries whose true match sits at rank 4-10 right now.
```

## [drill]

1. Task 6, true count 120, you predict 132. What is the per-image accuracy, and is the verdict above the floor?
2. Same absolute error of 12 at a true count of 480. Accuracy?
3. Why does a constant `+4` overcount hurt more than a zero-mean `±4` noise on task 6?
4. Task 4 has `ru`, `kaz`, `eng`. Your model predicts `eng` perfectly, `ru` well, and never predicts `kaz`. What does accuracy say versus macro-F1?
5. Task 6 prints `0, if Accuracy > 0.55`. What do you do?
6. Under Hit@3, which queries deserve your remaining hour?
7. Name one piece of information the HearMe metric completely ignores.

<details><summary>Answers</summary>

1. Error rate `12/120 = 0.10`, accuracy `0.90`; verdict `(0.90−0.55)/0.45 = 0.778`. Above the floor.
2. `12/480 = 0.025`, accuracy `0.975` — a quarter of the loss for the identical absolute error. Hence test on small `y`.
3. Because it shifts every image's error in the same direction, so nothing cancels; with relative error the shift is most damaging precisely on the low-`y` images. Zero-mean noise leaves many images nearly exact.
4. Accuracy stays high if `kaz` is a small fraction of the rows; macro-F1 loses the entire `kaz` term, capping you near 0.67. The grid search in the reference notebook tunes on `f1_macro` for exactly this reason.
5. Treat it as the typo it is: maximise accuracy as intended (0 *below* 0.55), and flag the ambiguity to the organisers if a clarification channel exists. Do not engineer a bad solution to exploit the literal wording.
6. The ones whose true match currently sits just outside the top 3 — ranks 4 to roughly 10. Moving a rank-2 hit to rank 1 adds exactly zero.
7. Magnitude of your predicted scores (only the top-50 identity matters), anything ranked past 50, and replays beyond the first full listen.

</details>

**Rep:** implement `final_verdict` above, feed it the constant prediction "always 300" against a plausible spread of counts from 100 to 500, and find out whether a constant clears the 0.55 floor. That number decides how urgent task 6 really is.

## Traps & 60-second recall

- Re-implement the metric locally before you model. Five lines, then poke it.
- List what the metric ignores — that list is permission to stop working.
- Relative error ⇒ small denominators dominate ⇒ validate on the low-`N` slice.
- Absolute error ⇒ fight bias, not variance. Check the median residual.
- A floor turns "not great" into exactly zero: clear it first, polish second.
- Macro-F1 weights every class equally; read per-class recall and consider `class_weight="balanced"` or augmenting the weak class.
- Ranking metrics see only the top-k set; calibration elsewhere is free to be wrong.
- An obvious typo in a printed rule is solved as intended and flagged, never exploited.
