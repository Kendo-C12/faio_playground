# T06 · Custom and competition metrics — Day 3

**Anchor task(s):**
- [`faio-2025/qualification/task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md)
- [`faio-2025/day2/Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md)
- [`faio-2025/day2/HearMe_Personalized_Music_Recommender.md`](../faio-2025/day2/HearMe_Personalized_Music_Recommender.md)

Rating 2 · **exam-probability rank 1** · ~40 min

The metric is part of the problem statement, not an afterthought. Three of the archive's tasks define their own, and each one rewards a different behaviour.

## [concept-first]

**How to read a metric definition.** Four questions, in order:

1. **What is the unit of scoring?** One row, one user, one image, one query? `Lost_in_the_Museum.md` scores per *query* (1,000 of them) even though you submit 20,000 rows.
2. **Is the error relative or absolute?** Absolute error treats "off by 5" the same everywhere. Relative error divides by the truth, so being off by 5 on a count of 100 costs five times what it costs on a count of 500.
3. **Is there a floor, cap or threshold?** A $\min(1, \cdot)$ caps per-example damage; a rescaling floor can turn a mediocre score into exactly zero.
4. **How are examples combined?** Mean, sum, or mean-of-per-group-sums. This decides whether a rare class or a quiet user matters.

**Relative-error accuracy**, from [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md):

$$
\text{Error Rate} = \frac{\lvert \hat{y} - y \rvert}{y}
$$

$$
\text{Accuracy} = 1 - \min(1, \text{Error Rate})
$$

$$
\text{Final} = \frac{\text{Accuracy} - 0.55}{1 - 0.55}
$$

The $\min(1, \cdot)$ means a wild overshoot costs exactly the same as predicting zero — you cannot go negative on one image. The division by $y$ means the task is *proportional* counting: on a blueprint with 100 figures you may miss 2 for the same cost as missing 10 on one with 500. And the 0.55 floor rescales, so 0.55 accuracy maps to 0 and 1.0 maps to 1 — a solution averaging 0.50 scores nothing at all.

**The archive contradicts itself here.** The printed verdict rule reads $0$, if $\text{Accuracy} > 0.55$ with the rescaling in the `otherwise` branch. Taken literally that awards 0 for *good* accuracy and a negative number for bad — backwards. The intended reading is plainly the other way round: **0 when accuracy is below 0.55**, rescaled above it. Assume the sane version, and clear 0.55 by a wide margin so the ambiguity cannot hurt you.

**Ranking metrics.** [`Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md) uses **Hit@3**: for each of 1,000 queries, rank all 19,000 gallery and distractor embeddings by cosine similarity, and score 1 if the true match is in the top 3, else 0:

$$
\text{Hit@3} = \frac{\text{hits}}{1000}
$$

Two properties follow. It is **binary per query** — rank 1 and rank 3 score identically, rank 4 scores nothing, so there is no reward for polishing an already-correct retrieval. And you submit **embeddings, not predictions**; the ranking happens server-side. Cosine similarity is scale-invariant, so L2-normalising your vectors changes nothing mathematically, but the statement says normalised submissions are preferred — do it anyway, it costs one line and removes a numerical risk.

**Graded, summed metrics.** [`HearMe_Personalized_Music_Recommender.md`](../faio-2025/day2/HearMe_Personalized_Music_Recommender.md) gives each of 50 ranked tracks per user a listened fraction from $\{0,\ 0.25,\ 0.5,\ 0.75,\ 1.0\}$, sums them per user (0–50), averages over users, then normalises to 0–1. So: partial credit exists, a quarter-listen is a real quarter-point, replays do not double-count, and **rank position is not weighted at all** inside the 50 — rank 1 and rank 50 are worth the same. That makes it a *set* selection problem, not an ordering problem, and it rewards predicting listening *depth* rather than mere clicks.

**Macro-F1**, task 4's real metric, is covered as formulas in [`day3-t05-classification-metrics-formula-only.md`](./day3-t05-classification-metrics-formula-only.md). The point here is only that it was never stated in [`task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md) — it had to be inferred from the solution notebook's `classification_report` and its `scoring="f1_macro"` grid search. When the statement is silent, look at whatever reference code ships with it.

## [problem-first]

Open [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md) and derive strategy from the metric alone, before touching an image.

1. Error is relative, so **large-N images are forgiving and small-N images are brutal**. An absolute-error instinct would make you tune on the wrong images. Validate stratified by true count.
2. $\min(1, \cdot)$ caps the damage, so a catastrophic failure on one image is bounded. Never skip an image — output a plausible number. Predicting the training mean beats predicting nothing.
3. A systematic bias costs on every image: a uniform +5% overcount scores $(0.95 - 0.55)/0.45 = 0.8889$, and it is correctable for free. Measure `mean(pred/true)` on the training images and divide it out before submitting — the cheapest point in the task.
4. The 0.55 floor is a cliff. Anything scoring below it is worth exactly as much as an empty submission, so a careless solution gets nothing.
5. It is also the only task in the archive stating tooling rules: "You may use any programming language and libraries (OpenCV, scikit-image, PIL, etc.)."

## [code-first]

```python
import numpy as np

# Re-implement the metric BEFORE modelling. This is the habit; the rest is detail.
def task6_score(y_true, y_pred, floor=0.55):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = np.abs(y_pred - y_true) / y_true          # RELATIVE, divided by truth
    acc = 1.0 - np.minimum(1.0, err)                # capped at one per image
    acc = acc.mean()
    return max(0.0, (acc - floor) / (1.0 - floor))  # the sane reading of the floor

def hit_at_k(q_emb, g_emb, true_idx, k=3):
    # Lost in the Museum: cosine on L2-normalised vectors is just a dot product.
    q = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    g = g_emb / np.linalg.norm(g_emb, axis=1, keepdims=True)
    top = np.argsort(-(q @ g.T), axis=1)[:, :k]
    return float(np.mean([t in row for t, row in zip(true_idx, top)]))

def hearme_score(recs, listened, n_items=50):
    # recs: {user: [item, ...]}  listened: {(user, item): fraction in 0..1}
    per_user = [sum(listened.get((u, i), 0.0) for i in items[:n_items])
                for u, items in recs.items()]
    return float(np.mean(per_user)) / n_items       # 0..50 -> 0..1

# Sanity-check every metric against a degenerate baseline before trusting it.
y = np.array([100, 250, 500])
print(task6_score(y, y))                            # 1.0, a perfect solution
print(task6_score(y, np.full(3, y.mean())))         # constant-mean baseline
print(task6_score(y, y * 1.05))                     # 5% systematic overcount -> 0.8889
```

## [drill]

1. Task 6: true count 120, you predict 132. What is the per-image accuracy?
2. Same task: true 500, predicted 0. What is the per-image accuracy, and why not negative?
3. Your mean task 6 accuracy is 0.52. What is the final verdict?
4. Hit@3: you move a correct match from rank 3 to rank 1 for 200 queries. How much does the score change?
5. HearMe: you reorder a user's 50 tracks from best-last to best-first. How much does the score change?
6. Task 4's metric is not in the statement. Where did macro-F1 come from, and why does it matter that it is not accuracy?

<details><summary>Answers</summary>

1. $\lvert 132 - 120 \rvert / 120 = 0.1$, so accuracy $0.9$.
2. Error rate $1.0$, accuracy $1 - \min(1, 1) = 0$. The $\min(1, \cdot)$ floors each image at zero, so one disaster cannot drag the mean below the others.
3. $0$ — below the 0.55 floor, identical in value to submitting nothing.
4. Not at all. Hit@3 is binary per query; anything inside the top 3 already counts as a hit.
5. Not at all. The metric sums listened fractions over the 50 recommendations with no positional weighting — only *which* 50 matters.
6. From the solution notebook: it prints `classification_report` and its grid search used `scoring="f1_macro"`. It matters because macro-F1 weights all three languages equally, so a small class (likely `kaz` or `eng`) can sink the score while accuracy still looks fine.

</details>

**Rep:** implement `task6_score` and test it on synthetic counts drawn uniformly from 100–500 under three error models — constant bias +5%, multiplicative noise $\pm 10\%$, and one catastrophic miss in fifty — and rank which error model costs most.

## Traps & 60-second recall

- Re-implement the metric locally before you model. Nothing else on this list matters if you skip it.
- Relative error means small-N examples dominate your loss. Stratify validation by the target's magnitude.
- $\min(1, \cdot)$ bounds per-example damage, so always emit a guess, never a blank.
- A rescaling floor is a cliff: below it, good-but-not-good-enough equals nothing.
- Hit@$k$ is binary per query — no credit for improving an already-correct rank.
- When the submission is embeddings, the metric runs server-side: L2-normalise and validate row count and order.
- When the statement omits the metric, read the shipped reference code for it.
- `task6_Simple_Objects.md`'s printed verdict rule is inverted; assume 0 *below* 0.55 and clear it comfortably.
