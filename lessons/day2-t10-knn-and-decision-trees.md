# T10 · kNN and decision trees — Day 2

**Anchor task(s):**
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)
- [`faio-2025/day1/qaz_letters.md`](../faio-2025/day1/qaz_letters.md)

Rating 1 · **exam-probability rank 2** · ~45 min

External reference: mlcourse.ai topic 3, "Classification, decision trees and k nearest neighbours" (not fetched here — the site is blocked from this environment).

## [concept-first]

These two are the opposite ends of tabular modelling, and knowing which one your data wants is most of the skill.

**kNN — no model at all.** Training stores the data. To predict, find the $k$ closest training rows and take the majority label (or the mean, for regression). The decision boundary is whatever the data draws.

Distance metrics — Euclidean (`p=2`) and Manhattan (`p=1`):

$$
d_{\text{euclid}}(a, b) = \sqrt{\sum_i (a_i - b_i)^2} \qquad d_{\text{manhattan}}(a, b) = \sum_i |a_i - b_i|
$$

Minkowski with general `p`, and cosine $1 - \dfrac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$ when only direction matters (text, embeddings). `weights="distance"` lets nearer neighbours vote harder.

$k$ is the only real knob. $k = 1$ memorises — zero training error, high variance, one mislabelled neighbour flips a prediction. Large $k$ smooths toward the majority class. Pick $k$ by CV; odd $k$ avoids ties in binary problems.

**kNN must have scaled features.** Distance sums squared differences across columns, so a column in the thousands drowns a column in [0,1]. For task 5 features, `wx_max` is in deg/s (hundreds) while `ax_mean` is in g (around 1) — unscaled, the gyro columns alone decide every neighbour. `StandardScaler` first, always, and fit it on train only.

**Decision tree — a learned sequence of questions.** At each node, choose the feature and threshold whose split makes the children purest. Impurity of a node with class shares $p_k$ — Gini maxes at 0.5, entropy at 1 bit, for two balanced classes:

$$
\text{Gini} = 1 - \sum_k p_k^2 \qquad \text{Entropy} = -\sum_k p_k \log_2 p_k
$$

The split maximises the weighted impurity drop

$$
\Delta I = I(\text{parent}) - \sum_i \frac{n_i}{n} \, I(\text{child}_i)
$$

Gini and entropy almost never disagree on the final tree; Gini is cheaper and is the default.

Depth is the overfitting knob. An unrestricted tree grows until every leaf is pure — it fits training data perfectly and generalises badly. `max_depth`, `min_samples_leaf` and `min_samples_split` are the three that matter; tune `max_depth` first.

**Trees need no scaling, and no encoding of ordinal columns.** A split is "is $x < t$", a comparison that any monotone rescaling leaves unchanged. This single fact explains why trees and boosting are the default on raw engineered tabular features and kNN is not.

Worked example: a 42-class problem like [`qaz_letters.md`](../faio-2025/day1/qaz_letters.md) has Hu moments (values ~1e-3) alongside pixel-count features (values ~1e3). kNN on those raw columns is decided entirely by the pixel counts. A tree is indifferent.

## [problem-first]

Open [`task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md). Binary label, accuracy metric, tabular features after you aggregate per `id`. What does it force?

1. **Few examples, many engineered columns.** One row per repetition means a few hundred rows against dozens of aggregate features — kNN in dozens of dimensions is weak (distances concentrate), a shallow tree is readable and fast.
2. **Accuracy on a roughly balanced binary target** → no class-weight gymnastics; a depth-limited tree is a legitimate baseline you can fit in one second and sanity-check by printing it.
3. **Heterogeneous units (g vs deg/s vs a row count)** → if you try kNN, scaling is mandatory; with a tree, you can skip the whole scaling question.
4. **Grouped rows.** If you model per-timestep rows instead of per-`id` aggregates, kNN finds the neighbouring timestep of the *same* repetition and reports near-perfect CV. That is leakage, not skill — see T07.

Now [`qaz_letters.md`](../faio-2025/day1/qaz_letters.md): 42 classes, features already handcrafted by the jury. A single tree cannot carve 42 classes well (each leaf sees few examples of each letter), but it is the unit that T11 and T12 stack into ensembles. Fit one tree there to learn which features the splits pick, then move to the ensemble.

## [code-first]

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold

# X: one row per id (aggregated task 5 features), y: 0/1
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# kNN: scaler INSIDE the pipeline, so each fold fits it on train rows only.
for k in (1, 3, 5, 9, 15, 25):
    knn = make_pipeline(StandardScaler(),
                        KNeighborsClassifier(n_neighbors=k, weights="distance"))
    s = cross_val_score(knn, X, y, cv=cv, scoring="accuracy")
    print(f"kNN k={k:2d}  {s.mean():.4f} +/- {s.std():.4f}")

# The same kNN without the scaler, to see how much scaling is worth here:
print("unscaled", cross_val_score(KNeighborsClassifier(5), X, y, cv=cv).mean())

# Tree: no scaler needed. Sweep depth — the overfitting knob.
for d in (2, 3, 5, 8, None):
    tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5,
                                  criterion="gini", random_state=42)
    s = cross_val_score(tree, X, y, cv=cv, scoring="accuracy")
    print(f"tree depth={str(d):4s}  {s.mean():.4f} +/- {s.std():.4f}")

# A depth-3 tree is small enough to read. Read it: it tells you which
# engineered features carry signal, which is input to T16 and T12.
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
print(export_text(tree, feature_names=list(X.columns)))
```

## [drill]

1. Which of kNN and a decision tree needs `StandardScaler`, and what in the algorithm makes the difference?
2. $k = 1$ gives training accuracy 1.000. What is that number worth?
3. Gini of a node with 30 positives and 10 negatives?
4. You raise `max_depth` and CV accuracy falls while training accuracy rises. Name the effect and the fix.
5. For task 5, why is per-timestep kNN CV accuracy not trustworthy?
6. Why does `export_text` on a depth-3 tree help you even when you intend to ship boosting?

<details><summary>Answers</summary>

1. kNN. It sums squared differences across columns, so a large-scale column dominates the distance; a tree only compares one feature to a threshold, which rescaling does not change.
2. Nothing — the nearest neighbour of a training point is itself. Only CV or a holdout score means anything for kNN.
3. $p = 0.75$, so $1 - (0.75^2 + 0.25^2) = 1 - (0.5625 + 0.0625) = 0.375$.
4. Overfitting. Cap `max_depth`, raise `min_samples_leaf`, or move to an ensemble (T11/T12).
5. Neighbouring timesteps of the same repetition are nearly identical, so the model retrieves the same `id` from the training fold. Aggregate per `id`, or use `GroupKFold` on `id`.
6. It names the features the splits actually use, which tells you which engineered columns earn their place and which to drop before the expensive model.

</details>

**Rep:** on the aggregated task 5 features, sweep $k$ and `max_depth` as above and write down the best CV accuracy for each family. Carry both numbers into T11 and T12 as the baseline to beat.

## Traps & 60-second recall

- kNN: scale, always, and inside a `Pipeline` so CV cannot leak the scaler.
- kNN has no training score worth reading; trust CV only.
- Odd $k$ for binary targets; tune $k$ by CV, never by the training set.
- Trees: ignore scaling, care enormously about `max_depth` / `min_samples_leaf`.
- Gini max 0.5, entropy max 1 bit, for two balanced classes.
- A single tree is a baseline and a diagnostic, not the final model — it is the building block for T11 and T12.
- High-dimensional engineered features hurt kNN (distances concentrate) and leave trees unmoved.
