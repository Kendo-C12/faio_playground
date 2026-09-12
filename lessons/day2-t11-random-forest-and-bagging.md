# T11 · Random forest and bagging — Day 2

**Anchor task(s):**
- [`faio-2025/day1/qaz_letters.md`](../faio-2025/day1/qaz_letters.md)
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)

Rating 1 · **exam-probability rank 2** · ~45 min

External reference: mlcourse.ai topic 5, "Bagging and random forest" (not fetched here — the site is blocked from this environment).

## [concept-first]

A single deep tree (T10) has low bias and high variance: change a few training rows and the whole tree changes. Bagging attacks exactly that variance.

**Bootstrap.** Sample `n` rows **with replacement** from the `n` training rows. Each bootstrap sample contains about 63.2 % of the distinct rows (`1 − (1−1/n)ⁿ → 1 − e⁻¹`); the rest are duplicates. Fit one tree per sample, average the predictions (classification: majority vote or averaged probabilities).

**Why averaging reduces variance.** For `B` predictors each of variance `σ²` with pairwise correlation `ρ`:

```
Var(mean) = ρσ² + (1−ρ)σ²/B
```

Raising `B` kills the second term but not the first. So the gain depends on **decorrelating** the trees — and bootstrap alone leaves them quite correlated, because one dominant feature gets picked at the root of every tree.

**Feature subsampling — what makes it a *random* forest.** At every split, consider only a random subset of `max_features` columns. That is the second source of randomness, and it is what drives `ρ` down. Defaults: `√p` features for classification, `p` (or `1.0`) for regression, where `p` is the number of columns.

So: random forest = bagging of deep trees + per-split feature subsampling. Bias stays roughly that of one deep tree; variance drops. This is why you grow trees **deep** in an RF (`max_depth=None`) and control overfitting with `n_estimators` and `min_samples_leaf` instead.

**Out-of-bag estimate.** Each row was left out of ~36.8 % of the bootstraps. Predict each row using only the trees that never saw it, and you get a free validation score: `oob_score=True` gives you `rf.oob_score_` with no extra fits. It is a fine sanity check, but it cannot be grouped or stratified, so for task 5 it is not a substitute for `GroupKFold` (T07).

**`feature_importances_`.** The default is *mean decrease in impurity*: total weighted impurity drop a feature produced, summed over all trees, normalised to sum to 1. Read it as a ranking, not as a truth. Two biases: it inflates **high-cardinality / continuous** features (more thresholds to try), and it **splits credit** between correlated duplicates, so a genuinely strong feature that appears twice can rank low twice. `permutation_importance` on a held-out split is the honest version when you have the seconds to spare.

**When RF beats a single tree:** always, on accuracy, at a cost of interpretability and time. When RF loses to boosting (T12): on most well-engineered tabular data, by a small but real margin, because boosting reduces bias as well as variance. RF's advantage is that it is almost impossible to misconfigure.

Worked example: [`qaz_letters.md`](../faio-2025/day1/qaz_letters.md) ships dozens of handcrafted features across 42 letter classes. A single tree must carve 42 regions and each leaf sees few letters; 500 bootstrapped trees each vote, and the averaged probability distribution over 42 classes is far better calibrated than any one tree's leaf.

## [problem-first]

Open [`qaz_letters.md`](../faio-2025/day1/qaz_letters.md). The statement says it "places the spotlight on expert feature engineering and classical machine learning" and hands you `train.csv` / `test.csv` of engineered features plus `labels.csv`. Derive what it forces:

1. **42 classes, tabular, no pixels** → a tree ensemble is the intended family, and RF is the version with nothing to tune.
2. **Many features, several of them near-duplicates** (histogram bins, zoning densities, projections are all correlated) → feature subsampling helps a lot here, and impurity importance will split credit between the correlated ones. Do not prune a feature just because its importance is low.
3. **No scaling question** → unlike kNN in T10, you can feed the raw engineered columns directly and spend the time on features instead.
4. **`sample_submission.csv` ships** → predict string labels, write them in the sample's exact shape (T31).

Now [`task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md): a few hundred aggregated rows, dozens of features, binary accuracy. Here RF's out-of-the-box robustness matters most, because the dataset is too small for careful tuning to be reliable — differences of 0.01 in CV accuracy on a few hundred rows are noise.

## [code-first]

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestClassifier(
    n_estimators=500,      # more is never worse, only slower: no overfit knob here
    max_features="sqrt",   # the decorrelating knob; "sqrt" = default for classification
    max_depth=None,        # grow deep on purpose; bagging handles the variance
    min_samples_leaf=1,    # raise to 2-5 on small/noisy data
    n_jobs=-1,             # all cores; the single biggest wall-clock win
    oob_score=True,        # free validation estimate from the unused ~36.8% of rows
    random_state=42,
)

print("CV :", cross_val_score(rf, X, y, cv=cv, scoring="accuracy").mean())
rf.fit(X, y)
print("OOB:", rf.oob_score_)   # should land close to the CV number; if not, suspect leakage

# Impurity importance: a ranking, biased toward continuous/high-cardinality columns.
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(15))

# The honest version, measured on held-out rows rather than on training impurity.
pi = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42)
print(pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(15))

# n_estimators sweep: the curve flattens, it does not turn back down.
for B in (10, 50, 100, 300, 500):
    m = RandomForestClassifier(n_estimators=B, n_jobs=-1, random_state=42)
    print(B, cross_val_score(m, X, y, cv=cv).mean())
```

## [drill]

1. What fraction of distinct rows does one bootstrap sample contain, and where does the number come from?
2. Why does raising `n_estimators` never overfit, while raising a single tree's `max_depth` does?
3. What does `max_features="sqrt"` buy that bootstrapping alone does not?
4. Your OOB score is 0.95 and your `GroupKFold` CV is 0.71. What happened?
5. A feature you know is predictive has importance 0.004. Give two reasons that can happen.
6. Name two situations where you should reach for RF over gradient boosting in a timed round.

<details><summary>Answers</summary>

1. About 63.2 %: the chance a given row is never drawn in `n` draws with replacement is `(1−1/n)ⁿ → e⁻¹ ≈ 0.368`.
2. Extra trees only average more samples of the same estimator — they shrink the variance term, they do not add capacity. Depth adds capacity, so it can memorise.
3. Decorrelation. With all features available, one dominant column is chosen at most roots and the trees stay similar, leaving the `ρσ²` floor in place.
4. OOB cannot respect groups, so rows of the same task 5 repetition sat in the bootstrap and the OOB set at once. Trust the grouped CV.
5. It is correlated with another feature that absorbed the credit; or the impurity measure favours continuous high-cardinality columns over it. Check with permutation importance.
6. When you have almost no time to tune and need a defensible number immediately; and when the dataset is small enough (a few hundred rows, as in task 5) that boosting's extra capacity mostly buys variance.

</details>

**Rep:** fit the 500-tree RF above on the aggregated task 5 features, compare `oob_score_` with the `StratifiedKFold` CV, print the top 15 importances, and then drop the bottom half of the features and re-score. Note whether accuracy moved at all.

## Traps & 60-second recall

- Bootstrap ≈ 63.2 % distinct rows; the held-out 36.8 % becomes the OOB estimate.
- `n_estimators`: more is safer, only slower. `max_features`: the knob that decorrelates.
- Grow trees deep in an RF; control it with `min_samples_leaf`, not `max_depth`.
- `n_jobs=-1` every time — it is free speed in a 4-hour round.
- `feature_importances_` is impurity-based: biased to continuous columns, splits credit among correlated ones. Rank with it, decide with permutation importance.
- OOB is not grouped and not stratified; for grouped data (task 5) it can be wildly optimistic.
- RF is the model you can ship without tuning; boosting (T12) is the one that usually wins once you do.
