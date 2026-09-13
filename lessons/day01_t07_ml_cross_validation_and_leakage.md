# T07 · Cross-validation and leakage — Day 1

**Anchor task(s):**
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)
- [`faio-2025/qualification/task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md)

Rating 1 · **exam-probability rank 1** · ~50 min

Validation is the only instrument you have during the round. The leaderboard shows you one number, late, on data you cannot inspect. If your local estimate is wrong, every decision you make after it is wrong too.

## [concept-first]

**Holdout.** Split once: train on a fraction, score on the rest. With $n$ rows and a test fraction $t$, you train on $\lfloor (1-t)\,n \rfloor$ rows. Cheap, and noisy — one unlucky split moves the number more than most model changes do.

**k-fold.** Cut the data into $k$ disjoint folds. Each fold is validated once against a model trained on the other $k-1$, so every row is predicted exactly once and you pay $k$ fits. The estimate is the fold mean:

$$
\text{CV} = \frac{1}{k}\sum_{i=1}^{k} m_i
$$

Always read it with its spread. The standard error of that mean is

$$
\text{SE} = \frac{\mathrm{sd}(m_i)}{\sqrt{k}}
$$

and a change smaller than roughly $2\,\text{SE}$ is not a real improvement. This single habit stops you from chasing noise for an hour.

**Stratified k-fold.** Each fold preserves the class proportions $p_c = n_c / n$. The default for classification, and what both ML tasks in the 2025 qualification need — [`task4`](../faio-2025/qualification/task4_Who_Speaks_What.md) has three languages, [`task5`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md) has two classes.

**Group k-fold.** Folds split on a grouping key so every row sharing that key lands in the same fold. This is the one that decides whether your task 5 score means anything, and the reason this lesson exists.

**Leave-one-out** is $k = n$: nearly unbiased, $n$ fits, unusable under a 4-hour clock. **Repeated stratified k-fold** runs $r$ shuffles of $k$ folds for $r \cdot k$ fits, and the spread of the estimate shrinks roughly as $1/\sqrt{r}$.

**Leakage** is any path by which information from the validation rows reaches the model before it is scored. It always shows up the same way: local score excellent, leaderboard mediocre. The gap is the diagnosis.

## [problem-first]

Open [`task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md). Signals `ax ay az wx wy wz` at **200 Hz**, one `id` per yoga repetition, binary label, metric accuracy.

Work out what that sampling rate does to validation. A two-second repetition is about 400 rows. Consecutive rows inside one repetition are nearly identical — 5 ms apart, same person, same motion. So if you shuffle rows at random into folds:

1. Roughly 80 % of every repetition lands in the training folds.
2. The validation rows are near-duplicates of rows the model just memorised.
3. Accuracy comes back at 0.99, and it is measuring nothing.

The label is per repetition, so the *example* is the repetition, not the row. That makes `id` the grouping key, and `GroupKFold` or `StratifiedGroupKFold` the only honest splitter here.

Now [`task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md), which leaks in a different place. The features come from a `TfidfVectorizer`, and the vocabulary and the IDF weights are *learned from data*. Fit that on train plus test and every fold has already seen the validation distribution. The fix is structural, not careful — put the vectoriser in a `Pipeline` so cross-validation refits it inside each fold and cannot cheat.

## [code-first]

```python
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

# --- task 5: the example is the repetition, so group on id -------------------
# X_feat: one row per id (already aggregated). groups: the id of each row.
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    HistGradientBoostingClassifier(), X_feat, y, groups=groups,
    cv=cv, scoring="accuracy", n_jobs=-1,
)
# Report the mean WITH its standard error, or you cannot tell noise from gain.
print(f"{scores.mean():.4f} ± {scores.std(ddof=1) / np.sqrt(len(scores)):.4f}")

# --- task 4: the vectoriser must be refit inside every fold ------------------
# A Pipeline makes leakage structurally impossible: cross_val_score calls
# fit() on the training folds only, vectoriser included.
pipe = make_pipeline(
    TfidfVectorizer(sublinear_tf=True, max_features=20_000),
    LogisticRegression(C=10, solver="saga", max_iter=200, random_state=42),
)
# scoring must match the competition metric, not the default accuracy.
print(cross_val_score(pipe, texts, labels, cv=5, scoring="f1_macro").mean())

# WRONG, and the commonest mistake in the room:
#   X_all = TfidfVectorizer().fit_transform(pd.concat([train.text, test.text]))
# The IDF weights now encode the test set. Local score rises, leaderboard does not.
```

## [drill]

1. Task 5 raw rows, random `KFold`, accuracy 0.99, leaderboard 0.63. Name the cause and the fix.
2. Write the standard error of a 5-fold CV whose fold scores are 0.80, 0.84, 0.78, 0.86, 0.82. Is a rival model at 0.83 better?
3. Why does `Pipeline` prevent vectoriser leakage when a manual `fit_transform` does not?
4. Data sorted by label, `KFold(n_splits=5)` with default arguments. What happens?
5. An integer `cv=5` is passed with a classifier. Which splitter does scikit-learn actually use, and does it shuffle?
6. You aggregate task 5 to one row per `id`. Do you still need `GroupKFold`?

<details><summary>Answers</summary>

1. Rows of one repetition are in both train and validation folds, so the model is scored on near-duplicates of its training data. Aggregate per `id` first, and split with `StratifiedGroupKFold(groups=id)`.
2. Mean $0.82$, $\mathrm{sd} = 0.0316$ (ddof=1), so $\text{SE} = 0.0316/\sqrt{5} = 0.0141$. A rival at $0.83$ is inside $1\,\text{SE}$ — not a real difference; keep the simpler model.
3. `cross_val_score` calls `fit` on the pipeline, so the vectoriser is refit on each training fold and only `transform`s the validation fold. A manual `fit_transform` over the whole frame happens once, before any split exists.
4. Every fold contains one class, or close to it. Training sees classes the validation fold does not, and the scores are garbage. Pass `shuffle=True, random_state=42`, and prefer stratification.
5. `StratifiedKFold` for a classifier (plain `KFold` for a regressor), and **no**, it does not shuffle by default.
6. No — once each `id` is one row, plain `StratifiedKFold` is correct. Grouping matters only while multiple rows share an example.

</details>

**Rep:** score the same task 5 features twice — once with random `KFold` on raw per-timestep rows, once with `StratifiedGroupKFold` on the aggregated table — and write down both numbers. The gap between them is the size of the lie you would otherwise have believed.

## Traps & 60-second recall

- Fit every learned transform — scaler, vectoriser, imputer, encoder, PCA basis, target encoding — on **training folds only**. Use a `Pipeline` and the problem disappears.
- Group on the example key whenever one example spans many rows. Task 5 is exactly this.
- `shuffle=True, random_state=42` on `KFold`/`StratifiedKFold`; neither shuffles by default.
- Set `scoring` to the competition metric. Task 4's shipped notebook tunes `f1_macro`, not accuracy.
- Report mean ± standard error. Ignore any gain under about $2\,\text{SE}$.
- Split time series chronologically, never at random. Deduplicate before splitting.
- Never tune and report on the same split. A local score far above the leaderboard is leakage until proven otherwise.
- Total fits in a search is $\lvert \text{grid} \rvert \times k$ — the cost formula that decides what you can afford (see T19).
