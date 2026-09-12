# T12 · Gradient boosting from zero — Day 2

**Anchor task(s):**
- [`faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)
- [`faio-2025/day1/qaz_letters.md`](../faio-2025/day1/qaz_letters.md)

Rating 0 · **exam-probability rank 1** · ~55 min

This is the default first model for tabular data in a timed round. Read it twice.

## [concept-first]

Start from zero. Forget forests — boosting is a different idea.

**The idea: fit what you got wrong.** Make a deliberately weak first guess. Measure how far off you are, row by row. Fit a small tree to *those errors*. Add a shrunken version of it to your prediction. Measure the remaining errors. Fit another small tree. Repeat a few hundred times.

For regression with squared error, "how far off" is literally the residual:

```
F₀(x) = mean(y)                          # the weak first guess
for m = 1..M:
    rᵢ   = yᵢ − F_{m−1}(xᵢ)              # residual: what is still missing
    h_m  = small tree fitted to (x, r)   # learn the pattern in the mistakes
    F_m  = F_{m−1} + ν · h_m(x)          # ν = learning rate, e.g. 0.1
```

That is the whole algorithm. The model is **additive and stagewise**: a sum of many small corrections, each one built knowing all the previous ones. Each tree is weak on purpose — depth 3 to 6, a few dozen leaves.

**Why "gradient".** Squared error makes the residual `y − F` equal to the negative gradient of the loss with respect to the prediction. Swap in a different loss and you fit the negative gradient instead — that is all "gradient boosting" means. For binary classification the loss is log-loss, the model predicts a **log-odds** score `F(x)`, the probability is `σ(F(x)) = 1/(1+e^{−F(x)})`, and the "residual" each tree fits is `yᵢ − σ(F(xᵢ))` — the probability error. Multi-class (42 letters in `qaz_letters.md`) fits one additive model per class and softmaxes at the end.

**Bagging vs boosting, in one line.** RF (T11) averages strong independent trees to cut **variance**; boosting sums weak dependent trees to cut **bias**. So boosting *can* overfit by adding trees where RF cannot — boosting has a stopping problem, RF does not.

**Learning rate vs n_estimators.** `ν` (`learning_rate`) scales each tree's contribution, `M` (`n_estimators`/`max_iter`) is how many you add, and their product is the capacity — so halve `ν` and roughly double `M`. Small `ν` with large `M` generalises better and costs more time; `ν = 0.05–0.1` is the working range, `ν = 0.3` is for when the clock binds.

**Early stopping** chooses `M` without a search: hold out a validation slice, watch the validation loss each iteration, stop after `n_iter_no_change` rounds with no improvement. One fit instead of one fit per candidate `M`.

Other knobs by value: `max_depth`/`max_leaf_nodes` (capacity, 3–8 / 15–63), `min_samples_leaf`, `subsample` (<1.0 = stochastic boosting: regularisation plus speed), `l2_regularization`, `colsample_bytree`.

**Which library.**

- `HistGradientBoostingClassifier` (sklearn) — zero installs, histogram-binned so fast, **handles NaN natively**, takes `categorical_features`. The default in an offline round.
- **LightGBM** — biggest data, fastest, leaf-wise growth, most knobs. Named by the task 4 jury.
- **CatBoost** — many high-cardinality categorical columns; it encodes them for you, and has the best defaults of the three. Also named by the task 4 jury.
- `GradientBoostingClassifier` (sklearn, old) — avoid: exact-split, slow, no NaN support.

**Native NaN handling** matters more than it sounds: a histogram learner sends missing values to whichever side of each split lowers the loss, so a `std` that is `NaN` for a single-row task 5 group is *information*. No imputation step, and no leak from imputing with train+test statistics.

Worked example: the closing cell of [`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb) lists, as the jury's own suggestions for improving the score, "SVM / Logistic Regression / **CatBoost / LightGBM** on TF-IDF features" and the same four "on FastText features". That is the FAIO jury naming boosting as a sanctioned tool, in the round you are sitting, for a text task. Boosting is not an exotic import here — it is on the official list.

## [problem-first]

Open [`task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md). After the T18 aggregation you have one row per repetition, dozens of numeric features, a binary label, accuracy as the metric. Why boosting first?

1. **Mixed units (g, deg/s, a row count)** → trees compare thresholds, so no scaling step and no scaler to leak.
2. **`std` is NaN on single-row groups** → `HistGradientBoostingClassifier` eats it; `LogisticRegression` and kNN raise.
3. **Non-linear interactions** between spread, range and duration decide "correct vs incorrect pose" → depth-4 trees express interactions for free, where a linear model needs you to invent the product terms by hand.
4. **A few hundred rows** → small `max_leaf_nodes` plus early stopping, or you memorise. This is the one place RF is the safer pick.

Then [`qaz_letters.md`](../faio-2025/day1/qaz_letters.md): 42 classes, so boosting fits 42 additive models and wall-clock is 42× a binary problem — lower `max_iter` and keep `early_stopping=True`. And for task 4's sparse 20,000-column TF-IDF matrix boosting is the *wrong* first model: linear models (T08) dominate high-dimensional sparse text, which is why the notebook ships logistic regression and lists boosting only as an ensemble partner.

## [code-first]

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf = HistGradientBoostingClassifier(
    learning_rate=0.06,     # nu: small steps generalise; trades off against max_iter
    max_iter=500,           # M: an upper bound, early stopping picks the real value
    max_leaf_nodes=31,      # tree capacity; drop to 8-15 on a few hundred rows
    min_samples_leaf=10,    # raise on small data
    l2_regularization=1.0,  # shrinks leaf values
    early_stopping=True,    # carve a validation slice and watch it
    validation_fraction=0.15,
    n_iter_no_change=30,    # stop after 30 rounds with no improvement
    random_state=42,
)                           # NaN needs no imputation: splits route it optimally

print("CV accuracy:", cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())

clf.fit(X, y)
print("trees actually kept:", clf.n_iter_)       # early stopping's answer for M

# Measure the lr/M tradeoff rather than assuming it: each lr costs one fit per fold,
# because early stopping picks M for you instead of a second loop over n_estimators.
for lr in (0.3, 0.1, 0.05, 0.02):
    m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=2000,
                                       early_stopping=True, random_state=42)
    print(lr, cross_val_score(m, X, y, cv=cv, scoring="accuracy").mean())

# The two the task 4 notebook names, same shape of call:
# LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31,
#                subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
#   .fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[early_stopping(50)])
# CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, verbose=0,
#                    early_stopping_rounds=50)
#   .fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=["some_string_col"])
```

## [drill]

1. In one sentence each: what does boosting iteration `m` fit, and what is added to the model?
2. You halve `learning_rate`. What must happen to `n_estimators`, and why?
3. Bagging reduces which error component, boosting which? Which of the two can overfit by adding more trees?
4. For binary classification, what does the boosted score `F(x)` represent, and how do you get a probability?
5. Task 5 features contain `NaN` from single-row groups. What does `HistGradientBoostingClassifier` do, and what would `LogisticRegression` do?
6. Which FAIO file names CatBoost and LightGBM, and in what context?
7. Why is boosting the wrong first model for task 4's 20,000-feature TF-IDF matrix?

<details><summary>Answers</summary>

1. A small tree fitted to the negative gradient of the loss at the current predictions (for squared error, the residual `y − F`); `learning_rate × that tree` is added to the running prediction.
2. Roughly double it. Capacity is governed by `ν·M`, so smaller steps need more of them to reach the same fit — which is why you set `max_iter` high and let early stopping find the real value.
3. Bagging reduces variance, boosting reduces bias. Boosting can overfit with more trees; RF cannot — averaging adds no capacity.
4. A log-odds score; `p = σ(F(x)) = 1/(1+e^{−F(x)})`.
5. It routes missing values to whichever side of each split lowers the loss, treating missingness as signal. `LogisticRegression` raises a `ValueError` on NaN input.
6. The closing markdown cell of [`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb), recommending "SVM / Logistic Regression / CatBoost / LightGBM" on TF-IDF and on FastText features as ways to push the score further.
7. Sparse, very high-dimensional, near-linearly-separable data is the regime where linear models win; axis-aligned tree splits over 20,000 mostly-zero columns are slow and weak. The notebook's own choice is `LogisticRegression`.

</details>

**Rep:** fit the classifier above on the aggregated task 5 features; record CV accuracy and `clf.n_iter_`, and compare with the T10 tree and T11 forest on the same folds. Then rerun with `early_stopping=False, max_iter=2000` and watch CV fall — overfitting by adding trees, the thing RF cannot do.

## Traps & 60-second recall

- Each tree fits the **current errors**, not the target: boosting is a sum of small corrections.
- `learning_rate` × `n_estimators` is the capacity; tune one and let early stopping set the other — without it, `max_iter` is a guess that overfits.
- Trees need no scaling; histogram boosting needs no imputation either.
- Small data (task 5) → small `max_leaf_nodes`, higher `min_samples_leaf`, or prefer RF.
- Multi-class costs one additive model per class: 42 classes in `qaz_letters.md` is 42× the fits.
- Sparse high-dimensional text → linear model first; boosting is the ensemble partner, not the baseline.
- `HistGradientBoostingClassifier` needs no install; LightGBM for speed, CatBoost for categorical columns.
