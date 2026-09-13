# T19 · Hyperparameter search under a clock — Day 2

**Anchor task(s):**
- [`faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)

Rating 2 · **exam-probability rank 1** · ~35 min

## [concept-first]

A hyperparameter is a setting you choose before fitting (`C`, `max_depth`, `learning_rate`); a parameter is what fitting learns (the coefficients, the splits). Search means trying settings and keeping the one with the best **cross-validated** score.

**Grid search.** Enumerate the full Cartesian product of the lists you give. Exhaustive, reproducible, and exponential in the number of knobs. `GridSearchCV` refits the best configuration on all the data at the end (`refit=True`), so `grid.best_estimator_` is ready to predict.

**Random search.** Sample `n_iter` configurations from distributions instead. Better per unit of time beyond two or three knobs, because most knobs barely matter and a grid still resolves them finely; `RandomizedSearchCV(n_iter=30)` usually lands as well as a 200-point grid.

**The cost formula, the only arithmetic that matters in a timed round:**

$$
\begin{aligned}
\text{fits} &= (\text{number of grid points}) \times \text{cv} && (+1 \text{ for the final refit}) \\
\text{wall clock} &\approx \frac{\text{fits} \times (\text{time of one fit})}{\texttt{n\_jobs}}
\end{aligned}
$$

`n_jobs=-1` uses every core: the cheapest speedup there is, identical results, only the clock changes — at the price of one copy of the data per worker.

**`scoring` must equal the competition metric.** This is the single most consequential argument. If the task is scored with accuracy and you search on `f1_macro`, you select the configuration that is best at something you are not being graded on. Task 5 says "evaluated using the **Accuracy** score" → `scoring="accuracy"`. Task 4's statement in this archive never states a metric, and the shipped notebook grades with `classification_report` and searches on `f1_macro` — which is the right default for a 3-class problem with uneven class sizes, but it is the notebook's choice, not a stated rule.

**Order of operations.** Fix the features and the model family first (T16, T12), then search — tuning before the features are settled means re-searching from scratch after every feature change.

## [problem-first]

[`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb) ships a grid search, **commented out**, with the winning settings hard-coded below it. Read it exactly as written:

```python
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2", "l1"],
    "solver": ["liblinear", "saga"],
    "class_weight": [None, "balanced"]
}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid,
                    cv=3, scoring="f1_macro", n_jobs=-1)
```

Dissect it.

- **Size.** $4 \times 2 \times 2 \times 2 = 32$ points, `cv=3` → $32 \times 3 = 96$ fits, plus 1 refit = **97**. On a 20,000-feature TF-IDF matrix that is minutes, not seconds — which is plausibly why it ships commented out and the result hard-coded instead.
- **`cv=3`, not 5.** A deliberate cost cut: 3 folds instead of 5 is 40 % fewer fits for a slightly noisier estimate. The right trade when you are ranking 32 candidates rather than reporting a final number.
- **`scoring="f1_macro"`.** Three classes, macro averaging, so every language counts equally regardless of how many messages it has. Matching the metric is the point; here the notebook had to choose one because the statement does not state it.
- **Solver/penalty validity.** `l1` works with both `saga` and `liblinear`, so this grid is legal — but in general not every solver takes every penalty (`lbfgs` has no L1), scikit-learn raises on an illegal pair, and `error_score` decides whether that kills the search. Pass a **list of dicts** so each solver only gets its own penalties. `n_jobs=-1` spreads the 97 fits over all cores.
- **What it settled on:** `LogisticRegression(C=10, penalty="l2", solver="saga", max_iter=200, random_state=42)` — the high end of the `C` range, i.e. the least regularisation on offer, which says the 20k TF-IDF features were not the overfitting risk here. Note `max_iter` dropped from the search's 1000 to 200 in the final model: a speed choice, and the one place a silent `ConvergenceWarning` could cost score (T08).

The lesson the notebook teaches by its own comment marks: **write the search, run it once, record the winner, then hard-code it.** Re-running a 97-fit grid every time you re-execute the notebook is time you do not have.

## [code-first]

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import loguniform, randint

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)   # 3 folds: cost cut

# Valid combinations only, as a LIST OF DICTS — no illegal solver/penalty pairs.
param_grid = [
    {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"],
     "solver": ["liblinear", "saga", "lbfgs"], "class_weight": [None, "balanced"]},
    {"C": [0.01, 0.1, 1, 10], "penalty": ["l1"],
     "solver": ["liblinear", "saga"], "class_weight": [None, "balanced"]},
]
n_points = 4*1*3*2 + 4*1*2*2                       # 24 + 16 = 40 points
print("fits:", n_points * cv.get_n_splits() + 1)   # 40*3 + 1 = 121 — budget this BEFORE running

grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=cv,
                    scoring="f1_macro",    # must equal the competition metric
                    n_jobs=-1, refit=True, verbose=1, error_score="raise")
grid.fit(X_train_tfidf, y_train)
print(grid.best_params_, grid.best_score_)
# Copy best_params_ into a hard-coded estimator, then stop re-running the search.

# Many knobs -> random search. 40 samples, not the 4*4*4*3 = 192-point grid.
rnd = RandomizedSearchCV(
    HistGradientBoostingClassifier(early_stopping=True, random_state=42),
    {"learning_rate": loguniform(0.01, 0.3),
     "max_leaf_nodes": randint(8, 64),
     "min_samples_leaf": randint(5, 40),
     "l2_regularization": loguniform(1e-3, 10)},
    n_iter=40, cv=cv, scoring="accuracy",  # task 5's stated metric
    n_jobs=-1, random_state=42)
rnd.fit(X, y)
print(rnd.best_params_, rnd.best_score_)
```

Time-budget rule for a 4-hour round: **no single search longer than 15 minutes, and none at all until you have a valid submission file on disk.** Time one fit first (`%time model.fit(...)`), multiply by $\dfrac{\text{points} \times \text{folds}}{\text{cores}}$, and shrink the grid until the product fits. A tuned model you never submitted scores zero (T31).

## [drill]

1. How many fits does the notebook's grid cost, including the refit?
2. Why `cv=3` there rather than `cv=5`?
3. Your metric is accuracy and you search on `f1_macro`. What goes wrong, concretely?
4. When does random search beat grid search?
5. What does `n_jobs=-1` change, and what does it cost?
6. The notebook's winner is `C=10`, the largest value in the grid. What does that suggest, and what is your next move?

<details><summary>Answers</summary>

1. $4 \times 2 \times 2 \times 2 = 32$ points $\times$ 3 folds = 96, plus 1 refit on the full data = **97**.
2. Cost. 32 candidates need ranking, not precise measurement; 3 folds is 40 % fewer fits for slightly more noise.
3. You select the configuration that balances per-class recall best rather than the one that gets the most rows right — typically a `class_weight="balanced"` setting that trades majority-class accuracy away.
4. Once you have more than two or three knobs, or knobs that are continuous. A grid spends its budget resolving unimportant dimensions finely; random sampling covers the important ones better per fit.
5. It parallelises the fits across cores — pure wall-clock saving, identical results. It costs memory: one copy of the data per worker.
6. The boundary was hit, so the optimum may lie outside the grid: less regularisation still helps. Extend the range (`C = 30, 100`) and re-search that one axis only.

</details>

**Rep:** recreate the notebook's grid as the list-of-dicts version above, print the fit count before fitting, then cut it to a 12-point grid that costs under 40 fits and say which knobs you dropped and why.

## Traps & 60-second recall

- $\text{fits} = \text{points} \times \text{folds}$ (+1 refit). Compute it before you press run, every time.
- `scoring` must be the competition's metric — accuracy for task 5, the notebook's `f1_macro` for task 4.
- Illegal parameter pairs (solver vs penalty) break a grid; use a list of dicts.
- `n_jobs=-1` always; watch memory on large sparse matrices.
- Search after the features are frozen, never before.
- Winner at a grid edge means extend the grid on that axis.
- Run a search once, hard-code `best_params_`, as the notebook itself does.
- No search until a valid submission file already exists on disk.
