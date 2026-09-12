# T07 · Cross-validation & leakage — formula sheet (Day 1)

**Anchor task(s):**
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)
- [`faio-2025/qualification/task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

- Holdout: train on `⌊(1−t)·n⌋` rows, score on the remaining `⌈t·n⌉`; typical `t = 0.2`.
- k-fold: `k` disjoint folds of size `n/k`; each fold is validated once, trained on `k−1` folds → `k` fits, each on `n·(k−1)/k` rows.
- CV score: `CV = (1/k) Σ_{i=1..k} m(fold i)`; report the **standard error** `sd(mᵢ)/√k` beside it.
- Stratified k-fold: each fold keeps class proportions `p_c = n_c/n` — the default for classification, and what task 4 (`ru`/`kaz`/`eng`) and task 5 (`0`/`1`) both need.
- Group k-fold: folds split on a grouping key, so all rows sharing that key land in one fold. For task 5 the key is `id` — one yoga repetition at 200 Hz is many rows, and rows of one repetition must never straddle a fold.
- Leave-one-out: `k = n`; unbiased but `n` fits, unusable at scale.
- Repeated stratified k-fold: `r` shuffles × `k` folds = `r·k` fits; variance of the estimate drops ≈ `1/√r`.
- Total fits in a search: `|grid| × k` (see T19) — the only cost formula that matters under a 4-hour clock.
- Leakage checklist: (1) `fit` scalers, vectorisers, imputers, encoders on **train folds only**, `transform` the rest; (2) never compute a statistic (mean, TF-IDF vocabulary, target encoding, PCA basis) over train **and** test together; (3) group by example id; (4) split time series chronologically, never at random; (5) drop any id-derived or row-order feature; (6) deduplicate before splitting, or a duplicated row sits in both sides.

## Values worth memorising

- `cross_val_score(..., cv=5)` — `cv=5` is the scikit-learn default for a plain integer and the usual round-time compromise; `cv=3` is what the grid search in [`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb) uses.
- Train/val tradeoff: larger `k` → more training data per fit (less pessimistic bias) but higher variance and `k`× the time. `k = 5` trains on 80 %, `k = 10` on 90 % at double the cost.
- An integer `cv` with a classifier silently means **StratifiedKFold**; with a regressor it means plain **KFold**.
- `StratifiedKFold` and `KFold` do **not** shuffle by default — pass `shuffle=True, random_state=42`.
- Rule of thumb for a gap: CV minus leaderboard above ~0.05 absolute is leakage or a distribution shift, not noise.

## One-line traps

- Fitting `TfidfVectorizer` or `StandardScaler` on `train + test` before splitting — the single commonest leak; use a `Pipeline` so CV cannot do it.
- Random `KFold` on task 5's raw per-timestep rows: near-duplicate neighbouring samples put the same repetition on both sides and the CV score becomes meaningless. Aggregate per `id` first, or use `GroupKFold(groups=id)`.
- Forgetting `shuffle=True` on data stored sorted by label → folds with one class only.
- Tuning on the same split you report: the reported number is then optimistic. Tune inside CV, report on a held-out split.
- `scoring` left at default (accuracy) when the task's metric differs — the fold ranking you trust is then the wrong one.
