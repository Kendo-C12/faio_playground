# 01 · Data preparation

Read this first, every time. Most lost points come from data handling, not from model choice.

## Load and look

```python
import pandas as pd, numpy as np

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print(train.shape, test.shape)
print(train.dtypes)
print(train.isna().sum()[lambda s: s > 0])      # only columns with missing values
print(train["target"].value_counts(normalize=True))
train.head()
```

Four things to note immediately:

- **Which columns are numbers and which are text/categories** — decides your encoding.
- **How many rows** — under ~500 means keep the model simple.
- **Class balance** — a 95/5 split means accuracy is a misleading metric.
- **Columns present in train but missing from test** — those are leaks or the target.

Docs: [pandas 10-minute intro](https://pandas.pydata.org/docs/user_guide/10min.html)

## Missing values

| Situation | Do this |
|---|---|
| Using LightGBM, CatBoost, XGBoost, or `HistGradientBoosting` | **Nothing.** They handle `NaN` natively and often better than imputation |
| Using anything else, numeric column | Fill with the median |
| Using anything else, category column | Fill with the string `"missing"` — it is a category like any other |
| A column is >50 % missing | Consider dropping it, but first add `df["col_was_missing"] = df.col.isna()` |

```python
from sklearn.impute import SimpleImputer

num_cols = train.select_dtypes(include=np.number).columns
cat_cols = train.select_dtypes(include="object").columns

train[num_cols] = train[num_cols].fillna(train[num_cols].median())
train[cat_cols] = train[cat_cols].fillna("missing")
```

Docs: [sklearn imputation](https://scikit-learn.org/stable/modules/impute.html)

## Categories into numbers

| Situation | Method | Code |
|---|---|---|
| Few distinct values (<15), no order | One-hot | `pd.get_dummies(df, columns=cat_cols)` |
| Many distinct values | Ordinal + a tree model | `OrdinalEncoder()` |
| Using CatBoost | **Nothing** — pass `cat_features=` and it handles them | see [02](./02_tabular_models.md) |
| Genuinely ordered (small/medium/large) | Map by hand | `df.size.map({"small":0,"medium":1,"large":2})` |

```python
# The safe default: one-hot, aligned across train and test
full = pd.concat([train.drop(columns="target"), test], keys=["tr", "te"])
full = pd.get_dummies(full, columns=cat_cols, dummy_na=False)
X, X_test = full.loc["tr"], full.loc["te"]
```

Aligning matters: if you encode train and test separately, a category appearing in only one of them produces mismatched columns and the model crashes or silently misreads.

Docs: [sklearn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

## Scaling — only for some models

| Needs scaling | Does not need scaling |
|---|---|
| kNN, SVM, logistic regression, k-means, PCA, neural networks | Decision trees, random forest, all gradient boosting |

Trees compare thresholds, so the units do not matter. Distance-based models compare magnitudes, so they do.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Put the scaler INSIDE a pipeline. Never scale before splitting.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
```

## Splitting for validation

```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Quick: one split
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Better: 5-fold, one line
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"{scores.mean():.4f} +/- {scores.std(ddof=1)/np.sqrt(5):.4f}")
```

Three rules that protect the score:

1. `stratify=y` on classification, always. Without it a fold can miss a class entirely.
2. `shuffle=True, random_state=42` — neither `KFold` nor `StratifiedKFold` shuffles by default.
3. **If several rows describe the same thing** — the same person, the same repetition, the same image — use `GroupKFold` so they cannot land on both sides. Otherwise the model is scored against near-copies of its own training data.

```python
from sklearn.model_selection import StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_score(model, X, y, groups=ids, cv=cv)
```

Docs: [cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)

## Leakage — the four ways to fool yourself

1. Fitting a scaler or vectoriser on train **and** test together. Use `Pipeline` and it cannot happen.
2. Rows of the same entity split across folds. Use `GroupKFold`.
3. A feature computed from the target, e.g. target encoding done before the split.
4. Row order or `id` used as a feature when the file was sorted by label.

Symptom for all four: excellent local score, poor leaderboard score.

Article: [Data leakage in machine learning — sklearn docs](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

## Writing the submission

```python
sub = pd.DataFrame({"id": test_ids, "label": preds})
sub.to_csv("solution.csv", index=False)

assert list(sub.columns) == ["id", "label"]
assert len(sub) == len(test)
assert not sub.isna().any().any()
print(sub.head())
```

`index=False` is not optional — without it the file gains a nameless extra column and a strict grader rejects it.

Copy the sample submission file's exact column names and row order. When the task description and the sample file disagree, believe the file.
