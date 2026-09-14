# 06 · Cheatsheet — keep this open

## Imports

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
```

## First five minutes

```python
train = pd.read_csv("train.csv"); test = pd.read_csv("test.csv")
print(train.shape, test.shape)
print(train.isna().sum()[lambda s: s > 0])
print(train["target"].value_counts(normalize=True))
print(train.groupby("id").size().max())        # >1 means many rows per example
train.head()
```

## Submit something immediately

```python
from sklearn.dummy import DummyClassifier
most_common = train["target"].mode()[0]
pd.DataFrame({"id": range(len(test)), "label": most_common}).to_csv("solution.csv", index=False)
```

## The default model

```python
model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
s = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"{s.mean():.4f} +/- {s.std(ddof=1)/np.sqrt(5):.4f}")
model.fit(X, y)
```

## The default text model

```python
model = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2), LinearSVC())
print(cross_val_score(model, texts, labels, cv=5, scoring="f1_macro").mean())
```

## Count objects in an image

```python
import cv2
gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
bg = int(np.bincount(gray.ravel()).argmax())
mask = (np.abs(gray.astype(np.int16) - bg) > 20).astype(np.uint8) * 255
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = sum(1 for c in cnts if cv2.contourArea(c) >= 20)
```

## Collapse many rows into one per example

```python
agg = df.groupby("id")[cols].agg(["mean", "std", "min", "max"])
agg.columns = ["_".join(c) for c in agg.columns]
agg["n_rows"] = df.groupby("id").size()
agg = agg.fillna(0)
```

## Grouped validation

```python
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
cross_val_score(model, X, y, groups=ids, cv=cv, scoring="accuracy")
```

## Embeddings and similarity

```python
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
sims = emb @ emb[i]
top3 = np.argsort(-sims)[1:4]
```

## Validate the submission before uploading

```python
sub = pd.read_csv("solution.csv")
assert list(sub.columns) == ["id", "label"]
assert len(sub) == len(test)
assert not sub.isna().any().any()
assert sub.id.is_unique and sub.id.tolist() == sorted(sub.id)
```

## Parameters worth adjusting, and nothing else

| Model | Parameter | Try |
|---|---|---|
| Any boosting | `learning_rate` / `n_estimators` | 0.1/300, 0.05/600, 0.03/1000 |
| LightGBM | `num_leaves` | 15, 31, 63 |
| RandomForest | `min_samples_leaf` | 1, 2, 5 |
| LogisticRegression, SVC, LinearSVC | `C` | 0.1, 1, 10 |
| kNN | `n_neighbors` | 1, 5, 15, 30 |
| TfidfVectorizer | `ngram_range`, `min_df` | (1,1)/(1,2), 1/2/5 |

## Common errors

| Message | Cause | Fix |
|---|---|---|
| `could not convert string to float` | text column fed to a numeric model | encode it, or use CatBoost |
| `Input contains NaN` | model that cannot take missing values | fill them, or switch to `HistGradientBoosting` |
| `Found input variables with inconsistent numbers of samples` | X and y differ in length | check the merge |
| `ValueError: Unknown label type` | float labels in a classifier | `y.astype(int)` |
| Score 0.99 locally, poor on the leaderboard | leakage | check group splitting and where you fit transforms |
| `MemoryError` | too many one-hot columns | use ordinal encoding with trees |

## The ten rules

1. Submit a valid file before improving anything.
2. `HistGradientBoosting` first, for any table.
3. TF-IDF plus a linear model first, for any text.
4. OpenCV before any model, for simple shapes.
5. Collapse many rows per example into one row of summary features.
6. Group the split whenever one example spans several rows.
7. Fit every transform inside a `Pipeline`.
8. Set `scoring` to the competition metric.
9. Ignore any improvement smaller than twice the reported spread.
10. `index=False` on every `to_csv`.
