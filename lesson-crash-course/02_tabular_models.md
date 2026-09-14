# 02 · Tabular models — rows and columns

Ordered by what to try first. Start at the top, stop when the score is acceptable.

## The decision table

| Situation | Model | Why |
|---|---|---|
| **Default, any table** | `HistGradientBoosting` | Fast, handles `NaN`, no scaling, usually wins |
| Want the strongest tabular result | LightGBM or CatBoost | Slightly better than the above, extra install |
| Many text-like category columns | CatBoost | Handles categories without encoding |
| Under ~500 rows | `RandomForest` or `LogisticRegression` | Boosting overfits on tiny data |
| Need to explain the result | `DecisionTree` (shallow) or `LogisticRegression` | You can read the rule out loud |
| Predicting a number, roughly linear | `Ridge` | Two lines, hard to beat on linear data |
| Very wide data (more columns than rows) | `Ridge` or `LogisticRegression` with L2 | Trees struggle, regularised linear does not |
| Need a baseline in 30 seconds | `DummyClassifier` | Gives you the score to beat |

## 0. The baseline you must beat

```python
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
print("baseline:", dummy.score(X, y))
```

If your clever model does not beat this, something is broken. Submit the dummy first so a valid file exists.

## 1. HistGradientBoosting — the default

Built into scikit-learn, nothing to install.

```python
from sklearn.ensemble import HistGradientBoostingClassifier   # or ...Regressor

model = HistGradientBoostingClassifier(
    max_iter=400,           # number of trees
    learning_rate=0.06,     # smaller = needs more trees but generalises better
    max_depth=None,         # None lets trees grow as needed
    early_stopping=True,    # stops when validation stops improving
    random_state=42,
)
model.fit(X, y)
preds = model.predict(X_test)
```

**Use when:** almost always, on any table.
**Do not bother with:** scaling, imputation, one-hot for ordered categories.

Docs: https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting

## 2. LightGBM — the competition workhorse

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(X, y)
```

**Use when:** you want a little more accuracy than HistGB and can install it.
**Tuning, in order of impact:** `n_estimators` up and `learning_rate` down; then `num_leaves`; then the subsample ratios. Three values each is enough.

Docs: https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html

## 3. CatBoost — when categories are messy

```python
from catboost import CatBoostClassifier

cat_cols = X.select_dtypes(include="object").columns.tolist()
model = CatBoostClassifier(iterations=600, learning_rate=0.05,
                           depth=6, verbose=0, random_state=42)
model.fit(X, y, cat_features=cat_cols)      # raw strings, no encoding needed
```

**Use when:** the table has several string columns with many distinct values.
**The selling point:** you skip encoding entirely, which removes a whole class of bugs.

Docs: https://catboost.ai/docs/en/concepts/python-usages-examples

## 4. XGBoost — same family

```python
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05,
                          max_depth=6, subsample=0.8, random_state=42)
```

**Use when:** you already know it. It is not better than the two above for this purpose — pick one boosting library and learn its parameters rather than switching.

Docs: https://xgboost.readthedocs.io/en/stable/python/python_intro.html

## 5. RandomForest — the safe choice on small data

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                               n_jobs=-1, random_state=42)
model.fit(X, y)
print(pd.Series(model.feature_importances_, index=X.columns).nlargest(10))
```

**Use when:** few rows, or you want feature importances without extra code.
**Advantage:** almost impossible to overfit badly with default settings.
**Disadvantage:** usually a point or two below boosting on larger data.

Docs: https://scikit-learn.org/stable/modules/ensemble.html#random-forests

## 6. ExtraTrees — RandomForest's faster cousin

```python
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)
```

**Use when:** RandomForest is too slow, or as a second opinion whose errors differ from it.

## 7. LogisticRegression — the linear default

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, max_iter=1000, random_state=42),
)
```

**Use when:** wide sparse data (especially text), very few rows, or you need probabilities you can trust.
**The one parameter:** `C` is **inverse** regularisation — larger `C` means less regularisation. Try `0.1, 1, 10`.
**Needs scaling.** Always wrap in a pipeline.

Docs: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

## 8. Ridge and Lasso — for predicting numbers

```python
from sklearn.linear_model import Ridge, Lasso, RidgeCV

model = RidgeCV(alphas=[0.1, 1.0, 10.0])    # picks alpha by cross-validation itself
model.fit(X_scaled, y)
```

**Ridge:** shrinks all coefficients, good default.
**Lasso:** drives some coefficients to exactly zero, so it also selects features.
**Use when:** the relationship looks roughly linear, or as a fast sanity check against boosting.

Docs: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression

## 9. kNN — the no-training model

```python
from sklearn.neighbors import KNeighborsClassifier
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
```

**Use when:** small data, features on comparable scales, or as a quick second opinion.
**Must scale.** Without scaling, the column with the largest units dominates the distance.
**Tune only `n_neighbors`:** try 1, 5, 15, 30.

Docs: https://scikit-learn.org/stable/modules/neighbors.html

## 10. SVM — strong on small clean data

```python
from sklearn.svm import SVC, LinearSVC

model = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf"))   # <10k rows
fast  = make_pipeline(StandardScaler(), LinearSVC(C=1.0))           # large or sparse
```

**Use when:** under ~10,000 rows and the classes are reasonably separable.
**Avoid when:** the data is large — `SVC` scales badly and can hang.
**`LinearSVC`** is the fast variant and is excellent on text.

Docs: https://scikit-learn.org/stable/modules/svm.html

## 11. Naive Bayes — instant and surprisingly decent

```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB
model = MultinomialNB()      # counts, e.g. text
model = GaussianNB()         # continuous numbers
```

**Use when:** you need a result in one second, or as a baseline for text.
**Why it survives:** trains in milliseconds and rarely embarrasses itself.

Docs: https://scikit-learn.org/stable/modules/naive_bayes.html

## Unsupervised, when there are no labels

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X_scaled)
labels = kmeans.labels_

pca = PCA(n_components=50, random_state=42)     # compress many columns to few
X_small = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_.sum())      # how much information survived
```

**k-means:** group rows into `k` clusters. Needs scaling.
**PCA:** compress correlated columns. Useful before kNN or to speed up anything.

Docs: [clustering](https://scikit-learn.org/stable/modules/clustering.html) · [PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)

## Comparing several models in one cell

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(5, shuffle=True, random_state=42)
candidates = {
    "dummy":   DummyClassifier(strategy="most_frequent"),
    "logreg":  make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "rf":      RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    "hgb":     HistGradientBoostingClassifier(max_iter=300, random_state=42),
}
for name, m in candidates.items():
    s = cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{name:8s} {s.mean():.4f} +/- {s.std(ddof=1)/np.sqrt(5):.4f}")
```

Run this once, pick the winner, move on. A difference smaller than twice the reported spread is not a real difference.

## Feature engineering that pays for itself

Cheap and often worth more than changing the model:

```python
df["ratio"]   = df.a / (df.b + 1e-9)          # ratios of related columns
df["total"]   = df[["a", "b", "c"]].sum(axis=1)
df["is_zero"] = (df.a == 0).astype(int)        # a flag for a special value
df["a_log"]   = np.log1p(df.a)                 # tames a long tail
df["hour"]    = pd.to_datetime(df.ts).dt.hour  # split dates into parts
```

Article: [sklearn feature engineering guide](https://scikit-learn.org/stable/modules/preprocessing.html)
