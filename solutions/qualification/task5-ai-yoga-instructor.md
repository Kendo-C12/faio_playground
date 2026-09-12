# FAIO 2025 Qualification — Task 5: Can You Become an AI Yoga Instructor?

Source: [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md) · Yandex.Contest 81908, problem 5

## 1. What the problem asks

Build a machine learning model that classifies yoga repetitions as correctly or incorrectly performed, from wearable **IMU sensor** time series.

- Signals at **200 Hz**: `ax, ay, az` (linear acceleration, g) and `wx, wy, wz` (angular velocity, deg/s)
- Each `id` is one repetition
- `X_train.csv`, `y_train.csv` (`0` = incorrect, `1` = correct), `X_test.csv`
- Submit `solution.csv` with `id,label`, ids in `test.csv` order starting at `0`
- Metric: **Accuracy**

## 2. Knowledge required

- Binary classification, and accuracy as a metric
- Time-series feature engineering — this is the core skill: collapse each repetition's many sensor rows into one fixed-length feature vector (mean, sd, min, max, range, RMS/energy, zero crossings, correlations between axes, magnitude `√(ax²+ay²+az²)`)
- `groupby` aggregation over `id`
- A tabular classifier: random forest or gradient boosting
- Cross-validation, grouped by repetition so rows of one `id` never straddle the split
- Optionally signal processing: low-pass filtering, FFT band energies

**Given in the statement:** nothing. Task 5 has no Theory section — only Introduction, Formal Problem Statement, Input, Output, Evaluation, Example. Sensor semantics and units are given; all modelling knowledge is yours to bring.

## 3. Languages and libraries

**Not stated** for this task. The statement says "Submit only the `solution.csv` file. Source code is not required", so the judge only scores predictions — in practice no tooling restriction applies. Task 6 of the same round explicitly permits any language and libraries, which suggests round-wide intent, but task 5 itself is silent, and no global rules file exists in the archive or on a reachable faio.kz page. Python with pandas, numpy, scikit-learn is the natural fit.

## Solution

Aggregate per `id`, then fit a tree ensemble.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

SIG = ["ax", "ay", "az", "wx", "wy", "wz"]

def featurise(X):
    X = X.copy()
    X["amag"] = np.sqrt(X.ax**2 + X.ay**2 + X.az**2)
    X["wmag"] = np.sqrt(X.wx**2 + X.wy**2 + X.wz**2)
    cols = SIG + ["amag", "wmag"]
    agg = X.groupby("id")[cols].agg(["mean", "std", "min", "max", "median",
                                     lambda s: np.sqrt((s**2).mean())])
    agg.columns = ["_".join(map(str, c)) for c in agg.columns]
    agg["n_rows"] = X.groupby("id").size()
    return agg.fillna(0)

Xtr = featurise(pd.read_csv("X_train.csv"))
ytr = pd.read_csv("y_train.csv").set_index("id").loc[Xtr.index, "label"]
Xte = featurise(pd.read_csv("X_test.csv"))

clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06)
print(cross_val_score(clf, Xtr, ytr, cv=5, scoring="accuracy").mean())

clf.fit(Xtr, ytr)
pred = clf.predict(Xte.loc[sorted(Xte.index)])
pd.DataFrame({"id": sorted(Xte.index), "label": pred}).to_csv("solution.csv", index=False)
```

If accuracy stalls, add frequency features: per axis, the FFT energy in a few bands, and the dominant frequency. At 200 Hz a repetition of a second or two has plenty of spectral structure.

**Traps:**

- **Check the file layout first.** The statement's example shows one row per `id`, but the text describes time series per repetition. If it really is one row per `id`, skip aggregation and treat it as plain tabular data. Look at the file before writing the pipeline.
- Never let rows from one repetition land in both train and validation folds — that leaks and inflates your score.
- `solution.csv` ids must follow `test.csv` order from `0` upward. Sorting the index, as above, guarantees it.
- `std` is `NaN` for a single-row group. Fill it, or the classifier rejects the input.
