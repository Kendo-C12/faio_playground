# T16 · Feature engineering: scaling, encoding, selection — Day 1

**Anchor task(s):**
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)
- [`faio-2025/day1/qaz_letters.md`](../faio-2025/day1/qaz_letters.md)

Rating 1 · **exam-probability rank 1** · ~50 min

## [concept-first]

A feature is a number the model can use. Feature engineering is deciding which numbers to compute. In a 4-hour round it buys more score than model choice does.

**Three jobs.**

*Build.* Turn raw structure into numbers. Raw rows → per-example statistics. Two columns → their ratio. A timestamp → hour of day. `qaz_letters.md` is a catalogue of this done to images: mean intensity, standard deviation, min/max, skewness, kurtosis, 16-bin histograms, Hu moments, radial distances, Fourier descriptors, projections, zoning, DCT, wavelets. Read that list as a menu — it is the jury telling you what handcrafted features they consider fair game.

*Scale.* `StandardScaler` → mean 0, sd 1. `MinMaxScaler` → $[0, 1]$. Needed by distance- and gradient-based models (kNN, SVM, logistic regression, k-means, PCA). **Not** needed by trees and boosting, which only compare thresholds. So: scaling is about the model, not the data.

*Encode.* Categories → numbers. One-hot for unordered with few levels; ordinal only when order is real. A label column like `ru`/`kaz`/`eng` is the target, not a feature — never encode it into X.

**Selection.** Drop constant columns (`df.nunique() == 1`), drop near-duplicates ($|\text{corr}| > 0.98$), then rank what is left by model importance. Fewer, better features beat more features when the training set is small.

Worked example — a derived feature that is better than any raw one. For IMU data, the acceleration *magnitude*

$$
\text{amag} = \sqrt{a_x^2 + a_y^2 + a_z^2}
$$

is invariant to how the sensor was rotated on the body. `ax` alone depends on the mounting angle; `amag` does not. One line, and it removes a nuisance factor the raw columns cannot.

## [problem-first]

Open [`task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md). The statement hands you six raw signals and a binary label. No feature list is given — unlike `qaz_letters.md`, which hands you the features already computed.

What does the task force you to invent?

1. The label is per repetition, the signals are per timestep → every feature must be an **aggregate over the repetition**.
2. "Correct vs incorrect pose" is about *shape of motion*, not average level. So spread features (`std`, $\max - \min$, RMS) carry more signal than `mean`.
3. Sensors have 3 axes each, and mounting orientation is arbitrary → build rotation-robust features: magnitudes, and correlations between axes.
4. Nothing says the segments are equal length → include `n_rows` as a feature; duration itself may separate the classes.

Compare with [`qaz_letters.md`](../faio-2025/day1/qaz_letters.md): there the jury did this work for you and the task is purely modelling. Recognising which of the two situations you are in, within the first five minutes, decides how you spend the round.

## [code-first]

```python
import numpy as np
import pandas as pd

SIG = ["ax", "ay", "az", "wx", "wy", "wz"]

def featurise(X):
    X = X.copy()
    # Derived, orientation-robust magnitudes: better than any single raw axis.
    X["amag"] = np.sqrt(X.ax**2 + X.ay**2 + X.az**2)
    X["wmag"] = np.sqrt(X.wx**2 + X.wy**2 + X.wz**2)
    cols = SIG + ["amag", "wmag"]

    # Spread matters more than level for "is the motion right".
    agg = X.groupby("id")[cols].agg(
        ["mean", "std", "min", "max", "median",
         lambda s: np.sqrt((s ** 2).mean()),        # RMS / energy
         lambda s: s.max() - s.min()]               # range
    )
    agg.columns = ["_".join(map(str, c)) for c in agg.columns]

    agg["n_rows"] = X.groupby("id").size()          # duration as a feature
    # Cross-axis correlation: captures coordinated vs sloppy movement.
    for a, b in [("ax", "ay"), ("ax", "az"), ("wx", "wy")]:
        agg[f"corr_{a}_{b}"] = X.groupby("id").apply(lambda g: g[a].corr(g[b]))

    return agg.fillna(0)                            # single-row groups give NaN std

def prune(df):
    df = df.loc[:, df.nunique() > 1]                # constants carry no information
    keep, seen = [], df.corr().abs()
    for c in df.columns:                            # drop near-duplicate columns
        if not any(seen.loc[c, k] > 0.98 for k in keep):
            keep.append(c)
    return df[keep]
```

Scaling is deliberately absent: the day-2 model of choice is boosting, which does not need it. Add `StandardScaler` the moment you try kNN or logistic regression on these features.

## [drill]

1. Which of these need scaled features: random forest, kNN, logistic regression, gradient boosting, k-means, PCA?
2. Why is `amag` more robust than `ax` for a body-worn sensor?
3. You one-hot encode a column with 5000 distinct values. What two problems follow?
4. Your validation accuracy is 0.99 but the leaderboard is 0.62. Which feature did you probably build?
5. `qaz_letters.md` lists Hu moments as a feature. What property makes them useful for letter images?
6. Name one feature that captures *duration* without touching the signal values.

<details><summary>Answers</summary>

1. Need it: kNN, logistic regression, k-means, PCA. Do not need it: random forest, gradient boosting.
2. It is invariant to sensor rotation — a rotated mounting changes how acceleration splits across x/y/z, but not the total magnitude.
3. A 5000-column explosion (memory and overfitting), and unseen categories at test time producing all-zero rows.
4. One leaking the target or the row identity — e.g. an id-derived number, or a statistic computed over train **and** test together.
5. They are invariant to translation, scale and rotation, so the same letter drawn bigger or slightly rotated maps to similar values.
6. `n_rows` — the number of samples in the group, which at a fixed 200 Hz is duration.

</details>

**Rep:** run `featurise` then `prune` on the task 5 training file, print how many columns survive pruning, and list the 10 highest-variance survivors.

## Traps & 60-second recall

- Fit scalers and any statistic on **train only**, then apply to test. Fitting on both leaks.
- Trees do not care about scale; distance models do.
- Spread beats level when the label is about *how* something was done.
- Build magnitudes for multi-axis sensors; they delete the orientation nuisance.
- Constant and duplicated columns are free to drop — do it before modelling.
- If the jury already lists the features (`qaz_letters.md`), skip this stage and spend the time on the model.
