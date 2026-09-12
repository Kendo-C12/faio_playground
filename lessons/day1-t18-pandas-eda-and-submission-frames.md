# T18 · pandas: read, groupby/agg, merge, build a submission — Day 1

**Anchor task(s):**
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)
- [`faio-2025/qualification/task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md)

Rating 2 · **exam-probability rank 1** · ~50 min

Every ML task in the 2025 qualification is "read CSV → transform → write CSV". This is the only topic that appears in all three of them.

## [concept-first]

Four operations carry almost all of it.

**Read.** `pd.read_csv(path)` gives a DataFrame: named columns, an index. Check `df.shape`, `df.head()`, `df.dtypes`, `df.isna().sum()` before anything else. Three lines that prevent most wasted hours.

**Group and aggregate.** `df.groupby(key)[cols].agg([...])` collapses many rows per key into one row per key. This is the shape change you need whenever the data has *many rows per example* — exactly task 5, where one `id` is one yoga repetition sampled at 200 Hz.

A `groupby().agg()` with a list of functions produces a **MultiIndex column** — `('ax', 'mean')`. Flatten it immediately or every later `df['ax_mean']` fails:

```python
agg.columns = ["_".join(map(str, c)) for c in agg.columns]
```

**Merge.** `pd.merge(left, right, on="id", how="left")` joins by key. Use it to attach `y_train.csv` labels to aggregated features. `how="left"` keeps every feature row even when a label is missing, which makes the missing ones visible instead of silently dropping them.

**Write.** `df.to_csv("solution.csv", index=False)`. Forget `index=False` and you ship a nameless extra first column, which breaks a strict grader.

Worked example, the exact reshape task 5 needs:

```
X_train.csv           ->  one row per (id, timestep)
groupby("id").agg()   ->  one row per id, many feature columns
merge(y_train)        ->  features + label, one row per id
```

## [problem-first]

Open [`task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md). It gives you `X_train.csv`, `y_train.csv`, `X_test.csv`, and says each `id` is one repetition with signals `ax ay az wx wy wz` at 200 Hz.

Derive what pandas you need:

1. "Each `id` corresponds to one yoga repetition" but sampling is 200 Hz → **rows outnumber ids**. You need `groupby("id")`.
2. Labels live in a separate file keyed by `id` → you need `merge`, or `set_index("id")` on both and align.
3. The output is `id,label` "strictly follow the order in `test.csv`, starting from `0`" → you need deterministic ordering, so `sorted(index)` or a reindex against the test ids.
4. The statement's own example shows *one row per id*. So the very first thing to check is `X_test.csv.groupby("id").size().max()`. If it is 1, skip the aggregation entirely.

That last point is the whole lesson: the statement and the example disagree, and pandas answers the question in one line.

## [code-first]

```python
import pandas as pd

X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv")

# 1. Always look first. Shape mismatch here tells you whether to aggregate.
print(X.shape, y.shape)
print(X.groupby("id").size().describe())   # rows per example

# 2. Collapse many rows per id into one feature row per id.
SIG = ["ax", "ay", "az", "wx", "wy", "wz"]
agg = X.groupby("id")[SIG].agg(["mean", "std", "min", "max"])
agg.columns = ["_".join(c) for c in agg.columns]   # flatten MultiIndex, always

# 3. Attach labels by key, never by row position.
data = agg.merge(y.set_index("id"), left_index=True, right_index=True, how="left")
print(data["label"].isna().sum(), "ids without a label")   # should be 0

# 4. Submission: explicit column names, explicit order, no index column.
ids = sorted(agg.index)
pd.DataFrame({"id": ids, "label": [1] * len(ids)}).to_csv("solution.csv", index=False)
```

## [drill]

1. `df.groupby("id")[["ax"]].agg(["mean","std"])` — what is `df.columns[0]` afterwards?
2. You merge features (400 ids) with labels (398 ids) using `how="inner"`. What silently goes wrong for the submission?
3. Why is `pd.concat([features, labels], axis=1)` dangerous where `merge(on="id")` is safe?
4. Single-row group, `.agg("std")` — what value appears, and what does a tree model do with it?
5. Write the one line that tells you whether task 5 needs aggregation at all.

<details><summary>Answers</summary>

1. The tuple `('ax', 'mean')` — a MultiIndex level, not the string `ax_mean`. Flatten first.
2. You lose 2 ids, so `solution.csv` has 398 rows where the grader expects 400 — a format failure, not a score dip. Use `how="left"` and inspect the NaNs.
3. `concat` aligns on the **index position/label**, so any reordering or differing sort silently pairs the wrong label with the wrong row. `merge(on="id")` pairs by key.
4. `NaN`. `HistGradientBoostingClassifier` tolerates NaN; most other estimators raise. Fill with `0` to be safe.
5. `print(pd.read_csv("X_test.csv").groupby("id").size().max())` — `1` means no aggregation needed.

</details>

**Rep:** build the task 5 feature table end to end using only the column names in the statement, and assert `len(solution) == X_test.id.nunique()` before writing.

## Traps & 60-second recall

- Flatten MultiIndex columns the line after you create them.
- `to_csv(..., index=False)`, every time.
- Join by key with `merge`, never by position with `concat`.
- `how="left"` then count NaNs — make losses visible rather than silent.
- Check rows-per-id before designing the pipeline; the statement may not match the data.
