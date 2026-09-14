# grader.md — score your own answers like the real competition

FAIO does not publish a grader. Nothing in [`faio-2024/`](./faio-2024) or [`faio-2025/`](./faio-2025) contains scoring code; the only evaluation code in the archive is `classification_report` inside [`task4_solution_Who_Speaks_What.ipynb`](./faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb).

Everything below is reimplemented from the wording of each task statement, with the source path cited. Where a statement is ambiguous or self-contradictory, that is flagged rather than guessed.

## How grading actually works

| Round | Platform | You submit | Scored by |
|---|---|---|---|
| 2025 qualification, tasks 1–3 | Yandex.Contest 81908 | a typed number | exact match, 10 points each |
| 2025 qualification, tasks 4–6 | Yandex.Contest 81908 | a CSV of predictions | a metric, server-side |
| 2025 day 1 and day 2 | Kaggle | a CSV | a metric, server-side |
| 2024 day 1 | Yandex.Contest 69765 | a program, stdin → stdout | a metric with a threshold |

Tasks 4, 5 and 6 state plainly: *"Submit only the CSV file. Source code is not required."* Your program is never run by the grader — only your predictions are read.

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
```

Every grader below takes a submission file and a ground-truth file and returns the number the competition would report.

## Tasks 1–3 — exact answer match

Source: [`task1_stats_101.md`](./faio-2025/qualification/task1_stats_101.md), [`task2_Probability_theory_101.md`](./faio-2025/qualification/task2_Probability_theory_101.md), [`task3_Medicine.md`](./faio-2025/qualification/task3_Medicine.md)

Ten points per correct answer, nothing partial. Verified answers:

| Task | Answer | Why |
|---|---|---|
| 1 | `1.5 1 1` | ten values sum to 15; mode is `1` (four times); median averages the 5th and 6th |
| 2 | `0.4213` | $1 - (5/6)^3 = 91/216$ |
| 3 | `0.3333` | $0.0099 / 0.0297 = 1/3$ |

```python
def grade_answer(submitted: str, expected: str, tol: float = 5e-5) -> bool:
    """Compare numerically, field by field, so 0.33330 == 0.3333."""
    s, e = submitted.split(), expected.split()
    if len(s) != len(e):
        return False
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(s, e))

assert grade_answer("1.5 1 1", "1.5 1 1")
assert grade_answer("0.4213", "0.4213")
assert not grade_answer("0.42", "0.4213")     # too few decimals fails
```

The statements demand four decimal places for tasks 2 and 3. Submitting `0.42` is wrong even though the value rounds correctly.

## Task 4 — Who Speaks What (macro-F1)

Source: [`task4_Who_Speaks_What.md`](./faio-2025/qualification/task4_Who_Speaks_What.md)

**The statement names no metric at all** — it is truncated and ends mid-example with no Evaluation section. The shipped solution notebook reports `classification_report` and grid-searches on `scoring="f1_macro"`, so macro-F1 is the best available reading. Grade both, and optimise the lower one.

```python
def grade_task4(solution_csv: str, truth_csv: str) -> dict:
    sub = pd.read_csv(solution_csv)
    tru = pd.read_csv(truth_csv)

    assert list(sub.columns) == ["id", "label"], f"columns must be id,label — got {list(sub.columns)}"
    assert len(sub) == len(tru), f"row count {len(sub)} != expected {len(tru)}"
    assert sub.id.tolist() == list(range(len(sub))), "id must run 0..n-1 in test.csv order"
    bad = set(sub.label) - {"ru", "kaz", "eng"}
    assert not bad, f"labels must be ru/kaz/eng — found {bad}"

    return {
        "macro_f1": f1_score(tru.label, sub.label, average="macro"),
        "accuracy": accuracy_score(tru.label, sub.label),
    }
```

Macro-F1 averages the three classes equally, so whichever language you handle worst sets your score. Accuracy hides that.

## Task 5 — AI Yoga Instructor (accuracy)

Source: [`task5_Can_You_Become_AI_Yoga_Instructor.md`](./faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md) — *"evaluated using the Accuracy score"*

```python
def grade_task5(solution_csv: str, truth_csv: str) -> float:
    sub = pd.read_csv(solution_csv)
    tru = pd.read_csv(truth_csv)

    assert list(sub.columns) == ["id", "label"], f"columns must be id,label — got {list(sub.columns)}"
    assert len(sub) == len(tru), f"row count {len(sub)} != expected {len(tru)}"
    assert sub.id.tolist() == sorted(sub.id.tolist()), "ids must ascend from 0, in test order"
    assert set(sub.label) <= {0, 1}, "label must be 0 or 1"

    return accuracy_score(tru.label, sub.label)
```

## Task 6 — Simple Objects (relative-error accuracy)

Source: [`task6_Simple_Objects.md`](./faio-2025/qualification/task6_Simple_Objects.md), tested on 8000 images

$$
\text{Error Rate} = \frac{\lvert \hat{y} - y \rvert}{y}, \qquad
\text{Accuracy} = 1 - \min(1, \text{Error Rate})
$$

**The printed verdict rule is inverted.** The statement says `0, if Accuracy > 0.55`, which would award zero to every good solution and reward bad ones. The intended reading is 0 when accuracy falls *below* 0.55, rescaled above it. The grader below implements the intent and can show you both.

```python
def grade_task6(submit_csv: str, truth_csv: str, count_col: str = "count") -> dict:
    sub = pd.read_csv(submit_csv)
    tru = pd.read_csv(truth_csv)
    assert len(sub) == len(tru), f"row count {len(sub)} != expected {len(tru)}"

    yhat = sub[count_col].to_numpy(dtype=float)
    y = tru[count_col].to_numpy(dtype=float)

    err = np.abs(yhat - y) / y                 # relative, so small counts dominate
    acc = 1.0 - np.minimum(1.0, err)           # per image, in [0, 1]

    FLOOR = 0.55
    verdict = np.where(acc < FLOOR, 0.0, (acc - FLOOR) / (1.0 - FLOOR))
    return {
        "mean_accuracy": acc.mean(),
        "mean_verdict": verdict.mean(),        # the intended scoring
        "images_below_floor": int((acc < FLOOR).sum()),
    }
```

Two consequences worth internalising before exam day. The error is **relative**, so being off by 5 costs far more on an image holding 100 figures than on one holding 500 — test on the low-count images. And the floor is a cliff: an image scoring 0.54 earns exactly the same as submitting nothing for it.

## Task 6 output format — unresolved

The statement contradicts itself: the Output section asks for *"a CSV file with a single integer in each row"*, while the next sentence refers to *"the order of `id` values in `submit.csv`"*. Its "Important" paragraph is also copy-pasted from task 5 and names `test.csv` and `solution.csv`, neither of which task 6 ships. **Copy the shipped `submit.csv` template rather than trusting the prose.** The grader above takes a `count_col` argument so it works either way.

## 2025 day 1 — accuracy, all three

Sources: [`Missing_Fundamental_Puzzle.md`](./faio-2025/day1/Missing_Fundamental_Puzzle.md), [`qaz_letters.md`](./faio-2025/day1/qaz_letters.md), [`Who_is_the_best_pitcher.md`](./faio-2025/day1/Who_is_the_best_pitcher.md)

```python
def grade_accuracy_csv(sub_csv: str, truth_csv: str, key: str, label: str) -> float:
    """Join on the key column, so row order cannot silently corrupt the score."""
    sub = pd.read_csv(sub_csv)
    tru = pd.read_csv(truth_csv)
    merged = tru.merge(sub, on=key, how="left", suffixes=("_true", "_pred"))
    assert merged[f"{label}_pred"].notna().all(), "some ground-truth keys are missing from the submission"
    return accuracy_score(merged[f"{label}_true"], merged[f"{label}_pred"])

# Missing Fundamental: grade_accuracy_csv("sub.csv", "truth.csv", "Path", "Pitch_ID")
# Who is the best pitcher: grade_accuracy_csv("submission.csv", "truth.csv", "ID", "ANSWER")
```

The pitcher task compares free-text answers, so the real grader almost certainly normalises case and whitespace before comparing. Normalise both sides the same way when practising.

## 2025 day 2 — HearMe (mean listened fraction)

Source: [`HearMe_Personalized_Music_Recommender.md`](./faio-2025/day2/HearMe_Personalized_Music_Recommender.md)

Each user gets 50 ranked tracks. Every recommended track scores how much of it the user actually listened — one of 0, 0.25, 0.5, 0.75, 1.0. Sum per user, average over users, then normalise to 0–1 by dividing by 50.

$$
S_u = \sum_{j=1}^{50} f_{u,j}, \qquad \text{score} = \frac{1}{50}\cdot\frac{1}{|U|}\sum_{u \in U} S_u
$$

```python
def grade_hearme(sub_csv: str, listens_csv: str) -> dict:
    """listens_csv: user_id, item_id, fraction  (0, .25, .5, .75, 1.0)."""
    sub = pd.read_csv(sub_csv)
    assert {"user_id", "item_id", "rank"} <= set(sub.columns)
    per_user = sub.groupby("user_id").size()
    assert (per_user == 50).all(), "each user needs exactly 50 rows"
    assert sub.groupby("user_id")["rank"].apply(lambda r: sorted(r) == list(range(1, 51))).all(), \
        "rank must be 1..50 within each user"

    listens = pd.read_csv(listens_csv)
    hit = sub.merge(listens, on=["user_id", "item_id"], how="left")
    hit["fraction"] = hit["fraction"].fillna(0.0)      # not listened at all

    per_user_score = hit.groupby("user_id")["fraction"].sum()   # 0..50
    return {"mean_user_score": per_user_score.mean(), "normalised": per_user_score.mean() / 50.0}
```

The metric rewards listening *depth*, not clicks. Ranking by raw popularity scores worse than ranking by how far a user is likely to listen.

## 2025 day 2 — Lost in the Museum (Hit@3 over cosine similarity)

Source: [`Lost_in_the_Museum.md`](./faio-2025/day2/Lost_in_the_Museum.md)

You submit **embeddings only** — 20,000 rows, no duplicates or omissions — and the organisers do the matching. Each of 1,000 private queries is compared against all 19,000 other images by cosine similarity; a hit means the true painting is in the top 3.

$$
\text{sim}(\mathbf{q}, \mathbf{g}) = \frac{\mathbf{q} \cdot \mathbf{g}}{\lVert \mathbf{q} \rVert_2 \, \lVert \mathbf{g} \rVert_2}, \qquad
\text{Hit@3} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\, y_i \in \text{top-3}(q_i) \,]
$$

```python
def grade_hit_at_k(emb_csv: str, pairs_csv: str, k: int = 3) -> float:
    """emb_csv: image_name + feature_*.  pairs_csv: query_name, gallery_name (the truth)."""
    emb = pd.read_csv(emb_csv).set_index("image_name")
    assert len(emb) == 20_000, f"need exactly 20000 rows, got {len(emb)}"
    assert not emb.index.duplicated().any(), "duplicate image_name rows"

    X = emb.to_numpy(dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12   # L2-normalise: dot == cosine
    pos = {name: i for i, name in enumerate(emb.index)}

    pairs = pd.read_csv(pairs_csv)
    hits = 0
    for q, g in zip(pairs.query_name, pairs.gallery_name):
        sims = X @ X[pos[q]]
        sims[pos[q]] = -np.inf                               # never match yourself
        if pos[g] in np.argpartition(-sims, k)[:k]:
            hits += 1
    return hits / len(pairs)
```

Note the statement disagrees with itself twice: its prose asks for `image_name` plus `feature_*`, while its example header shows a leading `ID` column; and the Overview says a 10,000-image gallery while Evaluation ranks against all 19,000. Follow the shipped sample submission.

## 2024 day 1 — threshold scoring

These rounds score a program's stdout, not a CSV, and each uses a pass/fail threshold rather than a smooth metric.

| Task | Source | Rule |
|---|---|---|
| C, radioactive decay | [`C_radioactive_decay.md`](./faio-2024/day1/C_radioactive_decay.md) | 100 points if within 1% of the true half-life, else 0 |
| D, K-means | [`D_Kmeans.md`](./faio-2024/day1/D_Kmeans.md) | metric $m = 1 - L/L_{\text{baseline}}$; 100 points if at least 0.95 × the reference metric |
| F, SVM | [`F_SVM.md`](./faio-2024/day1/F_SVM.md) | 0 if accuracy < 0.97, else $100 \times \text{accuracy}$ |

```python
def grade_2024_C(pred_T0: float, true_T0: float) -> int:
    return 100 if abs(pred_T0 - true_T0) <= 0.01 * true_T0 else 0

def grade_2024_F(y_true, y_pred) -> float:
    acc = accuracy_score(y_true, y_pred)
    return 0.0 if acc < 0.97 else 100.0 * acc

def grade_2024_D(points, centroids, baseline_loss: float, reference_metric: float) -> dict:
    """L = mean squared distance from each point to its nearest centroid."""
    P, C = np.asarray(points, float), np.asarray(centroids, float)
    d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    L = d2.min(axis=1).mean()
    m = 0.0 if L >= baseline_loss else 1.0 - L / baseline_loss
    return {"metric": m, "points": 100 if m >= 0.95 * reference_metric else 0}
```

## Before you submit anything — the format check

A format error costs the whole task; a weak model only costs points. Run this on every submission.

```python
def check_submission(path: str, expected_rows: int, expected_cols: list[str],
                     id_col: str | None = "id") -> None:
    df = pd.read_csv(path)
    assert list(df.columns) == expected_cols, f"columns {list(df.columns)} != {expected_cols}"
    assert len(df) == expected_rows, f"{len(df)} rows, expected {expected_rows}"
    assert not df.isna().any().any(), "submission contains empty cells"
    if id_col:
        assert df[id_col].is_unique, "duplicate ids"
        assert df[id_col].tolist() == sorted(df[id_col].tolist()), "ids are not in ascending order"
    print(f"{path}: {len(df)} rows, columns {list(df.columns)} — OK")
```

Checklist drawn from the 2025 statements:

- **Exact filename.** Task 4 and 5 want `solution.csv`; task 6 wants `submit.csv`; Lost in the Museum wants `submission.csv`.
- **Exact row count.** Lost in the Museum demands exactly 20,000 rows with no duplicates or omissions.
- **Exact id order.** Tasks 5 and 6 require ids following `test.csv` order starting at 0 — even though neither task actually ships a file called `test.csv`.
- **Exact label spelling.** Task 4 wants `ru`, `kaz`, `eng` — not `kk`, not `EN`.
- **No index column.** Always write with `index=False`.

## Practising without the real answers

None of the datasets ship ground truth for the test split, so hold out part of the *training* data and grade against that:

```python
from sklearn.model_selection import StratifiedGroupKFold   # task 5: group on id
from sklearn.model_selection import train_test_split       # tasks 4, 6

# Task 4: split train.csv, write a fake solution.csv from the held-out part,
# then run grade_task4 against the labels you kept back.
```

Two rules that keep this honest, both from [`lessons/day01_t07_ml_cross_validation_and_leakage.md`](./lessons/day01_t07_ml_cross_validation_and_leakage.md):

1. Fit every learned transform on the training part only. A `TfidfVectorizer` fitted on both halves makes your local score meaningless.
2. For task 5, group by `id`. Rows of one repetition must never sit on both sides of the split — at 200 Hz they are near-duplicates, and a random split returns a fake score near 0.99.

## What is not in here

- **No official grader exists in the archive.** All of the above is reimplemented from statement wording.
- **The pitcher task's answer normalisation is unknown.** Case and whitespace handling are not stated.
- **The Kaggle leaderboards are closed.** Competition links in the 2025 day-1 and day-2 statements point to ended competitions, so the only scoring available now is your own.
