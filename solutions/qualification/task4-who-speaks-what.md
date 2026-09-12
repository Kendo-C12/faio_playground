# FAIO 2025 Qualification — Task 4: Who Speaks What?

Source: [`faio-2025/qualification/task4_Who_Speaks_What.md`](../../faio-2025/qualification/task4_Who_Speaks_What.md) · Yandex.Contest 81908, problem 4 · reference notebook in repo: [`task4_solution_Who_Speaks_What.ipynb`](../../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)

## 1. What the problem asks

Build a machine learning model for **language identification**. Given short conversational support messages, predict the language of each: `ru`, `kaz` or `eng`.

- `train.csv` — columns `text`, `label`
- `test.csv` — column `text`, no labels
- Submit `solution.csv` with `id,label`

Metric is not stated in the file; accuracy is implied by the task shape.

## 2. Knowledge required

- Text classification pipeline: text → features → classifier → predictions
- Turning text into vectors: bag of words, TF-IDF, and **character** n-grams (the key choice here)
- A linear classifier — logistic regression or linear SVM
- Train/validation split to check accuracy before submitting
- CSV handling and submission format discipline

**Given in the statement:** nothing. Task 4 has no Theory section at all — unlike tasks 1–3, which supply their own theory. Sections are Story, Formal Problem Statement, Input, Output only. Treat every item above as knowledge you must bring. The shipped solution notebook is the exception: it effectively hands you a worked reference.

## 3. Languages and libraries

**Not stated** for this task. What is known:

- The submission is a CSV of predictions, so the judge never runs your code — any language and any library is usable in practice
- Task 6 of the same round explicitly says "You may use any programming language and libraries", which suggests the round-wide intent is permissive, though it is written only in task 6
- faio.kz publishes no rules page reachable from here, and no global rules file exists in the archive

So: no known restriction, and nothing confirming one. Python with scikit-learn and pandas is the natural choice and matches the shipped notebook.

## Solution

Character n-grams are what make this nearly free. Kazakh Cyrillic has nine letters Russian does not use — `ә ғ қ ң ө ұ ү һ і` — and English is Latin script, so the three classes separate on character evidence alone.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, C=10),
)

print(cross_val_score(model, train.text, train.label, cv=5, scoring="accuracy").mean())

model.fit(train.text, train.label)
pred = model.predict(test.text)
pd.DataFrame({"id": range(len(pred)), "label": pred}).to_csv("solution.csv", index=False)
```

Expect accuracy in the high 0.9s. A pure script-detection rule (count Latin vs Cyrillic characters, then look for Kazakh-specific letters) is a strong fallback and a good sanity baseline.

**Traps:**

- `analyzer="word"` is much weaker on short messages, and mixed-language or transliterated text breaks word features entirely.
- Kazakh and Russian share most of the alphabet. If a message has no Kazakh-specific letter, the model must lean on frequency patterns — do not hand-code a rule that defaults everything ambiguous to `ru`.
- Label strings are `ru`, `kaz`, `eng` — `kaz`, not `kk`. Submit exactly these.
- `id` must run `0..n−1` in the order of `test.csv`.
