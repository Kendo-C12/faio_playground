# T20 · TF-IDF, bag of words and char n-grams — Day 3

**Anchor task(s):**
- [`faio-2025/qualification/task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md)
- [`faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)

Rating 1 · **exam-probability rank 1** · ~50 min

Text is not a feature. TF-IDF is the cheapest way to turn it into one, and on the 2025 text task it is also the winning way.

## [concept-first]

**Bag of words.** Fix a vocabulary of V terms. Each document becomes a row of V counts: how often each term appears, word order thrown away. Stack the rows and you have the **document-term matrix**, shape `(n_docs, V)`, almost all zeros — scikit-learn keeps it sparse, so 20,000 columns costs nothing.

**Term frequency (TF).** Count of term *t* in document *d*. Raw counts over-reward long documents and repeated words, so two fixes exist. Dividing by document length gives a relative frequency; `sublinear_tf=True` instead replaces the count with the log-damped form:

$$
\text{tf}_{\text{sublinear}}(t,d) = 1 + \log\big(\text{tf}(t,d)\big)
$$

That log is damping: the 10th occurrence of a word adds far less than the 2nd. The shipped notebook sets `sublinear_tf=True` for exactly this reason — support messages are short and repetitive.

**Inverse document frequency (IDF).** A term in every document separates nothing, so IDF down-weights it — sklearn's smoothed form is

$$
\text{idf}(t) = \log\!\left(\frac{1+n}{1+\text{df}(t)}\right) + 1
$$

and the cell value is the tf-idf product:

$$
\text{tfidf}(t,d) = \text{tf}(t,d) \cdot \text{idf}(t)
$$

Here $\text{df}(t)$ is the number of documents containing *t*: rare term → high IDF → high weight. Rows are then L2-normalised by default, so document length stops mattering.

**Vocabulary controls.** `max_features=20_000` keeps only the 20k most frequent terms — a hard memory and overfitting cap. `min_df=2` drops terms seen in one document only (typos, noise). `max_df=0.9` drops terms present in over 90% of documents.

**Word vs character n-grams.** The `analyzer` argument decides what a "term" is:

- `analyzer="word"` (the default) — terms are whitespace/punctuation-delimited tokens. A test word never seen in training contributes nothing at all.
- `analyzer="char_wb"` with `ngram_range=(1,4)` — terms are character runs of length 1 to 4, taken inside word boundaries so n-grams never straddle a space. Unseen words still match on their pieces.

For language ID on short text the evidence is **sub-word**. Kazakh Cyrillic has letters Russian does not use — `ә ғ қ ң ө ұ ү һ і` — and a single one of those characters decides `kaz` vs `ru`. A char n-gram model sees that letter as a feature; a word model only sees it as part of a token it may never have met.

Worked example. `"Сәлем"` in training, `"Сәлеметсіз"` in test. Word analyzer: zero overlap, no information. `char_wb` 1–4 grams share `с`, `ә`, `сә`, `сәл`, `сәле` — and `ә` alone is near-conclusive for `kaz`.

## [problem-first]

Open [`task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md). It gives `train.csv` (`text`,`label`), `test.csv` (`text`), labels exactly `ru`/`kaz`/`eng`, output `solution.csv` with `id,label`. There is **no Theory section** — unlike tasks 1–3, nothing is handed to you. The vectoriser choice is entirely yours.

What the statement forces:

1. "short conversational texts" → few tokens per row, so word features are sparse per document and char features are dense per document. Points at `char_wb`.
2. Three labels, no stated balance → check `train.label.value_counts()` before trusting an averaged score.
3. Two Cyrillic languages plus one Latin → **script itself** is a feature, and char 1-grams encode it for free.
4. `id` must run $0 \dots n-1$ in `test.csv` order, so never shuffle or filter the test frame.

And here is the contradiction worth knowing: the shipped notebook [`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb) uses `TfidfVectorizer(sublinear_tf=True, max_features=20_000)` — **no `analyzer` argument, so word-level** — and scores well. It wins on word features because it first augments the data hard (see T21), which manufactures the word forms the model would otherwise miss. Char n-grams and augmentation are two routes to the same robustness.

**How to decide in the round, in 5 minutes:** cross-validate both and read the number. Do not reason about it.

## [code-first]

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def pipe(vec):
    # C=10 and saga match the shipped notebook's tuned logistic regression.
    return make_pipeline(vec, LogisticRegression(C=10, solver="saga",
                                                 max_iter=1000, random_state=42))

# The notebook's exact vectoriser: WORD level, log-damped TF, 20k vocabulary cap.
word = TfidfVectorizer(sublinear_tf=True, max_features=20_000)

# The standard language-ID vectoriser: character n-grams inside word boundaries.
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4),
                       sublinear_tf=True, min_df=2, max_features=200_000)

for name, vec in [("word", word), ("char_wb", char)]:
    # f1_macro, not accuracy — the notebook's grid search scored on f1_macro.
    s = cross_val_score(pipe(vec), train.text, train.label, cv=5, scoring="f1_macro")
    print(f"{name:8s} macro-F1 {s.mean():.4f} +/- {s.std():.4f}")

char.fit(train.text)                                     # inspect before trusting
print(len(char.vocabulary_), "features;", list(char.vocabulary_)[:10])

best = pipe(char).fit(train.text, train.label)          # swap in whichever won
pred = best.predict(test.text)
pd.DataFrame({"id": range(len(pred)), "label": pred}).to_csv("solution.csv", index=False)
```

## [drill]

1. In `TfidfVectorizer(sublinear_tf=True)`, what replaces a raw count of 8?
2. A term appears in every training document. What is its IDF under sklearn's smoothed formula, and what does that do to its weight?
3. Why does `max_features=20_000` act as regularisation and not just as a memory saving?
4. Why `char_wb` rather than plain `char` for language ID?
5. Which analyzer does the shipped task 4 notebook use, and why does it still work?
6. You call `vectorizer.fit_transform(test.text)` on the test set. Name the two things that break.

<details><summary>Answers</summary>

1. $1 + \log(8) \approx 3.08$ — the log damps repetition so a word said eight times is not eight times the evidence.
2. $\log\!\left(\frac{1+n}{1+n}\right) + 1 = 1$, the floor. It is not removed, only given the minimum weight; use `max_df` to actually drop it.
3. Fewer columns than examples means fewer free parameters for the linear model, so less capacity to memorise rare training-only terms.
4. `char_wb` pads each word and never builds n-grams spanning a space, so features stay within-word evidence instead of accidental cross-word junk.
5. Word level — no `analyzer` argument is passed, so the default applies. It works because the augmentation step first generates transliterated and de-diacriticised variants, putting the hard word forms into training directly.
6. The vocabulary and the IDF weights are refit on test, so train and test columns no longer mean the same thing — and it is label-free leakage of test distribution. Always `fit_transform` on train, `transform` on test.

</details>

**Rep:** run the two cross-validations above on task 4's `train.csv`, record both macro-F1 values, then re-run the char version with `ngram_range=(1,2)` and `(1,5)` and note which n-gram span is actually worth the fit time.

## Traps & 60-second recall

- `fit_transform` on train, `transform` on test. Never refit on test.
- The default analyzer is **word**. If you want characters you must say `analyzer="char_wb"`.
- `sublinear_tf=True` damps repetition; it is nearly free and almost always helps on short text.
- `min_df=2` kills single-document noise; `max_df` kills ubiquitous terms; `max_features` caps both memory and overfitting.
- Char n-gram vocabularies explode — cap `ngram_range` at 4, not 6.
- Score task 4 with `f1_macro`, matching the notebook's grid search, not accuracy.
- Decide word vs char by cross-validation in five minutes, not by argument.
