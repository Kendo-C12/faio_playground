# 03 · Text models

## The decision table

| Situation | Approach | Takes |
|---|---|---|
| **Default for any text classification** | TF-IDF + `LinearSVC` | 5 min |
| Short text, language ID, spelling variants | TF-IDF with **character** n-grams | 5 min |
| Need probabilities, not just labels | TF-IDF + `LogisticRegression` | 5 min |
| Instant baseline | TF-IDF + `MultinomialNB` | 1 min |
| Meaning matters more than exact words | `sentence-transformers` embeddings + `LogisticRegression` | 20 min |
| Find similar documents | embeddings + cosine similarity | 20 min |
| Search a large document set by keyword | BM25 (`rank_bm25`) | 10 min |
| Many labels, plenty of data, GPU available | fine-tune a small transformer | 1 h+ — usually skip |

## 1. TF-IDF + linear model — start here always

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

model = make_pipeline(
    TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                    max_features=100_000),
    LinearSVC(C=1.0, random_state=42),
)
print(cross_val_score(model, texts, labels, cv=5, scoring="f1_macro").mean())
model.fit(texts, labels)
preds = model.predict(test_texts)
```

**Why `LinearSVC`:** fastest strong classifier on sparse text, and usually a touch better than logistic regression. It gives no probabilities — swap to `LogisticRegression` if you need them.

The parameters that matter:

| Parameter | Meaning | Try |
|---|---|---|
| `ngram_range` | how many words per feature | `(1,1)`, `(1,2)` |
| `min_df` | ignore terms appearing in fewer than N documents | `2`, `5` |
| `max_features` | cap the vocabulary | `50000`–`200000` |
| `sublinear_tf` | damp repeated words with a log | `True` |
| `C` (on the model) | inverse regularisation | `0.1`, `1`, `10` |

Docs: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
Article: [Working with text data — sklearn tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

## 2. Character n-grams — for short or messy text

```python
model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4),
                    sublinear_tf=True, min_df=2),
    LinearSVC(C=1.0),
)
```

**Use when:** identifying a language, handling typos and slang, working with short messages, or any task where a specific *letter* is the evidence.

**Why it works:** word features are useless for an unseen word, while character features still extract signal from its letters. `char_wb` keeps n-grams inside word boundaries.

Always cross-validate `char_wb` against `word` — five minutes of comparison beats guessing.

## 3. Naive Bayes — the one-second baseline

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

model = make_pipeline(CountVectorizer(), MultinomialNB())
```

**Use when:** you want a number on the board immediately. Often within a few points of the best model on clean text.

Docs: https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes

## 4. Sentence embeddings — when meaning matters

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Small and fast. 'paraphrase-multilingual-MiniLM-L12-v2' if the text is not English.
encoder = SentenceTransformer("all-MiniLM-L6-v2")
Xtr = encoder.encode(list(texts), batch_size=64, show_progress_bar=True,
                     normalize_embeddings=True)
Xte = encoder.encode(list(test_texts), batch_size=64, normalize_embeddings=True)

clf = LogisticRegression(max_iter=2000).fit(Xtr, labels)
preds = clf.predict(Xte)
```

**Use when:** two sentences mean the same thing with different words, and TF-IDF cannot see it.
**Cost:** downloads ~90 MB, needs no GPU for small datasets.
**Note:** this uses the transformer as a *feature extractor* — you never train it. That is what keeps it fast.

Docs: https://sbert.net/docs/quickstart.html
Model list: https://sbert.net/docs/sentence_transformer/pretrained_models.html

## 5. Cosine similarity — find the closest text

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

emb = encoder.encode(docs, normalize_embeddings=True)   # already unit length
sims = cosine_similarity(emb[query_idx:query_idx+1], emb)[0]
top3 = np.argsort(-sims)[1:4]                           # skip itself at position 0
```

**Use when:** matching, retrieval, deduplication, "find the most similar item".
**With normalised vectors, cosine similarity is just a dot product** — `emb @ emb.T` is the whole computation.

Docs: https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity

## 6. BM25 — keyword search that beats TF-IDF for retrieval

```python
from rank_bm25 import BM25Okapi

corpus_tokens = [doc.lower().split() for doc in docs]
bm25 = BM25Okapi(corpus_tokens)
scores = bm25.get_scores("who scored the most runs".lower().split())
best = np.argsort(-scores)[:10]
```

**Use when:** searching a document collection with a keyword query. This is the retrieval half of a question-answering system.
**Why not TF-IDF:** BM25 saturates repeated terms and normalises by document length, which matters when documents differ in size.

The standard pattern is **retrieve then rerank**: BM25 fetches 50 candidates cheaply, then embeddings reorder those 50 accurately. Doing embeddings over the whole corpus is slow; over 50 candidates it is instant.

Docs: https://github.com/dorianbrown/rank_bm25
Article: [BM25 explained](https://en.wikipedia.org/wiki/Okapi_BM25)

## 7. Ready-made pipelines — zero training

```python
from transformers import pipeline

clf = pipeline("sentiment-analysis")
print(clf("this is great"))

zs = pipeline("zero-shot-classification")
print(zs("the engine makes a strange noise", candidate_labels=["car", "food", "music"]))
```

**Use when:** the task matches a common one and you have no labelled data at all.
**Do not use when:** you have training data — a TF-IDF model trained on it will beat a generic pipeline.

Docs: https://huggingface.co/docs/transformers/en/pipeline_tutorial

## Text cleaning — only what pays

```python
import re

def clean(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)        # strip URLs
    s = re.sub(r"\s+", " ", s)            # collapse whitespace
    return s.strip()
```

Skip stopword removal and stemming unless you measure a gain — TF-IDF already downweights common words, and aggressive cleaning often destroys signal.

**Augmentation is usually worth more than cleaning.** If you expect messy input at test time, manufacture messy training examples: swap characters, drop diacritics, transliterate between scripts. Keep the label unchanged.
