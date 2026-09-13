# T22 · Embeddings and cosine retrieval — Day 4

**Anchor task(s):**
- [`faio-2025/day2/Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md)
- [`faio-2025/day1/host-author-s-baseline.ipynb`](../faio-2025/day1/host-author-s-baseline.ipynb)

Rating 0 · **exam-probability rank 2** · ~50 min

## [concept-first]

**An embedding is a fixed-length vector of floats that stands in for an object**, chosen so that *closeness in the vector space means similarity in the real world*. An image becomes 512 numbers; a sentence becomes 384 numbers; the numbers themselves mean nothing individually. Only distances between them mean anything.

The dimension $D$ is fixed across the whole dataset — that is what makes the vectors comparable. `Lost_in_the_Museum` recommends 256 or 512.

**L2 normalisation.** The L2 norm is

$$
\lVert \mathbf{v} \rVert_2 = \sqrt{\sum_i v_i^2}
$$

Normalising means

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\lVert \mathbf{v} \rVert_2}
$$

giving a vector of length exactly 1 that points the same way. All normalised vectors live on the unit sphere, so only *direction* survives — magnitude is discarded on purpose, because for an image embedding magnitude usually encodes contrast or brightness, which is exactly the nuisance you want gone.

**Three similarity measures.** Dot product:

$$
\mathbf{v} \cdot \mathbf{w} = \sum_i v_i w_i
$$

Cosine similarity, in $[-1, 1]$:

$$
\cos(\mathbf{v}, \mathbf{w}) = \frac{\mathbf{v} \cdot \mathbf{w}}{\lVert \mathbf{v} \rVert_2 \, \lVert \mathbf{w} \rVert_2}
$$

Euclidean distance, in $[0, \infty)$:

$$
\lVert \mathbf{v} - \mathbf{w} \rVert_2 = \sqrt{\sum_i (v_i - w_i)^2}
$$

Cosine is the dot product with both magnitudes divided out — the cosine of the angle between the vectors. 1 means identical direction, 0 orthogonal, $-1$ opposite.

**When they coincide.** If both vectors are already L2-normalised then $\lVert \mathbf{v} \rVert_2 = \lVert \mathbf{w} \rVert_2 = 1$, so

- cosine similarity **is** the dot product, and
- $\lVert \mathbf{v} - \mathbf{w} \rVert_2^2 = 2 - 2\cos(\mathbf{v}, \mathbf{w})$.

So on the unit sphere, ranking by cosine descending, by dot product descending, and by Euclidean distance ascending give the **identical ordering**. That is the practical payoff: normalise once, then use the cheapest operation — a single matrix multiply `Q @ G.T` scores every query against every gallery item. Without normalisation the three disagree, and dot product is biased toward long vectors.

**Nearest-neighbour search.** Given a query $\mathbf{q}$, rank all gallery vectors by similarity and take the top $k$. Exact search is brute force: for 1,000 queries against 19,000 gallery vectors at $D = 512$, that is a $1000 \times 512$ by $512 \times 19000$ matmul — under a second, so no FAISS or approximate index is needed at this scale.

**Hit@k.** A "hit" is recorded for a query if the true match appears anywhere in its top $k$:

$$
\text{Hit@}k = \frac{\text{number of queries with a hit}}{\text{total queries}}
$$

It is a *ranking* metric: being second is as good as being first, and being fourth scores the same as being last. So the optimisation target is "get the truth into the top 3", not "make the top-1 similarity high".

Worked example: $\mathbf{q} = [3, 4]$, $\mathbf{g}_1 = [6, 8]$, $\mathbf{g}_2 = [-4, 3]$, so $\lVert \mathbf{q} \rVert_2 = 5$ and

$$
\cos(\mathbf{q}, \mathbf{g}_1) = \frac{50}{5 \cdot 10} = 1.0
$$

— same direction, so a perfect match despite $\mathbf{g}_1$ being twice as long. And $\cos(\mathbf{q}, \mathbf{g}_2) = 0/(5 \cdot 5) = 0$ — orthogonal. Raw dot products would have been 50 and 0; Euclidean distances 5 and 7.07, which ranks $\mathbf{g}_1$ first too. Normalisation is what makes the "twice as bright" copy score 1.0 instead of merely well.

## [problem-first]

Open [`Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md). The unusual thing is what you submit: "Your task is not to classify, detect, or manually match, but to generate a single fixed-dimensional feature vector (embedding) for each of the 20,000 images."

Read off the mechanics:

1. **20,000 rows, exactly.** 10,000 HQ paintings + 1,000 visitor photos + 9,000 distractors, "with no duplicates or omissions", "no filtering, sorting, or skipping allowed".
2. **You are never told which is which.** "You must treat all images equally and derive features purely from their pixel content." So there is no such thing as a query-specific trick: one function, applied to every file.
3. **Columns:** `image_name` (e.g. `00001.png`) plus `feature_0 … feature_{D-1}`, $D$ recommended 256 or 512. The statement's own example block also shows a leading `ID` column duplicating `image_name` — the archive is inconsistent with its prose here, so follow the sample `submission.csv` shipped in the dataset section, and include both if it does.
4. **Scoring is server-side** against a private ground truth: cosine similarity of each of the 1,000 private queries against all 19,000 others, top-3 by descending similarity, Hit@3. You cannot compute your own score, so the only local validation available is self-consistency — embed an image and a deliberately corrupted copy of it and check they are near neighbours.
5. **"Embeddings are expected to be L2-normalized for optimal performance."** Non-normalised is accepted but leaves the magnitude bias in. Normalise.
6. The distractors matter: 9,000 of the 19,000 ranked candidates are wrong by construction, so precision at the very top is what Hit@3 is really measuring.

The same machinery is not vision-specific. [`host-author-s-baseline.ipynb`](../faio-2025/day1/host-author-s-baseline.ipynb) uses `sentence_transformers` to rerank BM25 candidates by embedding the text of each candidate and the query into one space and sorting by cosine — identical idea, different modality. Learn it once and it covers both image retrieval and text reranking.

## [code-first]

```python
import numpy as np
import pandas as pd

D = 512

def l2(X, eps=1e-12):
    """Row-wise L2 normalise. Keepdims or the broadcast silently transposes your meaning."""
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

# E: (20000, D) embeddings from a frozen backbone (T30), rows aligned with `names`.
E = l2(E.astype(np.float32))

# After normalisation, cosine similarity IS the dot product -> one matmul scores everything.
S = E[:1000] @ E.T                      # (1000, 20000) similarity matrix, illustrative split
np.fill_diagonal(S, -np.inf)            # never retrieve the query itself
top3 = np.argsort(-S, axis=1)[:, :3]    # argsort of NEGATED scores = descending

def hit_at_k(top, truth, k=3):
    """Hit@k: credit if the true match is anywhere in the top k. Rank inside k is irrelevant."""
    return float(np.mean([t in row[:k] for row, t in zip(top, truth)]))

# Submission: exactly 20000 rows, image_name plus feature_0..feature_{D-1}.
sub = pd.DataFrame(E, columns=[f"feature_{i}" for i in range(D)])
sub.insert(0, "image_name", names)
assert len(sub) == 20000 and sub.image_name.nunique() == 20000      # no dupes, no omissions
assert np.isfinite(E).all()                                        # a single NaN poisons a row
assert np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-5)
sub.to_csv("submission.csv", index=False)
```

## [drill]

1. Define cosine similarity and give its range.
2. Both vectors are L2-normalised. Express cosine in terms of the dot product, and Euclidean distance in terms of cosine.
3. Two gallery vectors point in exactly the same direction but one is $10\times$ longer. Which does raw dot product rank higher, and which does cosine?
4. Your model ranks the true painting 2nd for 400 queries and 7th for 600. What is Hit@3?
5. Why can you not compute your own score on `Lost_in_the_Museum`, and what local check replaces it?
6. You submit 19,000 rows after dropping images that failed to load. What happens?
7. You forget `keepdims=True` in the norm. What breaks?

<details><summary>Answers</summary>

1. $\cos(\mathbf{v}, \mathbf{w}) = \dfrac{\mathbf{v} \cdot \mathbf{w}}{\lVert \mathbf{v} \rVert_2 \, \lVert \mathbf{w} \rVert_2}$, range $[-1, 1]$.
2. Cosine equals the plain dot product $\mathbf{v} \cdot \mathbf{w}$; and $\lVert \mathbf{v} - \mathbf{w} \rVert_2^2 = 2 - 2\cos(\mathbf{v}, \mathbf{w})$, so ranking by one is ranking by the other.
3. Dot product ranks the longer one higher; cosine scores them identically at 1.0 — which is what you want, since length here is brightness or contrast, not content.
4. 0.4 — only the rank-2 queries land inside the top 3.
5. The query/gallery mapping is a private server-side ground truth and you are not told which images are queries. Replace it with self-consistency: embed an image and a corrupted copy (blur, crop, jitter) and verify they are mutual near neighbours.
6. A format failure. The statement requires exactly 20,000 rows with no omissions; emit a zero or random vector for the broken files instead of dropping them.
7. `np.linalg.norm(X, axis=1)` has shape `(N,)`, which broadcasts along the *column* axis, so you divide each feature column by a different image's norm. No error is raised and every vector is wrong.

</details>

**Rep:** build `submission.csv` end to end from random vectors, run all three assertions, and confirm the file is exactly 20,000 data rows plus a header with $1 + D$ columns. Then replace the random vectors with backbone outputs from T30.

## Traps & 60-second recall

- L2-normalise, then cosine = dot product, and cosine/dot/Euclidean rankings all agree.
- Without normalisation, dot product is biased toward long vectors.
- `keepdims=True` in every row-wise norm.
- Mask the self-match before taking a top-k, or every query retrieves itself.
- `np.argsort(-S)` for descending; `argsort` is ascending by default.
- Hit@3 is a ranking metric: rank 3 scores the same as rank 1, rank 4 the same as rank 19,000.
- Exactly 20,000 rows, every `image_name`, no sorting or filtering — emit a dummy vector rather than dropping a file.
- The scorer is server-side, so your only local signal is self-consistency under corruption.
- Same machinery works for text: `sentence_transformers` reranking in `host-author-s-baseline.ipynb`.
