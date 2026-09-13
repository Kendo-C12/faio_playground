# T24 · RAG: retrieval, BM25, rerank, then generate — Day 5

**Anchor task(s):**
- [`faio-2025/day1/Who_is_the_best_pitcher.md`](../faio-2025/day1/Who_is_the_best_pitcher.md)
- [`faio-2025/day1/host-author-s-baseline.ipynb`](../faio-2025/day1/host-author-s-baseline.ipynb)

Rating 1 · **exam-probability rank 2** · ~55 min

The one FAIO task built entirely on an LLM. Read the host baseline once and you own the whole pattern: index → retrieve → rerank → let the model pick.

## [concept-first]

**Why retrieve at all.** [`Who_is_the_best_pitcher.md`](../faio-2025/day1/Who_is_the_best_pitcher.md) defines RAG in two halves: "Retrieval — fetching the most relevant, up-to-date information from a database, API, or knowledge source" and "Generation — using a large language model to understand the question and generate a natural, useful answer based on the retrieved facts." The reason the first half exists is brutally practical: `players.json` + `teams.json` + `league.json` flatten to tens of millions of values. No context window holds that, and even if one did, the answer would drown. The baseline's own comment: "The main challenge in this task is filtering out a large amount of data before even starting modeling." Retrieval is not a nicety, it is the task.

**Flattening is the indexing trick.** The notebook gives it its own heading, "flatten json (most important step)". `flatten_json.flatten(obj, ".")` turns nested dicts and lists into one flat dict where each key is a dotted path:

```
seasons.0.teams.0.total.assists  ->  3
```

Now every atomic fact is a `(path, value)` pair, and the path is *readable English-ish text*: `miles_mikolas.seasons.2025.REG.pitching.innings`. That is the move — a retrieval problem over text, not a traversal problem over JSON. Retrieve the path, and the value comes with it.

**BM25 = lexical retrieval.** Score a document $d$ against a query $q$ term by term:

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \, (k_1 + 1)}{f(t, d) + k_1 \left(1 - b + b \dfrac{\lvert d \rvert}{\text{avgdl}}\right)}
$$

Three ingredients:

- *IDF*: rare query terms (`mikolas`) count far more than common ones (`what`).
- *Term-frequency saturation*: the $f(t, d) / (f(t, d) + k_1 \cdots)$ shape means the 10th occurrence of a word adds almost nothing over the 3rd. Plain TF-IDF has no such ceiling, which is why a long repetitive document can beat a short relevant one there and not here.
- *Document length normalisation* (the $b$ parameter): a long document matching a term is less impressive than a short one matching it. Paths are short, so this mostly protects you from the few very deep ones.

*Stemming* matters because a path says `pitching` and the question says `pitch`, or `wins` versus `win`. `Stemmer.Stemmer("english")` maps both sides to a common root, so the term matches at all. Without it BM25 silently scores 0 on a perfectly relevant path.

**Dense retrieval = embeddings.** A sentence-transformer maps query and document into one vector space; similarity is cosine:

$$
\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\lVert \mathbf{q} \rVert_2 \, \lVert \mathbf{d} \rVert_2}
$$

It catches paraphrase (`how many innings` $\approx$ `ip`, `primary position` $\approx$ `position.primary`) that BM25 cannot, because BM25 only sees shared tokens. It is slower, and it needs the whole candidate set encoded.

**So: two stages.** Lexical first because it is cheap over millions of rows; dense second because it is accurate over thousands. This is retrieve-then-rerank, and it is the default architecture for a reason:

| Stage | Tool | Input size | Output |
|---|---|---|---|
| Retrieve | BM25 | millions of paths | ~2000 candidates |
| Rerank | sentence-transformer cosine | ~2000 | top 40 |
| Decide | LLM | 40 `(path, value)` pairs | one value |

**Then generate.** The LLM is not asked to know baseball. It is asked to choose, among 40 candidate pairs, the one that answers the question, and echo its value verbatim. That is a much easier job than recall, and it is why RAG works with a small 4B model.

## [problem-first]

Open [`Who_is_the_best_pitcher.md`](../faio-2025/day1/Who_is_the_best_pitcher.md). Four statements in it dictate the whole design.

1. "Root-level keys are entity names (e.g., player name or team name)" → the entity in the question *is* the top-level key. So extract the entity, and you can slice the corpus down to one prefix before retrieving anything. The baseline does exactly this and reports the players corpus dropping "from 35 million rows to 3.8 million".
2. "league.json — Top-level (no league root): metadata, teams, standings, leaders" → injuries, standings and venues live in `league`, not under the player. So routing a question to the right *file* is a separate decision from retrieving within it. The baseline hand-labels each question with a `prefix`, using `league` for the injury questions.
3. "The ANSWER should exactly match the original value from the JSON file… Do not round, truncate, or change formatting" → you must return the *stored string*, not a computed number. This forbids arithmetic and forbids letting the LLM rephrase. Hence the system prompt's "Respond with ONLY the exact value from that chosen path."
4. "Submitting this baseline file guarantees a non-zero score" — 5 correct rows, the rest `"no answer"`. Ship that first, then improve. The metric is plain accuracy, $\text{correct} / \text{total}$, so every question is worth the same and there is no penalty for a wrong guess: **never leave a row blank**.

What the statement does *not* give you is how to find entity names. The notebook is honest about it: "We can extract entity names from the CSV manually, with regex, using NER models, or via free web APIs… Reviewing 220 questions will take approximately 15 minutes." Under exam time pressure, 15 minutes of manual labelling that unlocks a $10\times$ corpus reduction is a good trade.

## [code-first]

The baseline's pipeline, compressed. Cell order preserved.

```python
# 1. Load. Watch RAM explicitly — the notebook prints psutil RSS after each file
import json, os, gc, psutil, pandas as pd, numpy as np
mem = lambda: psutil.Process(os.getpid()).memory_info().rss / 1024**3
loaded = {n: json.load(open(p)) for n, p in paths.items()}   # league/players/teams

# 2. Flatten: every leaf value gets one dotted path. THE step.
from flatten_json import flatten
flat = {n: flatten(o, ".") for n, o in loaded.items()}

# 3. (path, value) long frame, then free the dicts
to_df = lambda d: pd.DataFrame({"paths": list(d), "values": list(d.values())})
players_df, teams_df, league_df = (to_df(flat[k]) for k in ("players", "teams", "league"))
del flat; gc.collect()

# 4. Prune paths that can never be an answer: ids, comments, schema skeletons
for df in (players_df, teams_df, league_df):
    df.drop(df.index[df.paths.str.endswith((".id", "_comment"))
                     | df.paths.str.startswith("skeletons.")], inplace=True)

# 5. Entity prefix per question (manual/regex/NER) -> slice to one root key
players_df = players_df[players_df.paths.str.split(".").str[0].isin(entities)]

# 6. BM25 over the slice. Stem both sides or half the matches vanish.
import bm25s, Stemmer
stemmer = Stemmer.Stemmer("english")
def bm25_search(docs, query, k=2000):
    tokens = bm25s.tokenize(docs, stopwords="en", stemmer=stemmer)
    r = bm25s.BM25(); r.index(tokens)
    idx, _ = r.retrieve(bm25s.tokenize(query, stemmer=stemmer), k=min(k, len(docs)))
    return [docs[i] for i in idx[0]]

# 7. Rerank the BM25 survivors densely. normalize_embeddings=True -> dot == cosine.
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("bge-large-en-v1.5", device="cuda")
emb = model.encode(cand_paths, batch_size=100, normalize_embeddings=True)
q = model.encode(question, normalize_embeddings=True)
top40 = np.array(cand_paths)[np.argsort(emb @ q)[-40:][::-1]]

# 8. LLM picks one of the 40 and echoes its value verbatim
# Gemma3ForConditionalGeneration + AutoProcessor, 4-bit via BitsAndBytesConfig,
# do_sample=False (greedy — you want reproducibility, not creativity)

# 9. Submission: header ID,ANSWER, one row per test ID, answers as strings
submission[["ID", "ANSWER"]].set_index("ID").to_csv("submission.csv")
```

The notebook's closing line is the roadmap for the remaining hours: "We can further improve accuracy by refining the prompt, experimenting with different LLMs or embeddings for the reranker, and tuning BM25 parameters." Prompt refinement is first because it costs one cell re-run, not a re-encode of 3.8 million paths.

## [drill]

1. Why does flattening to dotted paths make retrieval possible, when the same values were already present in the nested JSON?
2. BM25 has term-frequency saturation. Which failure of plain TF-IDF does that fix?
3. You skip the stemmer. Question says "How many wins", the path says `...pitching.win`. What happens, and what is the BM25 score contribution of that term?
4. Why run BM25 *before* the sentence-transformer rather than the other way round?
5. The LLM is given 40 candidates. What breaks if you give it 2000 instead, and what breaks if you give it 3?
6. The statement says JSON value `123.4500` must be returned as `"123.4500"`. Which design decision does that force on the generation step?

<details><summary>Answers</summary>

1. Because each leaf becomes an independent *document* with a short descriptive text key. Retrieval scores documents; nested JSON has no documents, only a tree you would have to traverse with logic you do not have.
2. A long document that repeats a query term many times outranking a short, genuinely relevant one. Saturation caps the reward per term; the $b$ length-normalisation term handles the length side.
3. The terms do not match at all, so that term contributes 0 and the correct path may not even enter the top-2000. Stemming maps `wins`/`win` to one root.
4. Cost. BM25 is a sparse lexical index and runs over millions of rows; encoding millions of paths with a transformer is infeasible in a round. Cheap-and-broad first, expensive-and-precise on the survivors.
5. 2000 candidates will not fit the context, and recall of the right one among that much noise drops; 3 candidates risks the correct path not being there at all — the LLM cannot recover what retrieval dropped. Top-k is a recall/precision dial.
6. The model must copy the retrieved value verbatim, never restate or compute it — hence a prompt that says "ONLY the exact value from that chosen path", greedy decoding, and no post-processing of numbers.

</details>

**Rep:** take 5 questions from the baseline's `l2` list, flatten a small nested dict by hand to dotted paths, and write which path each question should retrieve and which of the three files it lives in.

## Traps & 60-second recall

- Flatten first. Everything downstream is text retrieval over `(path, value)` pairs.
- Filter the corpus by entity prefix before retrieving — it is a $10\times$ win and costs one `.str.split(".").str[0]`.
- Stem query and documents with the *same* stemmer, or matches silently vanish.
- BM25 for recall over millions, dense rerank for precision over thousands, LLM for the final pick.
- `normalize_embeddings=True`, then a dot product *is* cosine — no division, no bug.
- Drop `.id`, `_comment` and `skeletons.*` paths: they can never be the answer and they crowd the top-k.
- Accuracy has no penalty for wrong guesses, so emit a row for every ID; ship the provided baseline submission before tuning anything.
- Answers are strings, copied exactly. No rounding, no reformatting, greedy decoding.
- Cheapest remaining gain is the prompt, then the embedding model, then BM25 parameters — in that order.
