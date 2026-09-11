# FAIO 2025 — problem index

FAIO = **Fizmat AI Olympiad**, official site https://faio.kz/

Classification of every problem in [`faio-2025/`](../faio-2025), taken from the statements in this repo.

**11 problems total:** 6 qualification + 3 day 1 + 2 day 2.

| Category | Count |
|---|---|
| `math` | 3 |
| `competitive-programming` | 0 |
| `code (ML implementation)` | 8 |
| `ml-theory` | 0 |

Judges: Yandex.Contest 81908 for the qualification; Kaggle for day 1 and day 2. Kazakh and Russian translations of both main rounds are in the round PDFs (`1st-round-FAIO-2025-KZ-RU.pdf`, `2nd-round-FAIO-2025-KZ-RU.pdf`); the English `.md` files are translations, so the PDF wins on exact wording.

## Category legend

| Category | Meaning | Marker in the statement |
|---|---|---|
| `math` | Pen-and-paper. Every number is fixed in the statement, the answer is one value or a short tuple, no program needed. | "Answer Format: output three numbers", no dataset |
| `competitive-programming` | Classic CP: stated time and memory limit, stdin/stdout, exact-match auto-judge. | `Time limit`, `Memory limit`, `standard input or input.txt` |
| `code (ML implementation)` | Implement or train a model, graded by a metric on hidden data. | accuracy / Hit@3 / custom metric, `train.csv`, a submission CSV |
| `ml-theory` | Conceptual ML in prose — which model and why, metric trade-offs. No code. | — |

Two notes on the empty buckets:

- **No `competitive-programming` problems at all in 2025.** The format changed: 2024's qualification was Yandex CP with `1 s / 64 MB` limits, while 2025 keeps only short math answers plus metric-scored ML. See [the 2024 index](./faio-2024-problems.md).
- **No `ml-theory` problems either.** Tasks 5 and 6 come closest — both say "Source code is not required. Submit only the CSV" — but the predictions cannot be produced by hand, so they are `code (ML implementation)`. Nothing in the archive grades written reasoning.

## Qualification — 6 tasks

Directory: [`faio-2025/qualification/`](../faio-2025/qualification). Yandex.Contest 81908, problems 1–6. No time or memory limits stated anywhere.

| # | Task | File | Category | Topic | Input | Output | Scoring |
|---|---|---|---|---|---|---|---|
| 1 | Statistics 101? | [`task1_stats_101.md`](../faio-2025/qualification/task1_stats_101.md) | `math` | Descriptive statistics | 10 values listed in the statement: `2 0 1 3 2 1 0 4 1 1` | Three numbers on one line: mean, mode, median | 10 points for the correct answer |
| 2 | Probability Theory 101 | [`task2_Probability_theory_101.md`](../faio-2025/qualification/task2_Probability_theory_101.md) | `math` | Probability | Three rolls of a fair 6-sided die | Probability | 10 points |
| 3 | Medicine | [`task3_Medicine.md`](../faio-2025/qualification/task3_Medicine.md) | `math` | Bayes' theorem | Prevalence 1 %, sensitivity 99 %, specificity 98 % | `P(sick \| positive)` as a decimal to 4 places, e.g. `0.3147` | 10 points |
| 4 | Who Speaks What? | [`task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md) | `code (ML implementation)` | NLP, language ID | `train.csv` (`text`, `label` ∈ `ru`/`kaz`/`eng`), `test.csv` (`text`) | `solution.csv` with `id,label` | Not stated in the file; accuracy implied. A reference notebook ships in the repo: [`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb) |
| 5 | Can You Become an AI Yoga Instructor? | [`task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md) | `code (ML implementation)` | Time-series / IMU classification | `X_train.csv`, `y_train.csv` (`0` incorrect, `1` correct), `X_test.csv` | `solution.csv`; "source code is not required" | Accuracy |
| 6 | Simple Objects | [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md) | `code (ML implementation)` | Computer vision, counting | 2048×2048 RGB blueprints, 100–500 figures each (triangle / rectangle / circle), 8000 test images | `submit.csv`, one integer count per row | Customized accuracy: `Accuracy = 1 − min(1, abs(ŷ−y)/y)`, then rescaled from a 0.55 floor |

**Rollup:** 3 `math`, 3 `code (ML implementation)`, 0 `competitive-programming`, 0 `ml-theory`.
By topic: probability 2 (2, 3), statistics 1 (1), NLP 1 (4), sensor time series 1 (5), vision 1 (6).

### Solving notes

- **1** — mean `17/10 = 1.7`, mode `1` (appears four times), median `1` (average of the 5th and 6th of the sorted list). Print in the stated order: mean, mode, median.
- **2** — the statement gives the setup (three rolls) and the theory section defines classical probability; read the exact event being asked before counting. With 216 equally likely outcomes, count favourable outcomes rather than multiplying probabilities from memory.
- **3** — `P(sick|+) = 0.01·0.99 / (0.01·0.99 + 0.99·0.02) ≈ 0.3333`. Note specificity is 98 %, so the false-positive rate is 2 % — this is the step most people get wrong. Round to 4 decimals as instructed.
- **4** — character n-gram TF-IDF plus logistic regression separates ru / kaz / eng almost perfectly; Kazakh-specific Cyrillic letters (ә, ғ, қ, ң, ө, ұ, ү, һ, і) are the strongest signal against Russian. The shipped notebook is the reference if you want to compare.
- **5** — IMU segments: derive per-segment features (mean, sd, min, max, energy, correlation between axes) and use gradient boosting; a raw per-timestep model overfits at this data size. Check whether rows in `X_test.csv` are one segment each or many before aggregating.
- **6** — classical CV beats a trained model here: threshold the background, find contours, and filter by area, then classify the shape by its vertex count if you want a per-type breakdown. The metric is relative error on the count, so systematic over- or under-counting hurts more than noise; the 0.55 floor means a careless solution scores exactly 0.

## Day 1 (1st round) — 3 problems

Directory: [`faio-2025/day1/`](../faio-2025/day1). All hosted on Kaggle. A host baseline notebook ships in the repo: [`host-author-s-baseline.ipynb`](../faio-2025/day1/host-author-s-baseline.ipynb).

| # | Problem | File | Category | Topic | Data | Submission | Scoring |
|---|---|---|---|---|---|---|---|
| 1 | Missing Fundamental Puzzle | [`Missing_Fundamental_Puzzle.md`](../faio-2025/day1/Missing_Fundamental_Puzzle.md) | `code (ML implementation)` | Audio / signal ML | Mono 16-bit PCM WAV, 22050 Hz, 1–4 s, fundamental removed; 61 pitch classes (TinySOL-MF) | CSV `Path, Pitch_ID` | Classification accuracy. `kaggle competitions download -c missing-fundamental-puzzle` |
| 2 | QAZ Letters | [`qaz_letters.md`](../faio-2025/day1/qaz_letters.md) | `code (ML implementation)` | Tabular classification from engineered CV features | `train.csv` / `test.csv` of handcrafted features (moments, Hu moments, contours, radial, Fourier descriptors, projections, zoning, DCT, wavelets), `labels.csv`; 42 Kazakh characters | `sample_submission.csv` format | Not stated in the file; accuracy implied. `kaggle competitions download -c qaz-letters` |
| 3 | Who is the best pitcher | [`Who_is_the_best_pitcher.md`](../faio-2025/day1/Who_is_the_best_pitcher.md) | `code (ML implementation)` | LLM / RAG question answering | Large heterogeneous MLB JSON endpoints, `test.csv` of questions | `submission.csv`, one row per `ID` | Accuracy = correct answers / total. A provided baseline submission guarantees a non-zero score. `kaggle competitions download -c who-s-the-best-pitcher-v2` |

**Rollup:** 3 `code (ML implementation)`. No `math`, no `competitive-programming`, no `ml-theory`.

### Solving notes

- **Missing Fundamental** — the fundamental is gone but the harmonics are still spaced by `f0`, so the spacing recovers the pitch. Compute a spectrum (or cepstrum / autocorrelation of the spectrum) and look at harmonic spacing rather than the lowest visible peak; a CNN over a log-mel spectrogram also works given 61 classes. Do not trust the lowest-energy bin — that is exactly what was deleted.
- **QAZ Letters** — no images, only features, so this is a tabular problem: gradient boosting over the given columns, and treat the rotation- and scale-invariant descriptors as the strongest features. With 42 classes, check for confusable pairs that differ only by a diacritic and consider features that survive that distinction.
- **Who is the best pitcher** — ingest and normalise the JSON endpoints into queryable tables first, then retrieve and answer. Most of the score is in the ingestion and question-to-field mapping, not in the model. Start by submitting the baseline file to prove the pipeline works, then improve.

## Day 2 (2nd round) — 2 problems

Directory: [`faio-2025/day2/`](../faio-2025/day2). Both on Kaggle.

| # | Problem | File | Category | Topic | Data | Submission | Scoring |
|---|---|---|---|---|---|---|---|
| 1 | HearMe — Personalized Music Recommender | [`HearMe_Personalized_Music_Recommender.md`](../faio-2025/day2/HearMe_Personalized_Music_Recommender.md) | `code (ML implementation)` | Recommender system | User interaction history plus track and user aggregate features; test period 2025-09-01 → 2025-09-15, no cold-start users | CSV with `id`, `user_id`, `item_id`, `rank` 1–50 | Per-user sum of listened fractions over the 50 tracks (0–50), averaged over users, normalised to 0–1. `kaggle competitions download -c hear-me-personalized-music-recommender` |
| 2 | Lost in the Museum | [`Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md) | `code (ML implementation)` | Cross-domain image retrieval | 1,000 visitor photos (shadowed, zoomed, poorly lit) against 10,000 pristine paintings; no text, no metadata | `submission.csv`, exactly 20,000 rows: `image_name` plus `feature_0 … feature_D-1` embeddings (256 or 512 recommended) | Hit@3, computed server-side from your embeddings. `kaggle competitions download -c lost-in-the-museum-v2` |

**Rollup:** 2 `code (ML implementation)`.

### Solving notes

- **HearMe** — the metric rewards listened fraction, not just a click, so rank by expected listening depth rather than raw popularity. A popularity baseline plus per-user history filtering gets the first 0.25; collaborative filtering on the interaction matrix goes further. Submit exactly 50 ranked rows per test user.
- **Lost in the Museum** — you submit embeddings, not predictions, so the whole task is building an embedding space robust to lighting, crop and viewpoint. A pretrained vision backbone with strong augmentation, L2-normalised output, is the baseline; the row count is fixed at 20,000 with no duplicates or omissions, so validate the file shape before uploading.

## See also

- [FAIO 2024 problem index](./faio-2024-problems.md) — where the `competitive-programming` problems are
- Upstream source of the task files: https://github.com/abglnv/faio-tasks
