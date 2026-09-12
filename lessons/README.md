# FAIO 2026 qualification — 5-day lesson set

**Exam: Sunday 20 September 2026, 12:00–16:00, online, 4 hours.**
FAIO = Fizmat AI Olympiad, https://faio.kz/ — 3rd edition, international.

29 files. Study days are **13, 14, 15, 16 and 19 September**; 17–18 September are off.

The day prefix in each filename is the pace signal. If it is the evening of 15 September and you have not finished the `day3-` files, you are behind.

## The 5 days

| Day | Date | Theme | Files | Real total |
|---|---|---|---|---|
| 1 | Sat 13 Sep | Tabular workflow end to end | 5 | 160 min |
| 2 | Sun 14 Sep | The model family you cannot yet use | 5 | 190 min |
| 3 | Mon 15 Sep | Text path + metrics | 7 | 225 min |
| 4 | Tue 16 Sep | Vision, embeddings, signals | 7 | **335 min** |
| — | Wed 17 – Thu 18 Sep | off | — | — |
| 5 | Fri 19 Sep | RAG, expected value, exam craft + dry run | 5 | 200 min |

**Day 4 is overweight at 5 h 35 m** — the per-file estimates came out higher than planned, and every file on that day is rated 0 or 1 so none can be skimmed. If 16 September cannot absorb it, move `day4-t26-fft-and-spectral-features.md` and `day4-t22-embeddings-and-cosine-retrieval.md` to day 5: day 4 drops to 235 min and day 5 rises to 300 min, but day 5's three exam-craft files are light and can be read the morning of the exam.

Grand total: **1110 minutes ≈ 18.5 hours** over 5 days.

## How a file is built

Full lessons carry all four styles so one file serves whichever mood you are in:

- `[concept-first]` — theory, then a worked example
- `[problem-first]` — open the anchor task, derive what it forces you to know
- `[code-first]` — runnable snippet, theory as comments
- `[drill]` — questions with answers folded below, plus one past-task rep
- `Traps & 60-second recall` — what to reread the night before

Files ending `-formula-only` cover topics you rated **3** (can implement from scratch). No teaching, no derivation — formulas, memorisable values, one-line traps, under 40 lines each.

Every file opens with its **anchor task**: the real statement in this repo that the topic comes from. Every worked example traces to a cited path.

## Day 1 — Sat 13 Sep · tabular workflow · 160 min

| File | Topic | Rating | Rank | Min |
|---|---|---|---|---|
| [day1-t18](./day1-t18-pandas-eda-and-submission-frames.md) | pandas: read, groupby/agg, merge, submission frames | 2 | 1 | 50 |
| [day1-t16](./day1-t16-feature-engineering.md) | Feature engineering: build, scale, encode, select | 1 | 1 | 50 |
| [day1-t31](./day1-t31-submission-discipline.md) | Submission discipline: filename, header, row count, order | 2 | 1 | 40 |
| [day1-t01](./day1-t01-descriptive-stats-formula-only.md) | Descriptive statistics — formula sheet | 3 | 1 | 10 |
| [day1-t07](./day1-t07-cross-validation-and-leakage-formula-only.md) | Cross-validation & leakage — formula sheet | 3 | 1 | 10 |

## Day 2 — Sun 14 Sep · the model family · 190 min

| File | Topic | Rating | Rank | Min |
|---|---|---|---|---|
| [day2-t12](./day2-t12-gradient-boosting.md) | **Gradient boosting** — from zero | 0 | 1 | 55 |
| [day2-t10](./day2-t10-knn-and-decision-trees.md) | kNN and decision trees | 1 | 2 | 45 |
| [day2-t11](./day2-t11-random-forest-and-bagging.md) | Random forest and bagging | 1 | 2 | 45 |
| [day2-t19](./day2-t19-hyperparameter-search.md) | Hyperparameter search | 2 | 1 | 35 |
| [day2-t08](./day2-t08-logistic-regression-formula-only.md) | Logistic regression — formula sheet | 3 | 1 | 10 |

Start with t12. It is the only rank-1 topic you rated 0 in this group, and boosting is the default first model on tabular data in a timed round.

## Day 3 — Mon 15 Sep · text path + metrics · 225 min

| File | Topic | Rating | Rank | Min |
|---|---|---|---|---|
| [day3-t20](./day3-t20-tfidf-and-char-ngrams.md) | TF-IDF, bag of words, char n-grams | 1 | 1 | 50 |
| [day3-t21](./day3-t21-text-augmentation-and-normalisation.md) | **Text augmentation and normalisation** | 0 | 1 | 50 |
| [day3-t23](./day3-t23-pretrained-transformers.md) | Pretrained transformers | 1 | 1 | 55 |
| [day3-t06](./day3-t06-custom-and-competition-metrics.md) | Custom and competition metrics | 2 | 1 | 40 |
| [day3-t02](./day3-t02-classical-probability-formula-only.md) | Classical probability — formula sheet | 3 | 1 | 10 |
| [day3-t03](./day3-t03-bayes-and-conditional-probability-formula-only.md) | Bayes & conditional probability — formula sheet | 3 | 1 | 10 |
| [day3-t05](./day3-t05-classification-metrics-formula-only.md) | Classification metrics — formula sheet | 3 | 1 | 10 |

t20 + t21 together rebuild the 2025 qualification's task 4. t21 is the augmentation trick the shipped solution notebook actually wins with.

## Day 4 — Tue 16 Sep · vision, embeddings, signals · 335 min

| File | Topic | Rating | Rank | Min |
|---|---|---|---|---|
| [day4-t27](./day4-t27-image-basics-and-thresholding.md) | Image arrays and thresholding | 1 | 1 | 45 |
| [day4-t28](./day4-t28-contours-and-connected-components.md) | **Contours and connected components** | 0 | 1 | 50 |
| [day4-t29](./day4-t29-shape-descriptors.md) | Shape descriptors | 0 | 1 | 45 |
| [day4-t30](./day4-t30-cnn-transfer-learning.md) | CNN transfer learning | 1 | 2 | 50 |
| [day4-t22](./day4-t22-embeddings-and-cosine-retrieval.md) | Embeddings and cosine retrieval | 0 | 2 | 50 |
| [day4-t25](./day4-t25-sensor-window-features.md) | Sensor window features | 0 | 1 | 45 |
| [day4-t26](./day4-t26-fft-and-spectral-features.md) | FFT and spectral features | 0 | 2 | 50 |

t27 + t28 + t29 rebuild task 6, and t28 alone solves it. t25 rebuilds task 5.

## Day 5 — Fri 19 Sep · retrieval and exam craft · 200 min

| File | Topic | Rating | Rank | Min |
|---|---|---|---|---|
| [day5-t24](./day5-t24-rag-and-bm25-retrieval.md) | RAG and BM25 retrieval | 1 | 2 | 55 |
| [day5-t04](./day5-t04-expected-value.md) | Expected value | 2 | 2 | 35 |
| [day5-t32](./day5-t32-time-budget-for-a-4-hour-round.md) | Time budget for a 4-hour round | 0 | 1 | 35 |
| [day5-t33](./day5-t33-reading-the-metric-for-cheap-points.md) | Reading the metric for cheap points | 0 | 1 | 35 |
| [day5-t34](./day5-t34-speed-on-large-inputs.md) | Speed on large inputs | 0 | 1 | 40 |

No new heavy theory. Read t32 last and run a timed mock against the 2025 qualification statements.

## Exam-probability rank

Which archived round a topic comes from, and therefore how likely it is to appear on 20 September.

| Rank | Source | Meaning |
|---|---|---|
| 1 | `faio-2025/qualification/` | The round being sat. 22 topics. |
| 2 | IOAI 2024–2026, or a skill the rank-1 tasks implicitly require | 7 topics. |
| 3 | only `faio-2024/day1/` | A round whose from-scratch style 2025 dropped. |
| 4 | only `faio-2025/day1-2` | Finals, not qualification. |
| 5 | only mlcourse.ai | No FAIO task needs it. |

**Only rank 1–2 topics are in these 5 days.** Source priority was set as: 2025 qualification → IOAI 2024–2026 → 2024 day 1 → 2025 day 1–2 → mlcourse.ai. `faio-2024/qualification` is excluded entirely: it was Yandex.Contest competitive programming with 1 s / 64 MB limits, and 2025 dropped that format, so it does not predict 2026.

Nothing landed at rank 4 — every 2025 day-1/day-2 topic (embeddings, RAG, FFT, CNN transfer) also serves a qualification task or IOAI 2026, so each rose to rank 2.

## Not written yet — rank 3+ backlog

Five topics fell below the rank cut and have no file: **t09** least squares, **t13** SVM, **t14** k-means, **t17** outlier detection (IQR / z-score), **t15** PCA. All four of the first are anchored only in `faio-2024/day1/`; PCA only in mlcourse.ai. If a day frees up, **t17 is the one worth adding** — cheap to learn and plausible in any data-cleaning task.

## Known defects in the archive statements

Found while writing these lessons. Each is flagged in the file that touches it.

1. **`task6_Simple_Objects.md`'s metric is inverted.** `Final Verdict = 0, if Accuracy > 0.55` would zero out good solutions. Intended: 0 *below* 0.55. Solve the intent; never exploit the literal text.
2. **`task6`'s output spec contradicts itself.** Output says "a single integer in each row", the next sentence refers to "the order of `id` values". Its "Important" paragraph is copy-pasted from task 5 and names `test.csv` / `solution.csv` in a task whose file is `submit.csv`.
3. **Neither task 5 nor task 6 ships a `test.csv`**, despite both referring to "the order in `test.csv`". Task 5 gives `X_test.csv`; task 6 gives test images and a `submit.csv` template. Copy the shipped sample submission rather than trusting the prose.
4. **`task4_Who_Speaks_What.md` is truncated** — it ends mid-fenced-block and states **no metric at all**. The shipped notebook grades `classification_report` and tunes `scoring="f1_macro"`, so treat macro-F1 as the real metric.
5. **`task5`'s example contradicts its prose** — text says a 200 Hz time series per `id`, the example shows one row per `id`. First command of the round: `X_test.groupby("id").size().max()`.
6. **`Lost_in_the_Museum.md` disagrees with itself twice** — prose wants `image_name` + `feature_*` while its example header adds a leading `ID`; and the Overview says a 10,000-image gallery while Evaluation ranks against all 19,000.

## Where the statements live

- [`faio-2025/qualification/`](../faio-2025/qualification) — the 6 tasks of the round this set targets
- [`faio-2025/day1/`](../faio-2025/day1), [`faio-2025/day2/`](../faio-2025/day2) — 2025 finals, plus the host baseline notebook
- [`faio-2024/day1/`](../faio-2024/day1) — 2024 main round
- [`docs/faio-2025-problems.md`](../docs/faio-2025-problems.md), [`docs/faio-2024-problems.md`](../docs/faio-2024-problems.md) — category and format breakdown per problem
- [`solutions/qualification/`](../solutions/qualification) — worked notes for all 6 tasks of the 2025 qualification

External references, both recommended by faio.kz (neither is reachable from this repo's build environment — open them in a browser): [mlcourse.ai](https://mlcourse.ai/book/index.html) and [IOAI 2026 Individual Contest](https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest).
