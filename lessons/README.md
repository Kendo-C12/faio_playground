# FAIO 2026 qualification — 5-day lesson set

**Exam: Sunday 20 September 2026, 12:00–16:00, online, 4 hours.**
FAIO = Fizmat AI Olympiad, https://faio.kz/ — 3rd edition, international.

29 files. Study days are **13, 14, 15, 16 and 19 September**; 17–18 September are off.

The day prefix in each filename is the pace signal. If it is the evening of 15 September and you have not finished the `day03_` files, you are behind.

> **Math renders on GitHub.** Formulas are written as LaTeX (`$$…$$` for display, `$…$` inline), which github.com typesets natively. In a plain text editor you will see the raw `$$` instead; VS Code needs the Markdown+Math extension, Obsidian renders it out of the box.

## Filename format

```
day03_t20_ml_tfidf_and_char_ngrams.md
 │     │   │
 │     │   └── kind: math | ml | craft
 │     └────── topic number, ties back to the rating list
 └──────────── study day
```

| Kind | Count | Meaning |
|---|---|---|
| `math` | 6 | Pen-and-paper: statistics, probability, Bayes, expected value, metric definitions |
| `ml` | 19 | Models, features, text, vision, signals — anything you fit or transform |
| `craft` | 4 | Round tactics: submissions, time budget, metric reading, speed |

Files ending `_formula_only` cover topics you rated **3** — no teaching, just formulas, memorisable values and traps.

## The 5 days

| Day | Date | Theme | Files | Total |
|---|---|---|---|---|
| 1 | Sat 13 Sep | Tabular workflow end to end | 5 | 200 min |
| 2 | Sun 14 Sep | The model family you cannot yet use | 5 | 190 min |
| 3 | Mon 15 Sep | Text path + metrics | 7 | 225 min |
| 4 | Tue 16 Sep | Vision, embeddings, signals | 7 | **335 min** |
| — | Wed 17 – Thu 18 Sep | off | — | — |
| 5 | Fri 19 Sep | RAG, expected value, exam craft + dry run | 5 | 200 min |

**Day 4 is overweight at 5 h 35 m** — every file on it is rated 0 or 1, so none can be skimmed. If 16 September cannot absorb it, move `day04_t26_ml_fft_and_spectral_features.md` and `day04_t22_ml_embeddings_and_cosine_retrieval.md` to day 5: day 4 drops to 235 min, day 5 rises to 300 min, and day 5's three `craft` files are light enough to read on exam morning.

Grand total: **1150 minutes ≈ 19 hours** over 5 days.

## How a file is built

Full lessons carry all four styles, so one file serves whichever mood you are in:

- `[concept-first]` — theory, then a worked example
- `[problem-first]` — open the anchor task, derive what it forces you to know
- `[code-first]` — runnable snippet, theory as comments
- `[drill]` — questions with answers folded below, plus one past-task rep
- `Traps & 60-second recall` — what to reread the night before

Every file opens with its **anchor task**: the real statement in this repo the topic comes from. Every worked example traces to a cited path.

## Day 1 — Sat 13 Sep · tabular workflow · 200 min

| File | Kind | Topic | Rating | Rank | Min |
|---|---|---|---|---|---|
| [t18](./day01_t18_ml_pandas_eda_and_submission_frames.md) | ml | pandas: read, groupby/agg, merge, submission frames | 2 | 1 | 50 |
| [t16](./day01_t16_ml_feature_engineering.md) | ml | Feature engineering: build, scale, encode, select | 1 | 1 | 50 |
| [t07](./day01_t07_ml_cross_validation_and_leakage.md) | ml | **Cross-validation and leakage** | 1 | 1 | 50 |
| [t31](./day01_t31_craft_submission_discipline.md) | craft | Submission discipline: filename, header, row count, order | 2 | 1 | 40 |
| [t01](./day01_t01_math_descriptive_statistics_formula_only.md) | math | Descriptive statistics — formula sheet | 3 | 1 | 10 |

T07 was re-rated from 3 to 1 and is now a full lesson, not a sheet. It is the one that stops you trusting a fake 0.99 on task 5.

## Day 2 — Sun 14 Sep · the model family · 190 min

| File | Kind | Topic | Rating | Rank | Min |
|---|---|---|---|---|---|
| [t12](./day02_t12_ml_gradient_boosting.md) | ml | **Gradient boosting** — from zero | 0 | 1 | 55 |
| [t10](./day02_t10_ml_knn_and_decision_trees.md) | ml | kNN and decision trees | 1 | 2 | 45 |
| [t11](./day02_t11_ml_random_forest_and_bagging.md) | ml | Random forest and bagging | 1 | 2 | 45 |
| [t19](./day02_t19_ml_hyperparameter_search.md) | ml | Hyperparameter search | 2 | 1 | 35 |
| [t08](./day02_t08_ml_logistic_regression_formula_only.md) | ml | Logistic regression — formula sheet | 3 | 1 | 10 |

Start with t12: the only rank-1 topic you rated 0 here, and boosting is the default first model on tabular data in a timed round.

## Day 3 — Mon 15 Sep · text path + metrics · 225 min

| File | Kind | Topic | Rating | Rank | Min |
|---|---|---|---|---|---|
| [t20](./day03_t20_ml_tfidf_and_char_ngrams.md) | ml | TF-IDF, bag of words, char n-grams | 1 | 1 | 50 |
| [t21](./day03_t21_ml_text_augmentation_and_normalisation.md) | ml | **Text augmentation and normalisation** | 0 | 1 | 50 |
| [t23](./day03_t23_ml_pretrained_transformers.md) | ml | Pretrained transformers | 1 | 1 | 55 |
| [t06](./day03_t06_math_custom_and_competition_metrics.md) | math | Custom and competition metrics | 2 | 1 | 40 |
| [t02](./day03_t02_math_classical_probability_formula_only.md) | math | Classical probability — formula sheet | 3 | 1 | 10 |
| [t03](./day03_t03_math_bayes_and_conditional_probability_formula_only.md) | math | Bayes & conditional probability — formula sheet | 3 | 1 | 10 |
| [t05](./day03_t05_math_classification_metrics_formula_only.md) | math | Classification metrics — formula sheet | 3 | 1 | 10 |

t20 + t21 together rebuild the 2025 qualification's task 4. t21 is the augmentation trick the shipped solution notebook actually wins with.

## Day 4 — Tue 16 Sep · vision, embeddings, signals · 335 min

| File | Kind | Topic | Rating | Rank | Min |
|---|---|---|---|---|---|
| [t27](./day04_t27_ml_image_basics_and_thresholding.md) | ml | Image arrays and thresholding | 1 | 1 | 45 |
| [t28](./day04_t28_ml_contours_and_connected_components.md) | ml | **Contours and connected components** | 0 | 1 | 50 |
| [t29](./day04_t29_ml_shape_descriptors.md) | ml | Shape descriptors | 0 | 1 | 45 |
| [t30](./day04_t30_ml_cnn_transfer_learning.md) | ml | CNN transfer learning | 1 | 2 | 50 |
| [t22](./day04_t22_ml_embeddings_and_cosine_retrieval.md) | ml | Embeddings and cosine retrieval | 0 | 2 | 50 |
| [t25](./day04_t25_ml_sensor_window_features.md) | ml | Sensor window features | 0 | 1 | 45 |
| [t26](./day04_t26_ml_fft_and_spectral_features.md) | ml | FFT and spectral features | 0 | 2 | 50 |

t27 + t28 + t29 rebuild task 6, and t28 alone solves it. t25 rebuilds task 5.

## Day 5 — Fri 19 Sep · retrieval and exam craft · 200 min

| File | Kind | Topic | Rating | Rank | Min |
|---|---|---|---|---|---|
| [t24](./day05_t24_ml_rag_and_bm25_retrieval.md) | ml | RAG and BM25 retrieval | 1 | 2 | 55 |
| [t04](./day05_t04_math_expected_value.md) | math | Expected value | 2 | 2 | 35 |
| [t32](./day05_t32_craft_time_budget_for_a_4_hour_round.md) | craft | Time budget for a 4-hour round | 0 | 1 | 35 |
| [t33](./day05_t33_craft_reading_the_metric_for_cheap_points.md) | craft | Reading the metric for cheap points | 0 | 1 | 35 |
| [t34](./day05_t34_craft_speed_on_large_inputs.md) | craft | Speed on large inputs | 0 | 1 | 40 |

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

Five topics fell below the rank cut and have no file: **t09** least squares, **t13** SVM, **t14** k-means, **t17** outlier detection (IQR / z-score), **t15** PCA. The first four are anchored only in `faio-2024/day1/`; PCA only in mlcourse.ai. If a day frees up, **t17 is the one worth adding** — cheap to learn and plausible in any data-cleaning task.

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
