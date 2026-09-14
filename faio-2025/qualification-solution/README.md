# FAIO 2025 qualification — worked solutions for the ML tasks

Tasks 4, 5 and 6 of the 2025 qualification round: the three that need code. Tasks 1–3 are pen-and-paper maths and have no notebook here; their answers are `1.5 1 1`, `0.4213` and `0.3333`, worked out in [`../../solutions/qualification/`](../../solutions/qualification).

Each task has two files:

| Task | Notebook — run it | Write-up — why it is built that way |
|---|---|---|
| 4 · Who Speaks What | [`task4_who_speaks_what.ipynb`](./task4_who_speaks_what.ipynb) | [`task4_who_speaks_what.md`](./task4_who_speaks_what.md) |
| 5 · AI Yoga Instructor | [`task5_ai_yoga_instructor.ipynb`](./task5_ai_yoga_instructor.ipynb) | [`task5_ai_yoga_instructor.md`](./task5_ai_yoga_instructor.md) |
| 6 · Simple Objects | [`task6_simple_objects.ipynb`](./task6_simple_objects.ipynb) | [`task6_simple_objects.md`](./task6_simple_objects.md) |

## Before running anything

**The datasets are not in this repository.** Each statement links them — Google Drive for tasks 4 and 5, Yandex.Disk for task 6. Download them, put them beside the notebook, and adjust the path constants in the first code cell.

Requirements: `pandas`, `numpy`, `scikit-learn` for tasks 4 and 5; `opencv-python` and `tqdm` for task 6.

## The one-line summary of each solution

- **Task 4** — TF-IDF over character n-grams plus logistic regression. The trick that wins is *augmentation*: strip the Kazakh-specific letters from some Kazakh rows and transliterate others into Latin script, so the model cannot rely on a single letter as a shortcut. Graded on macro-F1, which the statement never states — it is truncated, and the metric comes from the jury's own reference notebook.
- **Task 5** — collapse each repetition to one row of summary features, then gradient boosting. The score is decided by validation design: at 200 Hz, a random row split scores the model against near-duplicates of its own training data and returns a meaningless 0.99.
- **Task 6** — no machine learning at all. The statement guarantees each figure is a single connected component, so counting outermost contours is the exact answer. Use `RETR_EXTERNAL`, not `RETR_LIST`, or every hollow shape counts twice.

## Scoring your own attempts

[`../../grader.md`](../../grader.md) reimplements each metric from the statements, including task 6's relative-error rule with its 0.55 floor, so you can grade a held-out split before the real thing.

## A warning about these files

These are worked solutions, useful for study and for rehearsal. They were written against the statements and the jury's reference notebook, **not run against the real datasets**, because those datasets are not in this repository. Treat every number in them as a design argument rather than a measured result, and calibrate on the training data before trusting anything.
