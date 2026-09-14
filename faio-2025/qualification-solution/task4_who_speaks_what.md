# Task 4 — Who Speaks What · why this solution

Runnable notebook: [`task4_who_speaks_what.ipynb`](./task4_who_speaks_what.ipynb)
Statement: [`../qualification/task4_Who_Speaks_What.md`](../qualification/task4_Who_Speaks_What.md)
Official reference: [`../qualification/task4_solution_Who_Speaks_What.ipynb`](../qualification/task4_solution_Who_Speaks_What.ipynb)

## What the task asks

Predict the language of a short customer-support message: `ru`, `kaz` or `eng`. You get `train.csv` with `text` and `label`, and `test.csv` with `text` only. You submit `solution.csv` with `id,label`.

## The metric is not stated — and that matters

The statement is **truncated**. It ends mid-fenced-block after the submission example and has no Evaluation section at all, so it never names a metric. The only evidence is the jury's own reference notebook, which reports `classification_report` and grid-searches on `scoring="f1_macro"`.

So the notebook optimises **macro-F1**, not accuracy. The difference is not cosmetic. Macro-F1 averages the three languages equally, so whichever language you handle worst sets your score. Accuracy lets a strong performance on a large class hide a weak one on a small class. Optimising accuracy when you are graded on macro-F1 means tuning toward the wrong answer.

## Why character n-grams

Language identification on short text is decided by letter patterns, not vocabulary. Three reasons character n-grams beat words here:

- **Kazakh-specific letters.** `ә ғ қ ң ө ұ ү һ і` appear in no Russian word. A character model picks them up as single-character features immediately.
- **Endings and clusters.** Even without a special letter, Kazakh and Russian differ in common suffixes and letter combinations. Word features cannot see inside a word; character features can.
- **Unseen words.** Support messages are short and full of names, typos and slang. A word model turns every unseen token into nothing at all; a character model still extracts signal from its letters.

`analyzer="char_wb"` keeps n-grams inside word boundaries, so it never builds a feature spanning a space — which would be noise, not signal.

The notebook cross-validates `char_wb (1,4)` against `char (2,5)` and `word (1,1)` rather than asserting a winner. Five minutes of comparison beats an argument.

**Note the disagreement with the official notebook.** It uses *word-level* TF-IDF with no `analyzer` argument and still wins. It gets away with that because its augmentation manufactures the hard word forms directly. Both are defensible; measure, do not assume.

## Why augmentation is the real trick

This is the part worth copying, and the part a beginner would never invent.

A model trained on clean data learns the cheapest rule available: *if it contains `ә`, it is Kazakh*. That rule scores well on training data and collapses on any Kazakh message that happens not to contain one of the nine letters — which is common in short messages.

The jury's notebook attacks this directly. It takes half the Kazakh rows and replaces every Kazakh-specific letter with its nearest Russian equivalent, so the model is forced to look at word shape and endings instead. It then transliterates half of *those* into Latin script, plus half the Russian rows, so the model also learns to handle Cyrillic written in Latin letters.

**Two details that are easy to get wrong:**

1. **The label never changes.** Kazakh written in Latin letters is still Kazakh. This is label-preserving augmentation. Augmentation that changes the meaning would poison the training set.
2. **It is replace-in-place, not additive.** The jury's notebook transforms each sampled half and re-concatenates, so the training set stays the *same size*. Kazakh ends up roughly 25 % de-diacriticised, 25 % de-diacriticised and Latinised, 50 % original. Doubling the data would change the class balance and is not what the reference does.

The closing cell of the official notebook confirms why this matters: part of the real test set is **code-switched** — English grammar containing Kazakh or Russian words, and the reverse. Augmentation is how you survive that without hand-labelling.

## Why logistic regression rather than a transformer

The official notebook's closing cell does suggest fine-tuning XLM-RoBERTa or Kaz-RoBERTa, and ensembling with CatBoost or LightGBM. In a four-hour round with six tasks, that is the wrong trade:

- TF-IDF plus logistic regression trains in seconds on this data size and needs no GPU.
- It reaches high macro-F1 because the problem is genuinely easy once augmentation is in place.
- A fine-tuned transformer might add a point or two, at the cost of downloading weights, debugging a training loop, and the risk of finishing with nothing.

`solver="saga"` is chosen because it handles sparse data and supports both L1 and L2 penalties. `C=10` matches the reference notebook's tuned value; `C` is **inverse** regularisation strength, so a larger `C` means less regularisation.

## Submission traps

- Labels are exactly `ru`, `kaz`, `eng` — not `kk`, not `EN`, not `Russian`.
- `id` runs `0..n-1` in `test.csv` order.
- `index=False`, or the file gains a nameless extra column.
- The filename is `solution.csv`.

The notebook asserts all four before it finishes, because a format error costs the whole task while a weak model only costs points.
