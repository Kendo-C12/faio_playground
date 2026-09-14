# Crash course — pick a model, implement it, move on

No theory. No derivations. Every page answers one question: **what do I type to get a working submission?**

The guiding rule: a finished mediocre solution scores more than an unfinished brilliant one. Get a baseline submitted first, improve only if time remains.

## Files

| File | Use it when |
|---|---|
| [`01_data_prep.md`](./01_data_prep.md) | Always. Loading, missing values, encoding, scaling, splitting, writing the submission |
| [`02_tabular_models.md`](./02_tabular_models.md) | Data is a table of rows and columns |
| [`03_text_models.md`](./03_text_models.md) | Input is sentences, messages, documents |
| [`04_image_models.md`](./04_image_models.md) | Input is pictures |
| [`05_signal_timeseries.md`](./05_signal_timeseries.md) | Input is sensor readings or audio |
| [`06_cheatsheet.md`](./06_cheatsheet.md) | The copy-paste page. Keep it open during the round |

## Which model, in one table

| Your data | First thing to try | Takes | Page |
|---|---|---|---|
| Table, predict a category | `HistGradientBoostingClassifier` | 5 min | [02](./02_tabular_models.md) |
| Table, predict a number | `HistGradientBoostingRegressor` | 5 min | [02](./02_tabular_models.md) |
| Table, very few rows (<500) | `LogisticRegression` or `RandomForest` | 5 min | [02](./02_tabular_models.md) |
| Text, predict a category | `TfidfVectorizer` + `LinearSVC` | 5 min | [03](./03_text_models.md) |
| Text, need meaning not words | `sentence-transformers` + `LogisticRegression` | 20 min | [03](./03_text_models.md) |
| Images, count or measure objects | OpenCV contours — no ML at all | 15 min | [04](./04_image_models.md) |
| Images, predict a category | pretrained CNN features + `LogisticRegression` | 30 min | [04](./04_image_models.md) |
| Images, find similar images | pretrained CNN embeddings + cosine | 30 min | [04](./04_image_models.md) |
| Sensor readings over time | window features + gradient boosting | 20 min | [05](./05_signal_timeseries.md) |
| Audio, predict a category | mel-spectrogram stats + gradient boosting | 30 min | [05](./05_signal_timeseries.md) |

## The 20-minute baseline, every time

1. **Look at the data.** `df.shape`, `df.head()`, `df.isna().sum()`, `df[target].value_counts()`.
2. **Make the dumbest possible submission.** Predict the most common class for everything. Submit it. You now have a score and a valid file format.
3. **Fit one gradient boosting model** on whatever features exist already.
4. **Cross-validate once** to see whether step 3 beat step 2.
5. **Submit.** Only now consider improving.

Steps 1–5 take twenty minutes and are worth more than three hours of clever work that never produces a file.

## What this course deliberately skips

These are real techniques that are not worth your time in a short contest:

- Training neural networks from scratch, writing custom PyTorch training loops
- Fine-tuning large language models
- Hyperparameter search beyond trying three or four values by hand
- Stacking and blending ensembles
- Custom loss functions, custom layers
- AutoML frameworks — setup cost exceeds the benefit at this scale

Use pretrained models as **feature extractors**, never as things you train yourself.

## Installing everything at once

```bash
pip install pandas numpy scikit-learn lightgbm catboost opencv-python pillow \
            sentence-transformers torch torchvision librosa scipy tqdm
```

If a library fails to install, fall back to `scikit-learn`, which alone covers most of this course.

## External references

Bookmark these three; they answer most questions faster than a search:

- **scikit-learn user guide** — https://scikit-learn.org/stable/user_guide.html
- **scikit-learn algorithm chooser** — https://scikit-learn.org/stable/machine_learning_map.html
- **pandas user guide** — https://pandas.pydata.org/docs/user_guide/index.html
