# Task 5 — AI Yoga Instructor · why this solution

Runnable notebook: [`task5_ai_yoga_instructor.ipynb`](./task5_ai_yoga_instructor.ipynb)
Statement: [`../qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)

## What the task asks

Decide whether one yoga repetition was performed correctly. You get IMU sensor readings — `ax ay az` for acceleration in g, `wx wy wz` for angular velocity in deg/s — sampled at **200 Hz**, with one `id` per repetition. Labels are `0` for incorrect and `1` for correct. The metric is **accuracy**. You submit `solution.csv` with `id,label`.

## The first command of the round

The statement contradicts itself, and the contradiction changes the entire pipeline.

The prose says each repetition is a time series at 200 Hz, which implies hundreds of rows per `id`. The worked example in the same file shows ids `0`, `1`, `2` on three consecutive rows with no time column, which implies exactly one row per `id`.

These cannot both be true, and no amount of reasoning settles it. One line does:

```python
X_train.groupby("id").size().max()
```

If the answer is `1`, the data is already one row per repetition and you skip feature aggregation entirely. If it is in the hundreds, you must aggregate. The notebook runs this check first and branches on the result, so it works either way.

Do not write the pipeline before running that line. Guessing wrong costs you the task.

## Why aggregate, if it is a time series

The label belongs to the **repetition**, not to the row. A classifier that sees individual timesteps is being asked a question the data cannot answer: a single 5-millisecond sample says nothing about whether a whole pose was correct.

So each repetition collapses to one feature vector. The features chosen are not arbitrary:

- **Spread beats level.** "Correct versus incorrect pose" is about the *shape* of the motion, not its average. Standard deviation, range and RMS carry more signal than the mean, which mostly reflects how the sensor was sitting.
- **Magnitudes remove an orientation nuisance.** `ax` depends on how the sensor was mounted; a rotated mount redistributes the same motion across the three axes. The magnitude $\sqrt{a_x^2 + a_y^2 + a_z^2}$ does not change under rotation, so it isolates the motion from the mounting.
- **Cross-axis correlation captures coordination.** A controlled movement moves axes together in a consistent relationship; a sloppy one decorrelates them.
- **`n_rows` is duration.** At a fixed 200 Hz, the number of rows in a group *is* the length of the repetition in time. Duration alone may separate the classes, and it costs one line.

## The validation trap that decides your score

This is the single most expensive mistake available in this task.

At 200 Hz, consecutive rows are 5 milliseconds apart, from the same person doing the same motion. They are near-identical. If you shuffle rows at random into cross-validation folds:

1. About 80 % of every repetition lands in the training folds.
2. The rows held out for validation are near-duplicates of rows the model just memorised.
3. Accuracy comes back around 0.99, and it measures nothing at all.

You would then spend the round tuning against a number that cannot move, and discover the truth only when the leaderboard disagrees.

The fix is to treat the repetition as the unit: aggregate to one row per `id` first (after which plain `StratifiedKFold` is correct), or use `StratifiedGroupKFold(groups=id)` if you insist on modelling raw rows. The notebook does the former and shows the latter.

## Why gradient boosting

Once the data is a table of per-repetition features, this is an ordinary tabular classification problem, and gradient boosting is the right default:

- It handles features on wildly different scales without any scaling step, because trees compare thresholds rather than distances.
- `HistGradientBoostingClassifier` tolerates `NaN` natively, which matters because the standard deviation of a single-row group is `NaN`.
- It trains in seconds at this data size, leaving time for the other five tasks.

A neural network on raw timesteps would be the textbook answer for sensor data, and the wrong answer here: far more code, far more time, and it overfits at this dataset size.

The notebook compares two learning-rate settings by cross-validation and reports the standard error alongside each mean, so you can tell a real improvement from noise. A gain smaller than roughly twice the standard error is not a gain.

## Permutation importance

The notebook ends the modelling section by measuring which features actually carry signal, on a held-out split. This is not decoration: if `n_rows` dominates, the model may be separating long repetitions from short ones rather than good form from bad, and that is worth knowing before you trust the score.

## Submission traps

- `id,label` with labels `0`/`1` as integers, not booleans or strings.
- Ids ascending, following `test.csv` order from `0` — note that the task ships `X_test.csv`, not a file literally named `test.csv`.
- `index=False`.
- The filename is `solution.csv`.
