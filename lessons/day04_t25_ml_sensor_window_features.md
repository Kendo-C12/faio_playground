# T25 · Sensor windows and segment features — Day 4

**Anchor task(s):**
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)

Rating 0 · **exam-probability rank 1** · ~45 min

## [concept-first]

**A window (or segment) is a contiguous run of timesteps treated as one example.** Sensor data arrives one row per sample: `ax, ay, az, ...`. A label, though, describes a *stretch* of time — "this yoga repetition was correct". The window is the unit the label belongs to, and this topic is about turning a window of many rows into a single row of numbers.

**Why per-timestep rows cannot be fed to a tabular classifier.** Three independent reasons, each alone fatal. *Shape:* the label exists once per repetition, so 400 rows would each carry a copy, training the model to answer "is this 5-millisecond instant correct?" — a question no instant can answer. *Blindness:* a row-wise model sees six numbers with no neighbours, so smoothness, timing and how the axes move *together* — everything that separates good form from bad — is invisible. *Aggregation:* you end up with 400 predictions per repetition and no principled way to vote them into one.

So: aggregate first, classify second. One row per `id`.

**Sampling-rate arithmetic.** $f_s$ Hz means $f_s$ rows per second, so

$$
n_{\text{rows}} = \text{duration} \times f_s
\qquad
\text{duration} = \frac{n_{\text{rows}}}{f_s}
$$

At task 5's **200 Hz**: one second is 200 rows, a 2-second repetition is ~400 rows, consecutive rows are 5 ms apart. Do this on the real data before believing the text — $n_{\text{rows}} / 200$ must land in a plausible range for a yoga repetition (roughly 1–10 s).

**The standard feature battery,** per window per axis:

| Feature | Formula | What it captures |
|---|---|---|
| mean | $\bar{x}$ | resting level / orientation |
| std, range, RMS | $\sqrt{\sum_i (x_i - \bar{x})^2 / n}$, $\max - \min$, $\sqrt{\sum_i x_i^2 / n}$ | how much it moved, total activity |
| zero-crossing rate | fraction of $i$ with $\operatorname{sign}(x_i) \ne \operatorname{sign}(x_{i+1})$ | oscillation rate — a cheap frequency proxy |
| cross-axis correlation | $\operatorname{corr}(a_x, a_y)$ etc. | coordinated vs sloppy movement |
| magnitude | $\sqrt{a_x^2 + a_y^2 + a_z^2}$, then its stats | orientation-invariant activity |
| n_rows | count | duration |

`mean` tells you the pose; `std`, `range` and `RMS` tell you how it was performed; the zero-crossing rate stands in for the spectral features of T26; cross-axis correlation is the only entry that can see coordination. Run the battery on the magnitude channels too — they are invariant to how the sensor was mounted.

**The grouped-split rule.** A window must never straddle train and validation. Split rows at random and timesteps from the *same* repetition land on both sides; the model memorises that repetition and validation accuracy becomes meaningless optimism. Aggregating to one row per `id` first makes a plain split automatically grouped — a second reason to aggregate first. If you ever split raw rows, use `GroupShuffleSplit` or `GroupKFold` with `groups=df["id"]`. Same rule for overlapping windows: two windows sharing samples belong in the same fold.

## [problem-first]

Open [`task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md). The Input section says "Each repetition is represented by time series of IMU readings sampled at **200 Hz**", with `ax, ay, az` in g and `wx, wy, wz` in deg/s, and "Each `id` corresponds to one yoga repetition."

Then look at the Example block in the same file: rows `0`, `1`, `2` each carry a *different* `id`, six signal columns, and no time column at all.

**The statement contradicts itself.** The prose describes a time series per repetition — many rows sharing one `id`. The example shows one row per repetition. Both cannot be true of the same file.

This is not a puzzle to reason about; it is a measurement to take. The first command of the round is `X_test.groupby("id").size().max()`.

- **> 1** — the prose is right: windows exist, and this lesson *is* the task. Aggregate with the battery above.
- **= 1** — the example is right: each row is already an example, and you skip to T16 feature engineering on six columns. Every `groupby` you wrote is dead code.

Check the second thing too: `X_train.groupby("id").size().describe()`. Unequal window lengths make `n_rows` a feature in its own right (duration may separate correct from incorrect) and rule out any fixed-length model input. And since there is no timestamp column, row order *is* time order — never sort or shuffle before computing anything order-dependent.

Output is `solution.csv` with `id,label`, "strictly follow the order in `test.csv`, starting from `0`", scored on plain accuracy. Whatever happens upstream, the deliverable is one label per `id` in ascending `id` order.

## [code-first]

```python
import numpy as np
import pandas as pd

SIG = ["ax", "ay", "az", "wx", "wy", "wz"]
FS = 200                                     # Hz, from the statement: 200 rows = 1 second

X = pd.read_csv("X_train.csv")

# COMMAND ONE of the round: does this data have windows at all?
sizes = X.groupby("id").size()
print("rows/id:", sizes.min(), sizes.median(), sizes.max(), "| seconds:", sizes.max() / FS)
if sizes.max() == 1:
    raise SystemExit("One row per id: no windows. Go to T16 instead.")

zcr = lambda s: float(np.mean(np.sign(s.values[:-1]) != np.sign(s.values[1:]))) if len(s) > 1 else 0.0
rms = lambda s: float(np.sqrt(np.mean(s.values ** 2)))
rng = lambda s: float(s.max() - s.min())

def featurise(X):
    X = X.copy()
    # Magnitudes first: invariant to how the sensor was mounted on the body.
    X["amag"] = np.sqrt(X.ax**2 + X.ay**2 + X.az**2)
    X["wmag"] = np.sqrt(X.wx**2 + X.wy**2 + X.wz**2)
    cols = SIG + ["amag", "wmag"]

    g = X.groupby("id")
    agg = g[cols].agg(["mean", "std", "min", "max", rms, rng, zcr])
    agg.columns = ["_".join(map(str, c)) for c in agg.columns]   # flatten immediately
    agg["n_rows"] = g.size()                                    # duration as a feature
    # Coordination: the only feature that sees axes moving TOGETHER.
    for a, b in [("ax", "ay"), ("ax", "az"), ("wx", "wy"), ("amag", "wmag")]:
        agg[f"corr_{a}_{b}"] = g.apply(lambda d: d[a].corr(d[b]))
    return agg.fillna(0)                       # single-row groups give NaN std/corr

F = featurise(X).sort_index()                  # one row per id == already grouped
y = pd.read_csv("y_train.csv").set_index("id").loc[F.index, "label"]
# A plain train_test_split is now safe; on raw rows it would have leaked across the split.
```

## [drill]

1. At 200 Hz, how many rows is a 2.5-second repetition, and how much time separates two consecutive rows?
2. `X_train` has 80,000 rows and 200 distinct ids. What is the average window length in seconds?
3. Name the three independent reasons per-timestep rows cannot be fed to the classifier.
4. Which feature sees *coordination* between axes, and why can no single-axis statistic replace it?
5. You split raw rows 80/20 at random, validate at 0.99 and score 0.63. Name the mechanism and the fix.
6. Write the one line that decides whether this lesson applies to task 5 at all.

<details><summary>Answers</summary>

1. 500 rows; $1/200$ s = 5 ms.
2. $80{,}000 / 200 = 400$ rows per id, and $400 / 200 = 2.0$ seconds.
3. Shape mismatch with the per-repetition label; no temporal structure visible to a row-wise model; no defined way to vote 400 predictions back into one.
4. Cross-axis correlation. Two axes moving in lockstep and two moving independently can have identical per-axis `std`, `min` and `max` — only the joint term separates them.
5. Leakage across the grouped unit: rows of one repetition sit on both sides, so the model recognises the repetition rather than the form. Aggregate to one row per `id` first, or use `GroupKFold(groups=df["id"])`.
6. `print(pd.read_csv("X_test.csv").groupby("id").size().max())` — `1` means no windows.

</details>

**Rep:** run the window check on `X_test.csv` and `X_train.csv`, write down both answers, and state in one sentence which of T25 and T16 the round actually needs. Then build `featurise` and assert `len(F) == X.id.nunique()`.

## Traps & 60-second recall

- 200 Hz means 200 rows per second; duration is $n_{\text{rows}} / 200$.
- Aggregate to one row per `id` *before* modelling — it fixes the shape and removes the leakage risk at once.
- Never random-split raw timesteps; `GroupKFold` with `groups=id`, or aggregate first.
- Flatten MultiIndex columns the line after `agg`.
- `std` and `corr` are NaN for single-row groups — `fillna(0)`.
- Build magnitude channels; they delete the sensor-mounting nuisance.
- No timestamp column means row order is time order: never sort or shuffle first.
- Task 5's prose and example disagree on rows per `id`; measure it in one line before writing any pipeline.
