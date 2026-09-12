# T25 · Sensor windows and segment features — Day 4

**Anchor task(s):**
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)

Rating 0 · **exam-probability rank 1** · ~45 min

## [concept-first]

**A window (or segment) is a contiguous run of timesteps treated as one example.** Sensor data arrives as one row per sample: `t, ax, ay, az, ...`. A label, though, describes a *stretch* of time — "this yoga repetition was correct". The window is the unit the label belongs to, and the job of this topic is to turn a window of many rows into a single row of numbers.

**Why you cannot feed per-timestep rows to a tabular classifier.** Three separate reasons, and each alone is fatal.

1. *Shape mismatch.* The label exists once per repetition; 400 rows would each need a copy of it. You would be training a model to answer "is this single 5-millisecond instant correct?", a question nobody asked and no instant can answer.
2. *No temporal information reaches the model.* A row-wise classifier sees six numbers with no neighbours. Everything that distinguishes good form from bad — smoothness, timing, how the axes move *together* — lives in relationships between rows and is invisible.
3. *Prediction aggregation is undefined.* You would get 400 predictions per repetition and have to vote them into one, and a majority vote over near-identical instants is driven by whichever instants are most common, not by the repetition.

So: aggregate first, classify second. One row per `id`.

**Sampling-rate arithmetic.** A rate of `fs` Hz means `fs` rows per second. So the number of rows is `duration × fs`, and duration is `n_rows / fs`. At task 5's **200 Hz**: one second is 200 rows, a 2-second repetition is ~400 rows, and the time between consecutive rows is 5 ms. Do this arithmetic on the real data before believing the statement: `n_rows / 200` should land in a plausible range for a yoga repetition (roughly 1–10 s). If it gives 0.005 s, the data is not what the text says.

**The standard feature battery,** computed per window per axis:

| Feature | Formula | What it captures |
|---|---|---|
| mean | `x̄` | resting level / orientation |
| std | `sqrt(Σ(xᵢ−x̄)²/n)` | how much it moved |
| min, max, range | `max − min` | extremes of the motion |
| RMS / energy | `sqrt(Σxᵢ²/n)` | total magnitude of activity |
| zero-crossing rate | fraction of `i` with `sign(xᵢ) ≠ sign(xᵢ₊₁)` | oscillation rate, cheap frequency proxy |
| cross-axis correlation | `corr(ax, ay)` etc. | coordinated vs sloppy movement |
| magnitude | `sqrt(ax²+ay²+az²)` then its stats | orientation-invariant activity |
| n_rows | count | duration |

`mean` tells you the pose; `std`, `range` and `RMS` tell you how it was performed; the zero-crossing rate is a one-line stand-in for the spectral features of T26; cross-axis correlation is the only feature in the list that can see coordination. Compute the battery on the magnitude channels too — they are invariant to how the sensor was mounted.

**The grouped-split rule.** A window must never straddle train and validation. If you split rows at random, timesteps from the *same* repetition land on both sides; the model memorises that repetition and validation accuracy becomes meaningless optimism. Because you aggregate to one row per `id` first, a plain split is already grouped — which is a second reason to aggregate first. If you ever do split raw rows, use `GroupShuffleSplit` or `GroupKFold` with `groups=df["id"]`. The same rule applies to overlapping windows: two windows sharing samples must go to the same fold.

## [problem-first]

Open [`task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md). The Input section says: "Each repetition is represented by time series of IMU readings sampled at **200 Hz**" with `ax, ay, az` in g and `wx, wy, wz` in deg/s, and "Each `id` corresponds to one yoga repetition."

Then look at the Example block in the same file:

```
id,ax,ay,az,wx,wy,wz
0,0.12,0.04,0.98,-1.2,0.3,0.1
1,0.01,0.15,1.02,0.8,-0.5,0.0
2,-0.10,0.20,0.95,0.7,0.1,-0.2
```

**The statement contradicts itself.** The prose describes a time series per repetition — many rows sharing one `id`. The example shows three distinct `id` values on three consecutive rows, i.e. one row per repetition, with no time column anywhere. Both cannot be true of the same file.

This is not a puzzle to reason about; it is a measurement to take. The first command of the round is:

```python
X_test.groupby("id").size().max()
```

- If it is **> 1**, the prose is right: windows exist, and this whole lesson is the task. Aggregate with the battery above.
- If it is **1**, the example is right: there are no windows, each row is already an example, and you skip straight to T16 feature engineering on six columns. Any `groupby` machinery you wrote is dead code.

Either way, check the second thing too: `X_train.groupby("id").size().describe()`. Unequal window lengths mean `n_rows` is itself a feature (duration may separate correct from incorrect), and they rule out any fixed-length model input. And note there is no timestamp column, so row order *is* time order — never sort or shuffle `X_train` before aggregating anything order-dependent.

Output is `solution.csv` with `id,label`, "strictly follow the order in `test.csv`, starting from `0`", scored on plain accuracy. So whatever you do upstream, the deliverable is one label per `id` in ascending `id` order.

## [code-first]

```python
import numpy as np
import pandas as pd

SIG = ["ax", "ay", "az", "wx", "wy", "wz"]
FS = 200                                    # Hz, from the statement: 200 rows = 1 second

X = pd.read_csv("X_train.csv")

# COMMAND ONE of the round: does the data have windows at all?
sizes = X.groupby("id").size()
print("rows per id:", sizes.min(), sizes.median(), sizes.max())
print("implied duration (s):", sizes.min() / FS, "-", sizes.max() / FS)
if sizes.max() == 1:
    raise SystemExit("One row per id: no windows. Go to T16 instead.")

def zcr(s):                                 # cheap oscillation-rate proxy (see T26 for the real thing)
    v = s.to_numpy()
    return float(np.mean(np.sign(v[:-1]) != np.sign(v[1:]))) if len(v) > 1 else 0.0

def rms(s):
    return float(np.sqrt(np.mean(s.to_numpy() ** 2)))

def featurise(X):
    X = X.copy()
    # Magnitudes first: invariant to how the sensor was mounted on the body.
    X["amag"] = np.sqrt(X.ax**2 + X.ay**2 + X.az**2)
    X["wmag"] = np.sqrt(X.wx**2 + X.wy**2 + X.wz**2)
    cols = SIG + ["amag", "wmag"]

    g = X.groupby("id")
    agg = g[cols].agg(["mean", "std", "min", "max", rms, zcr])
    agg.columns = ["_".join(map(str, c)) for c in agg.columns]   # flatten immediately
    for c in cols:
        agg[f"{c}_range"] = g[c].max() - g[c].min()

    agg["n_rows"] = g.size()                       # duration as a feature
    agg["dur_s"] = agg["n_rows"] / FS              # same number, human-readable units
    # Coordination: the only thing in the battery that sees axes moving TOGETHER.
    for a, b in [("ax", "ay"), ("ax", "az"), ("wx", "wy"), ("amag", "wmag")]:
        agg[f"corr_{a}_{b}"] = g.apply(lambda d: d[a].corr(d[b]))

    return agg.fillna(0)                           # single-row groups give NaN std/corr

F = featurise(X).sort_index()                      # one row per id == already grouped
y = pd.read_csv("y_train.csv").set_index("id").loc[F.index, "label"]
# A plain train_test_split is now safe; on raw rows it would have leaked across the split.
```

## [drill]

1. At 200 Hz, how many rows is a 2.5-second repetition, and how much time separates two consecutive rows?
2. `X_train` has 80,000 rows and 200 distinct ids. What is the average window length in seconds?
3. Name the three independent reasons per-timestep rows cannot be fed to the classifier.
4. Which feature in the battery sees *coordination* between axes, and why can no single-axis statistic replace it?
5. You split raw rows 80/20 at random and validate at 0.99 but score 0.63. Name the mechanism and the fix.
6. Write the one line that decides whether this lesson applies to task 5 at all.
7. Windows have unequal lengths. Name one feature this gives you for free.

<details><summary>Answers</summary>

1. 500 rows; 1/200 s = 5 ms.
2. 80,000 / 200 = 400 rows per id, and 400 / 200 Hz = 2.0 seconds.
3. Shape mismatch with the per-repetition label; no temporal structure visible to a row-wise model; no defined way to aggregate 400 predictions back into one.
4. Cross-axis correlation. A per-axis `std` tells you each axis moved, but two axes moving in lockstep and two moving independently can have identical per-axis statistics — only the joint term distinguishes them.
5. Leakage across the grouped unit: rows of one repetition appear in both halves, so the model recognises the repetition rather than the form. Aggregate to one row per `id` first, or use `GroupKFold(groups=df["id"])`.
6. `print(pd.read_csv("X_test.csv").groupby("id").size().max())` — `1` means no windows.
7. `n_rows` (equivalently `n_rows / 200` seconds) — duration, which may itself separate correct from incorrect repetitions.

</details>

**Rep:** run the window check on `X_test.csv` and `X_train.csv`, write down both answers, and state in one sentence which of T25 and T16 the round actually needs. Then build `featurise` and assert `len(F) == X.id.nunique()`.

## Traps & 60-second recall

- 200 Hz means 200 rows per second; duration is `n_rows / 200`.
- Aggregate to one row per `id` *before* modelling — it fixes the shape and removes the leakage risk at once.
- Never random-split raw timesteps; `GroupKFold` with `groups=id`, or aggregate first.
- Flatten MultiIndex columns the line after `agg`.
- `std` and `corr` are NaN for single-row groups — `fillna(0)`.
- Build magnitude channels; they delete the sensor-mounting nuisance.
- No timestamp column means row order is time order: do not sort or shuffle before aggregating.
- Task 5's prose and example disagree about rows per `id`; measure it in one line before writing any pipeline.
