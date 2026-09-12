# T32 · Time budget for a 4-hour round — Day 5

**Anchor task(s):**
- [`faio-2025/qualification/task1_stats_101.md`](../faio-2025/qualification/task1_stats_101.md)
- [`faio-2025/qualification/task2_Probability_theory_101.md`](../faio-2025/qualification/task2_Probability_theory_101.md)
- [`faio-2025/qualification/task3_Medicine.md`](../faio-2025/qualification/task3_Medicine.md)
- [`faio-2025/qualification/task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md)
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)
- [`faio-2025/qualification/task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md)

Rating 0 · **exam-probability rank 1** · ~35 min

Tomorrow is 12:00–16:00. 240 minutes, shared across everything. This lesson is the plan you execute, not a topic you learn.

## [concept-first]

**Read everything first.** Ten minutes reading all statements before touching any of them. You cannot triage what you have not seen, and the cheapest points in the 2025 round were in tasks 1–3, which a reader spots instantly and a head-down coder discovers at 15:40.

**Score per minute, not difficulty.** The 2025 qualification paid 10 points each for tasks 1, 2 and 3 — three pen-and-paper answers ([`task1_stats_101.md`](../faio-2025/qualification/task1_stats_101.md): mean/mode/median of ten listed numbers; [`task2_Probability_theory_101.md`](../faio-2025/qualification/task2_Probability_theory_101.md): `P(at least one six in three rolls) = 1 − (5/6)³ = 0.4213`; [`task3_Medicine.md`](../faio-2025/qualification/task3_Medicine.md): Bayes, `0.01·0.99 / (0.01·0.99 + 0.99·0.02) ≈ 0.3333`). Each is two minutes with a calculator. 30 points for six minutes is a rate no modelling task can touch.

**Baseline, then improve.** Each of tasks 4, 5 and 6 is graded on a metric over hidden data. A submission that is *valid and mediocre* scores; a brilliant pipeline that never writes its CSV scores zero. So every ML task gets a crude end-to-end pass before any of them gets a second look: TF-IDF + logistic regression for [`task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md), per-`id` aggregates + boosting for [`task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md), threshold + `findContours` for [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md).

**The hard rule.** *Something valid submitted for every task before refining any task.* This is not a preference. It is the only rule that survives a round going wrong — a dataset that will not download, a kernel that dies, an hour lost to one bug. Breaking it is how people finish with three perfect tasks and three zeros.

**When to abandon.** Abandon a task when (a) you have a valid submission for it and the next improvement has no clear mechanism, or (b) you have spent 25 minutes without a valid submission and cannot name the next concrete step. Write down the best guess you have and move. For a counting task, a constant prediction near the training median still earns partial credit; for a classifier, the majority class does.

**Reserve the last 20 minutes.** Not for modelling. For validation: row counts, column names, file names, ordering, and the actual upload. [`task5`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md) demands `solution.csv` with `id,label` where "the order of `id` values must strictly follow the order in `test.csv`, starting from `0`"; [`task6`](../faio-2025/qualification/task6_Simple_Objects.md) demands `submit.csv`. Different names, same round. That is what the last 20 minutes is for.

**Budget for one surprise.** `faio-2024/qualification` was a completely different animal — Yandex.Contest competitive programming with 1 s / 64 MB limits — and 2025 dropped that format entirely. Formats change between years, so assume one task will not look like anything in the archive, and leave it slack rather than letting it eat the round.

## [problem-first]

Take the 2025 round as if it were tomorrow's and build the schedule. Six tasks, 240 minutes, the 12:00–16:00 slot.

| Clock | Minutes | What |
|---|---|---|
| 12:00–12:10 | 10 | Read all six statements. Note for each: output filename, columns, row count, metric. |
| 12:10–12:25 | 15 | Tasks 1–3 on paper. Three typed numbers, 30 points banked. Respect the formats: three numbers space-separated; four decimal places. |
| 12:25–12:35 | 10 | Start every dataset downloading. Confirm files open and shapes look sane. |
| 12:35–13:00 | 25 | Task 4 baseline: char n-gram TF-IDF + logistic regression → `solution.csv`. Submit. |
| 13:00–13:30 | 30 | Task 5 baseline: `groupby("id")` aggregates + `HistGradientBoostingClassifier` → `solution.csv`. Submit. |
| 13:30–14:00 | 30 | Task 6 baseline: grayscale → threshold → `RETR_EXTERNAL` contours, area filter. Time 10 images, extrapolate, launch the full 8000 in the background. |
| 14:00–14:10 | 10 | Checkpoint. All six have something. Rank the three ML tasks by expected gain per minute. |
| 14:10–14:50 | 40 | Best-gain task, improvement pass #1. Re-submit only if local validation improves. |
| 14:50–15:25 | 35 | Second-best task, improvement pass. Keep the previous submission file until the new one validates. |
| 15:25–15:40 | 15 | Slack: the surprise task, or the one bug you have been avoiding. |
| 15:40–16:00 | 20 | **Frozen.** Validate every file: name, header, row count, ordering, no index column. Upload. Confirm each upload registered. |

Three things to read off that table. The 2025 maths tasks were 12.5% of the points for 6% of the time. The task 6 runtime must be *launched early* because it runs for tens of minutes regardless of how clever you are (see [`day5-t34-speed-on-large-inputs.md`](./day5-t34-speed-on-large-inputs.md)). And nothing after 15:40 is creative work.

## [code-first]

A timer and a validator, written before the round starts, pasted into the first cell.

```python
import time, os, pandas as pd
T0 = time.time()
def clock(label):                      # call at every checkpoint; keeps you honest
    m = (time.time() - T0) / 60
    print(f"[{m:6.1f} min elapsed | {240-m:6.1f} left] {label}")

# Expected deliverables, written down BEFORE coding anything. One line per task,
# straight from the statements: task4/task5 -> solution.csv, task6 -> submit.csv.
SPEC = {
    "task4": ("solution.csv", ["id", "label"], None),
    "task5": ("solution.csv", ["id", "label"], None),
    "task6": ("submit.csv",   None,            None),   # one integer per row
}

def check(task, expected_rows=None):
    name, cols, _ = SPEC[task]
    assert os.path.exists(name), f"{task}: {name} MISSING — this is a zero"
    df = pd.read_csv(name)
    if cols: assert list(df.columns) == cols, f"{task}: columns {list(df.columns)} != {cols}"
    if expected_rows: assert len(df) == expected_rows, f"{task}: {len(df)} rows != {expected_rows}"
    assert df.isna().sum().sum() == 0, f"{task}: NaNs in submission"
    print(f"{task}: OK — {len(df)} rows, {list(df.columns)}")
    return df

# Fallback written FIRST, before the real model. Valid beats absent, always.
def fallback_constant(ids, value, path, cols=("id", "label")):
    pd.DataFrame({cols[0]: ids, cols[1]: value}).to_csv(path, index=False)
    print(f"fallback written to {path}: {len(ids)} rows of {value}")
```

The discipline: `fallback_constant` runs at minute 30 for every ML task. `check` runs at minute 220 for all of them.

## [drill]

1. It is 15:45 and your task 5 model is mid-training, 8 minutes from done. What do you do?
2. Which three 2025 tasks pay the most points per minute, and roughly what rate?
3. You have 25 minutes in on task 6 and no valid `submit.csv`. Name the action.
4. Why must the task 6 full run be launched before your improvement passes rather than after?
5. Why is the `faio-2024/qualification` format worth knowing even though 2025 dropped it?
6. You improve local validation on task 4 at 15:50. Do you re-submit?

<details><summary>Answers</summary>

1. Kill it. You already have a valid `solution.csv` from the baseline; the last 20 minutes is frozen for validation and upload. An unfinished run is worth zero and it risks the file you do have.
2. Tasks 1, 2 and 3 — 10 points each for roughly two minutes of arithmetic, so about 5 points per minute versus a whole baseline pipeline for one metric score.
3. Write the fallback: predict a constant near the training median count for every row, save `submit.csv`, then continue. Relative-error scoring gives partial credit for a sane constant.
4. Because its cost is wall-clock, not effort: 8000 images at 2048×2048 runs for tens of minutes whether or not you are watching. Start it, then work on something else while it runs.
5. As evidence the format can change between years — 2024 was Yandex.Contest competitive programming with 1 s / 64 MB limits, 2025 had none of that. Budget slack for one task that resembles nothing in the archive.
6. Only if the new file passes `check()` and you keep the old one on disk until it does. If validation is tight, keep the submitted file and stop.

</details>

**Rep:** write out tomorrow's 12:00–16:00 table in your own hand, with the filename and column spec of each 2025 task filled in from the statements. Tape it next to the keyboard.

## Traps & 60-second recall

- 10 minutes reading all statements. Every time.
- Bank the pen-and-paper points first — highest points per minute in the round.
- Something valid submitted for every task *before* refining any task.
- Write the constant-prediction fallback before the real model.
- Launch long-running jobs early; they cost wall-clock, not attention.
- Abandon after 25 minutes with no valid submission and no next step.
- Last 20 minutes frozen: filenames, headers, row counts, ordering, `index=False`, upload confirmed.
- Filenames differ between tasks in the same round (`solution.csv` vs `submit.csv`). Check, do not assume.
- Expect one task that looks like nothing in the archive, and leave it slack.
