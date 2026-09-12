# T31 · Submission discipline: filename, header, row count, order — Day 1

**Anchor task(s):**
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)
- [`faio-2025/qualification/task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md)
- [`faio-2025/qualification/task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md)

Rating 2 · **exam-probability rank 1** · ~40 min

A weak model costs points. A malformed file costs the whole task. In a 4-hour round those are not comparable risks, and only one of them is free to eliminate.

## [concept-first]

Five things a grader checks before it ever looks at your predictions.

**1. Filename, exactly.** Not "a CSV". Task 4 and task 5 both name **`solution.csv`**; task 6 names **`submit.csv`**. Same round, two different names — so the filename is a per-task fact you re-read from the statement, never a habit you carry across tasks.

**2. Header, exactly.** Task 4 and task 5 show `id,label`. No spaces after the comma, no capitals, no extra columns. `to_csv(index=False)` or you ship a nameless first column and every column shifts by one.

**3. Row count, exactly.** One row per test example, no more, no fewer. Task 6 is "tested on **8000** images" so 8000 counts. [`faio-2025/day2/Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md) is the harshest statement in the archive: "must contain exactly 20,000 rows — one for each image in the dataset — **with no duplicates or omissions**… no filtering, sorting, or skipping allowed."

**4. Row order, deterministic.** Both task 5 and task 6 say, verbatim: "The order of `id` values must strictly follow the order in `test.csv`, starting from `0` up to the highest ID." So ordering is not yours to choose. Derive it from the test file, never from a dict, a set, or `groupby` output you have not sorted.

**5. Label spelling, exactly.** Task 4 labels are the three strings `ru`, `kaz`, `eng`. Not `kz`, not `kaz ` with a trailing space, not `Kazakh`, not the integer your `LabelEncoder` produced. Task 5 labels are the integers `0` and `1`, not `0.0`/`1.0` and not `False`/`True`.

Worked example — the cost asymmetry. Suppose task 5 is worth the same as the others and your model reaches 0.88 accuracy. Shipping `prediction.csv` instead of `solution.csv` scores 0. Shipping a 0.72-accuracy model in a correct `solution.csv` scores most of the task. The five minutes of validation below are the best-paid five minutes in the round.

## [problem-first]

Open [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md) and read its Output section closely. It asks for "a CSV file (`submit.csv`) with a single integer in each row" — and then says the order of **`id` values** must follow `test.csv`. There is no `id` column in a file of single integers. The statement contradicts itself.

That is the real skill: the statement is the only spec you get, and it is imperfect. What you do about it:

1. **Download the provided sample submission and copy its shape.** Task 6 ships `submit.csv` as a file link for exactly this reason; `qaz_letters.md` ships `sample_submission.csv`; Lost in the Museum ships one too. The sample resolves every ambiguity the prose leaves open — column names, whether a header exists, row order.
2. **Where a sample exists, the sample wins over the prose.** Where no sample exists (task 4 and task 5 ship none), follow the prose literally and copy the header out of the statement's fenced block character by character.
3. **Match row count against the test file, not against your own intermediate frame.** `len(submission) == len(pd.read_csv("test.csv"))` for task 4; `== X_test.id.nunique()` for task 5, where many rows collapse to one id.
4. **Make the id column come from the test file.** For task 5, `sorted(X_test["id"].unique())` gives `0..max` in the demanded order and cannot silently drop an id.

## [code-first]

```python
import pandas as pd

def write_submission(ids, preds, path="solution.csv", n_expected=None,
                     allowed=None):
    """Write, then immediately re-read and validate. Never ship unvalidated."""
    sub = pd.DataFrame({"id": ids, "label": preds})

    # Order is fixed by the statement: ids ascending from 0 to the highest ID.
    sub = sub.sort_values("id").reset_index(drop=True)

    assert sub["id"].is_unique, "duplicate ids"
    assert sub["id"].iloc[0] == 0, f"ids must start at 0, got {sub['id'].iloc[0]}"
    assert sub["id"].tolist() == list(range(len(sub))), "gap in the id sequence"
    assert sub["label"].notna().all(), "NaN in label column"
    if n_expected is not None:
        assert len(sub) == n_expected, f"{len(sub)} rows, expected {n_expected}"
    if allowed is not None:                       # task 4: {"ru","kaz","eng"}
        bad = set(sub["label"]) - set(allowed)
        assert not bad, f"illegal labels: {bad}"

    sub.to_csv(path, index=False)                 # index=False, every time

    # Re-read from disk: the only check that tests what the grader will see.
    back = pd.read_csv(path)
    print(open(path).readline().rstrip())         # the literal header line
    print(back.shape, back.dtypes.to_dict())
    return back

# task 4 — ids are row positions in test.csv, labels are three exact strings
test = pd.read_csv("test.csv")
write_submission(range(len(test)), model.predict(test["text"]),
                 "solution.csv", n_expected=len(test),
                 allowed={"ru", "kaz", "eng"})

# task 5 — ids come from X_test, one per repetition, labels are ints 0/1
Xte = pd.read_csv("X_test.csv")
ids = sorted(Xte["id"].unique())
write_submission(ids, preds.astype(int), "solution.csv", n_expected=len(ids),
                 allowed={0, 1})
```

For task 6 the shape differs: no header, one integer per line, so `pd.DataFrame({"n": counts}).to_csv("submit.csv", index=False, header=False)` — but confirm against the shipped `submit.csv` sample first.

## [drill]

1. Task 6 asks for which filename, and task 5 for which? Why is this the first fact to write down?
2. You forget `index=False`. What exactly does the grader read as the `id` column?
3. Your task 4 pipeline used `LabelEncoder`. What appears in `solution.csv` and what does it score?
4. Task 5 gives 400 test repetitions; your inner join dropped 2. What is the score?
5. Name the one line that proves your Lost-in-the-Museum file is legal on row count.
6. The statement and the shipped sample submission disagree on whether a header exists. Which do you follow?

<details><summary>Answers</summary>

1. `submit.csv` for task 6, `solution.csv` for tasks 4 and 5. Because the same round uses two names, so a name carried over from the previous task is a silent zero.
2. The old DataFrame index as an unnamed first column, so `id` is read from the index values and `label` is read from what used to be `id` — every column shifted one left.
3. Integers `0,1,2` instead of `eng,kaz,ru`. Zero matching labels, so zero score with a perfectly good model. Call `le.inverse_transform` or just never encode the target.
4. 398 rows where 400 are expected — a format rejection, not a 0.5 % accuracy loss.
5. `assert len(sub) == 20000 and sub["image_name"].is_unique` — the statement demands exactly 20,000 rows with no duplicates or omissions.
6. The sample file. It is generated by the grader's own code; the prose is written by hand and, as task 6 shows, can contradict itself.

</details>

**Rep:** write `write_submission` from memory, then run it for task 5 with a deliberately broken input (drop one id) and confirm the assertion fires before anything reaches disk.

## Traps & 60-second recall

- Re-read the filename from the statement for every task; `solution.csv` and `submit.csv` both occur in 2025.
- `index=False` unless the statement shows an index column.
- Row count equals the test file's count, checked with an `assert`, not by eye.
- Ids ascending from `0` to the highest, taken from the test file.
- Labels in the statement's exact spelling and dtype — `kaz` not `kz`, `1` not `1.0`.
- Write, re-read from disk, print the header and shape. Unvalidated files are how good models score zero.
- Where a sample submission ships, it outranks the prose.
