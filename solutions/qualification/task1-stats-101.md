# FAIO 2025 Qualification — Task 1: Statistics 101?

Source: [`faio-2025/qualification/task1_stats_101.md`](../../faio-2025/qualification/task1_stats_101.md) · Yandex.Contest 81908, problem 1 · 10 points

## 1. What the problem asks

Solve a descriptive-statistics problem by hand. Ten students report how many pets they have: `2, 0, 1, 3, 2, 1, 0, 4, 1, 1`. Compute the mean, the mode and the median, and print the three numbers on one line in that order.

No dataset, no model, no code required — the numbers are in the statement and the answer is three values.

## 2. Knowledge required

- Mean — sum divided by count
- Mode — most frequent value
- Median — middle of the ordered list, and the even-length rule of averaging the two middle values

**Given in the statement:** all three, in a Theory section. It defines mean, mode and median, with a data-analysis example and an ML example for each (MSE, majority voting, median absolute error). So this problem supplies its own knowledge — nothing external needed beyond arithmetic.

## 3. Languages and libraries

Irrelevant for this task: the answer is typed into Yandex.Contest as three numbers, not submitted as a program. No language or library restriction is stated anywhere in the repo for the 2025 qualification, and faio.kz publishes no rules page — so for the answer-only tasks, treat tooling as unrestricted and unverified. You may of course check your arithmetic with anything, a calculator included.

## Solution

Sorted: `0, 0, 1, 1, 1, 1, 2, 2, 3, 4`

- **Mean** = `(2+0+1+3+2+1+0+4+1+1) / 10 = 15/10 = 1.5`
- **Mode** = `1` — it appears four times, more than any other value
- **Median** = with 10 values, average the 5th and 6th of the sorted list = `(1 + 1)/2 = 1`

Answer line:

```
1.5 1 1
```

**Traps:**

- Even count means the median is the average of two middle values, not a single element. Here both are `1`, so the distinction is invisible — but do not learn the wrong rule from it.
- Order matters: mean, then mode, then median. The example output in the statement (`99 3.14 2.72`) is just a format illustration, not a hint at the values.
- Do not round the mean to an integer.
