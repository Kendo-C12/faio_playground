# FAIO 2025 Qualification — Task 2: Probability Theory 101

Source: [`faio-2025/qualification/task2_Probability_theory_101.md`](../../faio-2025/qualification/task2_Probability_theory_101.md) · Yandex.Contest 81908, problem 2 · 10 points

## 1. What the problem asks

Solve a probability problem by hand. A standard 6-sided die is rolled three times. Find the probability of rolling **at least one** six. Answer as a decimal with exactly four digits after the point.

No dataset, no model — a single number.

## 2. Knowledge required

- Classical probability, `P(A) = m/n` over equally likely outcomes
- Independence of repeated rolls, so probabilities multiply
- The complement trick: `P(at least one) = 1 − P(none)`

**Given in the statement:** the classical definition `P(A) = m/n`, the 0–1 range, and ML framing (spam threshold, credit scoring). The complement rule and independence are **not** given — you need those yourself.

## 3. Languages and libraries

Irrelevant: the answer is typed in as a number. No restriction is stated for the 2025 qualification anywhere in the repo or on faio.kz, so treat tooling as unrestricted and unverified.

## Solution

Count the complement. "No six in three rolls" means each roll lands on one of 5 faces:

```
P(no six)      = (5/6)³ = 125/216
P(at least one) = 1 − 125/216 = 91/216 = 0.421296...
```

Rounded to four decimals:

```
0.4213
```

**Traps:**

- `3 × (1/6) = 0.5` is the classic wrong answer. It double-counts the outcomes with two or three sixes, and would exceed 1 for seven rolls — a quick sanity check that the method is broken.
- `1 − 125/216` is `91/216`, not `90/216`. Do the subtraction on the fraction, then divide once.
- Round, do not truncate: `0.42129…` → `0.4213`.
