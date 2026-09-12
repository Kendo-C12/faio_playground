# FAIO 2025 Qualification — Task 3: Medicine

Source: [`faio-2025/qualification/task3_Medicine.md`](../../faio-2025/qualification/task3_Medicine.md) · Yandex.Contest 81908, problem 3 · 10 points

## 1. What the problem asks

Solve a conditional-probability problem by hand. A disease affects 1 % of the population. The test returns positive for 99 % of sick people, and negative for 98 % of healthy people. A randomly chosen person tests **positive** — what is the probability they are actually sick? Answer as a decimal with four digits after the point.

## 2. Knowledge required

- Conditional probability, `P(A|B) = P(A ∩ B) / P(B)`
- Bayes' theorem
- Law of total probability, to build `P(positive)` from the sick and healthy branches
- Reading test accuracy correctly: 98 % **specificity** means a 2 % false-positive rate

**Given in the statement:** all of it. The Theory section covers probability, conditional probability and Bayes' theorem explicitly. This task is self-contained — the hard part is arithmetic care, not missing knowledge.

## 3. Languages and libraries

Irrelevant: the answer is one typed number. No restriction stated for the 2025 qualification in the repo or on faio.kz — unrestricted and unverified.

## Solution

Let `S` = sick, `+` = positive test.

```
P(S) = 0.01          P(+|S)  = 0.99
P(¬S) = 0.99         P(+|¬S) = 1 − 0.98 = 0.02
```

Total probability of a positive test:

```
P(+) = 0.01·0.99 + 0.99·0.02 = 0.0099 + 0.0198 = 0.0297
```

Bayes:

```
P(S|+) = 0.0099 / 0.0297 = 1/3 = 0.3333...
```

Answer:

```
0.3333
```

**Traps:**

- Answering `0.99` — confusing `P(+|S)` with `P(S|+)`. This is the whole point of the problem: a 99 %-accurate test on a 1 %-prevalence disease is wrong about two thirds of its positives.
- The 98 % figure is the **true-negative** rate. Using `0.98` as the false-positive rate inverts the healthy branch and gives a wildly different answer.
- The exact value is `1/3`, so the rounded answer is `0.3333`, not `0.3334`.
