# T03 · Bayes and conditional probability — formula sheet — Day 3

**Anchor task(s):**
- [`faio-2025/qualification/task3_Medicine.md`](../faio-2025/qualification/task3_Medicine.md)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

- Conditional probability: `P(A|B) = P(A ∩ B) / P(B)`, for `P(B) > 0`.
- Multiplication: `P(A ∩ B) = P(B)·P(A|B) = P(A)·P(B|A)`.
- Law of total probability, for a partition `A₁…Aₖ`: `P(B) = Σᵢ P(Aᵢ)·P(B|Aᵢ)`.
- Two-case version: `P(B) = P(A)·P(B|A) + P(Ā)·P(B|Ā)`.
- Bayes: `P(A|B) = P(B|A)·P(A) / P(B)`; `P(A)` prior, `P(B|A)` likelihood, `P(A|B)` posterior.
- Odds form: `posterior odds = prior odds × likelihood ratio`, `LR⁺ = sensitivity / (1 − specificity)`.
- Prevalence: `P(D) = π`, the base rate of the condition.
- Sensitivity = true positive rate = `P(+|D)`; miss rate = `P(−|D) = 1 − sensitivity`.
- Specificity = true negative rate = `P(−|D̄)`; false positive rate = `P(+|D̄) = 1 − specificity`.
- `PPV = P(D|+) = π·sens / (π·sens + (1−π)·(1−spec))`.
- `NPV = P(D̄|−) = (1−π)·spec / ((1−π)·spec + π·(1−sens))`.
- Independence restated: `P(A|B) = P(A)` ⟺ A and B independent.

## Values worth memorising

- Task 3: prevalence `0.01`, sensitivity `0.99`, specificity `0.98` ⇒ false positive rate `0.02`.
- `P(+) = 0.01·0.99 + 0.99·0.02 = 0.0099 + 0.0198 = 0.0297`.
- `P(sick|+) = 0.0099 / 0.0297 = 1/3 = 0.3333`.
- `P(sick|−) = (0.01·0.01) / (0.01·0.01 + 0.99·0.98) = 0.0001/0.9703 = 0.0001`, so `NPV = 0.9999`.
- `LR⁺ = 0.99 / 0.02 = 49.5`; prior odds `1:99` × 49.5 = `0.5:1` ⇒ posterior `1/3`. Same answer, one line.
- Per 10,000 people: 99 true positives, 198 false positives, 297 positives total.

## One-line traps

- Specificity 98% means the **false positive** rate is 2%, not 98% — this is the step that sinks most attempts.
- The healthy group is 99× larger, so its 2% error outnumbers the sick group's 99% hit rate 2:1.
- `P(+|sick)` and `P(sick|+)` are different numbers; Bayes exists to convert between them.
- Never drop the prior: a 99%-accurate test on a 1% disease is still wrong two times in three.
- The denominator is `P(B)` over the whole population, not just the positive-and-sick cell.
- Task 3 wants **four decimals**: write `0.3333`, not `1/3` and not `33%`.
