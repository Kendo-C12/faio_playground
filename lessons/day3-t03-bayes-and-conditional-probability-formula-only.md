# T03 · Bayes and conditional probability — formula sheet — Day 3

**Anchor task(s):**
- [`faio-2025/qualification/task3_Medicine.md`](../faio-2025/qualification/task3_Medicine.md)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

- Conditional probability: $P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}$, for $P(B) > 0$.
- Multiplication: $P(A \cap B) = P(B) \cdot P(A \mid B) = P(A) \cdot P(B \mid A)$.
- Law of total probability, for a partition $A_1 \dots A_k$:

$$
P(B) = \sum_{i=1}^{k} P(A_i) \cdot P(B \mid A_i), \qquad P(B) = P(A) \cdot P(B \mid A) + P(\bar{A}) \cdot P(B \mid \bar{A})
$$

- Bayes — $P(A)$ prior, $P(B \mid A)$ likelihood, $P(A \mid B)$ posterior:

$$
P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}
$$

- Odds form: $\text{posterior odds} = \text{prior odds} \times \text{likelihood ratio}$, with $LR^{+} = \dfrac{\text{sensitivity}}{1 - \text{specificity}}$.
- Prevalence: $P(D) = \pi$, the base rate of the condition.
- Sensitivity = true positive rate = $P(+ \mid D)$; miss rate = $P(- \mid D) = 1 - \text{sensitivity}$.
- Specificity = true negative rate = $P(- \mid \bar{D})$; false positive rate = $P(+ \mid \bar{D}) = 1 - \text{specificity}$.
- $\text{PPV} = P(D \mid +) = \dfrac{\pi \cdot \text{sens}}{\pi \cdot \text{sens} + (1-\pi) \cdot (1 - \text{spec})}$.
- $\text{NPV} = P(\bar{D} \mid -) = \dfrac{(1-\pi) \cdot \text{spec}}{(1-\pi) \cdot \text{spec} + \pi \cdot (1 - \text{sens})}$.
- Independence restated: $P(A \mid B) = P(A) \iff A$ and $B$ independent.

## Values worth memorising

- Task 3: prevalence $0.01$, sensitivity $0.99$, specificity $0.98$ $\Rightarrow$ false positive rate $0.02$.
- $P(+) = 0.01 \cdot 0.99 + 0.99 \cdot 0.02 = 0.0099 + 0.0198 = 0.0297$.
- The full Bayes application:

$$
P(\text{sick} \mid +) = \frac{0.01 \cdot 0.99}{0.01 \cdot 0.99 + 0.99 \cdot 0.02} = \frac{0.0099}{0.0297} = \frac{1}{3} = 0.3333
$$

- $P(\text{sick} \mid -) = \dfrac{0.01 \cdot 0.01}{0.01 \cdot 0.01 + 0.99 \cdot 0.98} = \dfrac{0.0001}{0.9703} = 0.0001$, so $\text{NPV} = 0.9999$.
- $LR^{+} = 0.99 / 0.02 = 49.5$; prior odds $1:99$ $\times$ $49.5$ = $0.5:1$ $\Rightarrow$ posterior $1/3$. Same answer, one line.
- Per 10,000 people: 99 true positives, 198 false positives, 297 positives total.

## One-line traps

- Specificity 98% means the **false positive** rate is 2%, not 98% — this is the step that sinks most attempts.
- The healthy group is $99\times$ larger, so its 2% error outnumbers the sick group's 99% hit rate 2:1.
- $P(+ \mid \text{sick})$ and $P(\text{sick} \mid +)$ are different numbers; Bayes exists to convert between them.
- Never drop the prior: a 99%-accurate test on a 1% disease is still wrong two times in three.
- The denominator is $P(B)$ over the whole population, not just the positive-and-sick cell.
- Task 3 wants **four decimals**: write $0.3333$, not $1/3$ and not $33\%$.
