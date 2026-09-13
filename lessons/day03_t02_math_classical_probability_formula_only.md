# T02 · Classical probability — formula sheet — Day 3

**Anchor task(s):**
- [`faio-2025/qualification/task2_Probability_theory_101.md`](../faio-2025/qualification/task2_Probability_theory_101.md)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

- Classical probability: $P(A) = \dfrac{m}{n}$ — $m$ favourable equally likely outcomes, $n$ total.
- Bounds: $0 \le P(A) \le 1$, $P(\Omega) = 1$, $P(\varnothing) = 0$.
- Complement: $P(\bar{A}) = 1 - P(A)$.
- Addition rule: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$; mutually exclusive $\Rightarrow P(A \cap B) = 0$.
- Independence / multiplication: $P(A \cap B) = P(A) \cdot P(B)$ if independent, else $P(A) \cdot P(B \mid A)$.
- $n$ independent repeats of an event of probability $p$: all occur $= p^{n}$; none occurs $= (1-p)^{n}$.
- **At least one** pattern:

$$
P(\ge 1) = 1 - P(\text{none}) = 1 - (1-p)^{n}
$$

- Exactly $k$ of $n$ (binomial): $P(k) = C(n,k) \cdot p^{k} \cdot (1-p)^{n-k}$.
- Permutations of $n$ distinct items: $n!$. Ordered choices of $k$ from $n$: $P(n,k) = \dfrac{n!}{(n-k)!}$.
- Combinations:

$$
C(n,k) = \frac{n!}{k! \cdot (n-k)!}, \qquad C(n,k) = C(n,\,n-k)
$$

- Ordered outcomes of $n$ rolls of a $d$-sided die: $d^{n}$.

## Values worth memorising

- Task 2, three rolls of a fair die, at least one six:

$$
1 - \left(\frac{5}{6}\right)^{3} = 1 - \frac{125}{216} = \frac{91}{216} = 0.4213
$$

- $(5/6)^{1} = 0.8333$, $(5/6)^{2} = 0.6944$, $(5/6)^{3} = 0.5787$, $(5/6)^{4} = 0.4823$.
- $6^{1} = 6$, $6^{2} = 36$, $6^{3} = 216$, $6^{4} = 1296$.
- One six in three rolls, exactly: $3 \cdot \dfrac{1}{6} \cdot \left(\dfrac{5}{6}\right)^{2} = \dfrac{75}{216} = 0.3472$.
- $1/6 = 0.1667$, $1/3 = 0.3333$, $1/2 = 0.5$.
- $C(6,2) = 15$, $C(6,3) = 20$, $C(52,2) = 1326$, $4! = 24$, $5! = 120$, $6! = 720$.

## One-line traps

- "At least one" is the complement, never a sum of overlapping cases — $1 - (5/6)^{3}$, not $3 \cdot (1/6)$.
- "At least one six" $\ne$ "exactly one six": $0.4213$ vs $0.3472$.
- The addition rule needs $-\,P(A \cap B)$ unless the events are genuinely disjoint.
- Independence is an assumption: dice yes, draws without replacement no.
- Count ordered outcomes consistently in both numerator and denominator, or $m/n$ is meaningless.
- Task 2 wants **four decimals** as a decimal fraction: write $0.4213$, not $91/216$ and not $0.42$.
