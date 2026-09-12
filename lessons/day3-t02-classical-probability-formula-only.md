# T02 · Classical probability — formula sheet — Day 3

**Anchor task(s):**
- [`faio-2025/qualification/task2_Probability_theory_101.md`](../faio-2025/qualification/task2_Probability_theory_101.md)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

- Classical probability: `P(A) = m / n` — `m` favourable equally likely outcomes, `n` total.
- Bounds: `0 ≤ P(A) ≤ 1`, `P(Ω) = 1`, `P(∅) = 0`.
- Complement: `P(Ā) = 1 − P(A)`.
- Addition rule: `P(A ∪ B) = P(A) + P(B) − P(A ∩ B)`; mutually exclusive ⇒ `P(A ∩ B) = 0`.
- Independence / multiplication: `P(A ∩ B) = P(A)·P(B)` if independent, else `P(A)·P(B|A)`.
- n independent repeats of an event of probability `p`: all occur = `pⁿ`; none occurs = `(1−p)ⁿ`.
- **At least one** pattern: `P(≥1) = 1 − P(none) = 1 − (1 − p)ⁿ`.
- Exactly k of n (binomial): `P(k) = C(n,k) · pᵏ · (1−p)ⁿ⁻ᵏ`.
- Permutations of n distinct items: `n!`. Ordered choices of k from n: `P(n,k) = n! / (n−k)!`.
- Combinations: `C(n,k) = n! / (k!·(n−k)!)`, and `C(n,k) = C(n,n−k)`.
- Ordered outcomes of n rolls of a d-sided die: `dⁿ`.

## Values worth memorising

- Task 2, three rolls of a fair die, at least one six: `1 − (5/6)³ = 1 − 125/216 = 91/216 = 0.4213`.
- `(5/6)¹ = 0.8333`, `(5/6)² = 0.6944`, `(5/6)³ = 0.5787`, `(5/6)⁴ = 0.4823`.
- `6¹ = 6`, `6² = 36`, `6³ = 216`, `6⁴ = 1296`.
- One six in three rolls, exactly: `3 · (1/6) · (5/6)² = 75/216 = 0.3472`.
- `1/6 = 0.1667`, `1/3 = 0.3333`, `1/2 = 0.5`.
- `C(6,2) = 15`, `C(6,3) = 20`, `C(52,2) = 1326`, `4! = 24`, `5! = 120`, `6! = 720`.

## One-line traps

- "At least one" is the complement, never a sum of overlapping cases — `1 − (5/6)³`, not `3·(1/6)`.
- "At least one six" ≠ "exactly one six": `0.4213` vs `0.3472`.
- The addition rule needs `− P(A ∩ B)` unless the events are genuinely disjoint.
- Independence is an assumption: dice yes, draws without replacement no.
- Count ordered outcomes consistently in both numerator and denominator, or `m/n` is meaningless.
- Task 2 wants **four decimals** as a decimal fraction: write `0.4213`, not `91/216` and not `0.42`.
