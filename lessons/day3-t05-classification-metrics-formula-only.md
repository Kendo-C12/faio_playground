# T05 · Classification metrics — formula sheet — Day 3

**Anchor task(s):**
- [`faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

- Confusion matrix, per class $c$: $TP$ predicted c and is c; $FP$ predicted c, is not; $FN$ is c, predicted otherwise; $TN$ the rest.
- Accuracy — correct / total; multiclass $\sum_c TP_c / N$:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

- Precision — of what you called c, how much was c; recall — of the true c, how much you found:

$$
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \qquad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}
$$

- F1 — harmonic mean, 0 if either term is 0; $\beta > 1$ favours recall, $\beta < 1$ favours precision:

$$
F1_c = \frac{2 \cdot P \cdot R}{P + R}, \qquad F_\beta = \frac{(1 + \beta^{2}) \cdot P \cdot R}{\beta^{2} \cdot P + R}
$$

- **Macro**: $\text{macro-}F1 = \dfrac{1}{K} \sum_c F1_c$ — unweighted mean over the $K$ classes.
- **Micro**: pool all TP/FP/FN across classes, then compute one F1. In single-label multiclass, $\text{micro-}F1 = \text{accuracy}$.
- **Weighted**: $\sum_c \text{support}_c \cdot F1_c \,/\, N$ — macro with class sizes as weights.
- $\text{Balanced accuracy} = \dfrac{1}{K} \sum_c \text{Recall}_c$ — macro-averaged recall.
- $\text{Specificity}_c = \dfrac{TN_c}{TN_c + FP_c}$; $FPR_c = 1 - \text{Specificity}_c$; Cohen's $\kappa = \dfrac{p_o - p_e}{1 - p_e}$.
- sklearn: `classification_report(y, ŷ, zero_division=0)`; `f1_score(y, ŷ, average="macro")`.

## Values worth memorising

- Task 4's metric: **macro-F1** over `ru`/`kaz`/`eng` — `classification_report` in the solution notebook, `scoring="f1_macro"` in its grid search. $K = 3$.
- Three classes, per-class F1 $0.99 / 0.99 / 0.60$ $\Rightarrow$ macro $0.86$, while accuracy can still read $0.97$.
- A class with 1% support contributes $1/K = 0.333$ of macro-F1 and $0.01$ of weighted-F1.
- $P = R = 0.9 \Rightarrow F1 = 0.9$. $P = 1.0,\ R = 0.5 \Rightarrow F1 = 0.667$. $P = 1.0,\ R = 0.1 \Rightarrow F1 = 0.182$.
- Majority-class baseline on a 90/10 split: accuracy $0.90$, macro-F1 $0.474$.
- `f1_score(..., average="micro")` on single-label multiclass equals `accuracy_score(...)` exactly.

## One-line traps

- Macro-F1 weights every class equally, so one small weak class sinks the score while accuracy looks fine.
- Accuracy hides class imbalance: always print `y.value_counts()` next to it.
- A class the model never predicts gives $0/0$ precision — pass `zero_division=0` or sklearn warns and returns 0.
- F1 is a harmonic mean, so it is dragged toward the *smaller* of precision and recall, never the average.
- `average` defaults to `"binary"` in `f1_score`; multiclass without `average="macro"` raises or misleads.
- Cross-validate with `scoring="f1_macro"` if you report macro-F1 — tuning on accuracy optimises the wrong thing.
