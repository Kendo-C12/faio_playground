# T05 · Classification metrics — formula sheet — Day 3

**Anchor task(s):**
- [`faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

- Confusion matrix, per class `c`: `TP` predicted c and is c; `FP` predicted c, is not; `FN` is c, predicted otherwise; `TN` the rest.
- `Accuracy = (TP + TN) / (TP + TN + FP + FN)` = correct / total; multiclass: `Σ_c TP_c / N`.
- `Precision_c = TP_c / (TP_c + FP_c)` — of what you called c, how much was c.
- `Recall_c = TP_c / (TP_c + FN_c)` — of the true c, how much you found.
- `F1_c = 2·P·R / (P + R)` — harmonic mean; 0 if either term is 0.
- `Fβ = (1+β²)·P·R / (β²·P + R)`; β>1 favours recall, β<1 favours precision.
- **Macro**: `macro-F1 = (1/K)·Σ_c F1_c` — unweighted mean over the K classes.
- **Micro**: pool all TP/FP/FN across classes, then compute one F1. In single-label multiclass, `micro-F1 = accuracy`.
- **Weighted**: `Σ_c support_c·F1_c / N` — macro with class sizes as weights.
- `Balanced accuracy = (1/K)·Σ_c Recall_c` — macro-averaged recall.
- `Specificity_c = TN_c / (TN_c + FP_c)`; `FPR_c = 1 − Specificity_c`; `Cohen's κ = (p_o − p_e)/(1 − p_e)`.
- sklearn: `classification_report(y, ŷ, zero_division=0)`; `f1_score(y, ŷ, average="macro")`.

## Values worth memorising

- Task 4's metric: **macro-F1** over `ru`/`kaz`/`eng` — `classification_report` in the solution notebook, `scoring="f1_macro"` in its grid search. K = 3.
- Three classes, per-class F1 `0.99 / 0.99 / 0.60` ⇒ macro `0.86`, while accuracy can still read `0.97`.
- A class with 1% support contributes `1/K = 0.333` of macro-F1 and `0.01` of weighted-F1.
- `P = R = 0.9` ⇒ `F1 = 0.9`. `P = 1.0, R = 0.5` ⇒ `F1 = 0.667`. `P = 1.0, R = 0.1` ⇒ `F1 = 0.182`.
- Majority-class baseline on a 90/10 split: accuracy `0.90`, macro-F1 `0.474`.
- `f1_score(..., average="micro")` on single-label multiclass equals `accuracy_score(...)` exactly.

## One-line traps

- Macro-F1 weights every class equally, so one small weak class sinks the score while accuracy looks fine.
- Accuracy hides class imbalance: always print `y.value_counts()` next to it.
- A class the model never predicts gives `0/0` precision — pass `zero_division=0` or sklearn warns and returns 0.
- F1 is a harmonic mean, so it is dragged toward the *smaller* of precision and recall, never the average.
- `average` defaults to `"binary"` in `f1_score`; multiclass without `average="macro"` raises or misleads.
- Cross-validate with `scoring="f1_macro"` if you report macro-F1 — tuning on accuracy optimises the wrong thing.
