# T08 · Logistic regression — formula sheet (Day 2)

**Anchor task(s):**
- [`faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

- Linear score (logit): `z = wᵀx + b`
- Sigmoid: `σ(z) = 1/(1 + e^{−z})` · `σ(0) = 0.5` · `σ'(z) = σ(z)(1 − σ(z))`
- Binary prediction: `p̂ = P(y=1|x) = σ(wᵀx + b)`; label 1 iff `p̂ ≥ 0.5`, i.e. `z ≥ 0`
- Inverse: `z = logit(p) = ln(p/(1−p))` — the log-odds; `w_j` is the change in log-odds per unit of `x_j`, `e^{w_j}` the odds ratio
- Log-loss (binary cross-entropy): `L = −(1/n) Σ [yᵢ ln p̂ᵢ + (1−yᵢ) ln(1−p̂ᵢ)]`
- Gradient: `∇_w L = (1/n) Σ (p̂ᵢ − yᵢ) xᵢ` — the "residual × feature" form
- Multinomial / softmax (`K` classes, as in `ru`/`kaz`/`eng`): `p̂_k = e^{z_k} / Σ_j e^{z_j}`, `z_k = w_kᵀx + b_k`
- Multi-class log-loss: `L = −(1/n) Σᵢ Σ_k y_{ik} ln p̂_{ik}` (`y` one-hot)
- Penalised objective, scikit-learn's parameterisation: `min_w  C · Σᵢ loss(xᵢ, yᵢ) + R(w)`
- L2 (ridge): `R(w) = ½‖w‖₂² = ½ Σ w_j²` — shrinks all weights, keeps them all non-zero, strictly convex
- L1 (lasso): `R(w) = ‖w‖₁ = Σ |w_j|` — drives weights exactly to 0, so it selects features
- Elastic net: `R(w) = ρ‖w‖₁ + ((1−ρ)/2)‖w‖₂²`, `ρ = l1_ratio` (`saga` only)
- `C = 1/λ`: **inverse** regularisation strength. Large `C` → weak penalty → large weights → fits harder. `C → ∞` is unregularised.

## Values worth memorising

- The winning settings in [`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb): `LogisticRegression(C=10, penalty="l2", solver="saga", max_iter=200, random_state=42)` on `TfidfVectorizer(sublinear_tf=True, max_features=20_000)`, graded with `classification_report` (macro-F1).
- Its grid search: `C ∈ {0.01, 0.1, 1, 10}`, `penalty ∈ {l2, l1}`, `solver ∈ {liblinear, saga}`, `class_weight ∈ {None, balanced}`, `cv=3`, `scoring="f1_macro"`, `n_jobs=-1` — and the winner `C=10` sits at the grid's edge.
- `saga` is the solver that supports **both L1 and L2** and works on **large sparse** matrices (it also does elastic net). `liblinear` does L1 and L2 but is one-vs-rest only; `lbfgs` (the default) is L2-only.
- Defaults: `C=1.0`, `penalty="l2"`, `solver="lbfgs"`, `max_iter=100`, `multi_class` multinomial for the multinomial-capable solvers.
- `class_weight="balanced"` sets weight `n/(K·n_c)` for class `c`. One-vs-rest = `K` binary fits then `argmax`, not calibrated across classes.

## One-line traps

- `C` is **inverse** regularisation: larger `C` = *less* regularisation. Raising it to fix overfitting does the opposite.
- `max_iter` too low → a `ConvergenceWarning` that is easy to miss in notebook output, and a silently worse model. The notebook searched at `max_iter=1000` and then shipped `200`; if the score moves, raise it.
- `lbfgs` + `penalty="l1"` is invalid; `saga` or `liblinear` only. An illegal pair inside a grid can abort the whole search.
- Scale your features for any solver except when the matrix is TF-IDF (already normalised) — unscaled columns make gradient solvers crawl and distort the shared penalty.
- `predict` hard-codes the 0.5 threshold; use `predict_proba` when the metric rewards a different one.
- A perfectly separable training set sends L2-free weights to infinity — regularisation is what keeps the fit finite.
