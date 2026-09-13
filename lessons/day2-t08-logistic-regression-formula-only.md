# T08 · Logistic regression — formula sheet (Day 2)

**Anchor task(s):**
- [`faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

- Linear score (logit): $z = w^\top x + b$
- Sigmoid, its value at 0 and its derivative:

$$
\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \sigma(0) = 0.5 \qquad \sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)
$$

- Binary prediction — label 1 iff $\hat{p} \ge 0.5$, i.e. $z \ge 0$:

$$
\hat{p} = P(y = 1 \mid x) = \sigma(w^\top x + b)
$$

- Inverse — the log-odds; $w_j$ is the change in log-odds per unit of $x_j$, $e^{w_j}$ the odds ratio:

$$
z = \operatorname{logit}(p) = \ln\frac{p}{1-p}
$$

- Log-loss (binary cross-entropy):

$$
L = -\frac{1}{n}\sum_{i=1}^{n}\left[\, y_i \ln \hat{p}_i + (1 - y_i)\ln(1 - \hat{p}_i) \,\right]
$$

- Gradient — the "residual $\times$ feature" form:

$$
\nabla_w L = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)\,x_i
$$

- Multinomial / softmax ($K$ classes, as in `ru`/`kaz`/`eng`):

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad z_k = w_k^\top x + b_k
$$

- Multi-class log-loss ($y$ one-hot):

$$
L = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \ln \hat{p}_{ik}
$$

- Penalised objective, scikit-learn's parameterisation:

$$
\min_{w}\; C \cdot \sum_{i=1}^{n} \text{loss}(x_i, y_i) + R(w)
$$

- L2 (ridge) — shrinks all weights, keeps them all non-zero, strictly convex:

$$
R(w) = \tfrac{1}{2}\lVert w \rVert_2^2 = \tfrac{1}{2}\sum_j w_j^2
$$

- L1 (lasso) — drives weights exactly to 0, so it selects features:

$$
R(w) = \lVert w \rVert_1 = \sum_j |w_j|
$$

- Elastic net, $\rho$ = `l1_ratio` (`saga` only):

$$
R(w) = \rho\lVert w \rVert_1 + \frac{1-\rho}{2}\lVert w \rVert_2^2
$$

- $C = 1/\lambda$: **inverse** regularisation strength. Large $C$ → weak penalty → large weights → fits harder. $C \to \infty$ is unregularised.

## Values worth memorising

- The winning settings in [`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb): `LogisticRegression(C=10, penalty="l2", solver="saga", max_iter=200, random_state=42)` on `TfidfVectorizer(sublinear_tf=True, max_features=20_000)`, graded with `classification_report` (macro-F1).
- Its grid search: `C` $\in \{0.01, 0.1, 1, 10\}$, `penalty` $\in$ {`l2`, `l1`}, `solver` $\in$ {`liblinear`, `saga`}, `class_weight` $\in$ {`None`, `balanced`}, `cv=3`, `scoring="f1_macro"`, `n_jobs=-1` — and the winner `C=10` sits at the grid's edge.
- `saga` is the solver that supports **both L1 and L2** and works on **large sparse** matrices (it also does elastic net). `liblinear` does L1 and L2 but is one-vs-rest only; `lbfgs` (the default) is L2-only.
- Defaults: `C=1.0`, `penalty="l2"`, `solver="lbfgs"`, `max_iter=100`, `multi_class` multinomial for the multinomial-capable solvers.
- `class_weight="balanced"` sets weight $n/(K \cdot n_c)$ for class $c$. One-vs-rest = $K$ binary fits then `argmax`, not calibrated across classes.

## One-line traps

- $C$ is **inverse** regularisation: larger $C$ = *less* regularisation. Raising it to fix overfitting does the opposite.
- `max_iter` too low → a `ConvergenceWarning` that is easy to miss in notebook output, and a silently worse model. The notebook searched at `max_iter=1000` and then shipped `200`; if the score moves, raise it.
- `lbfgs` + `penalty="l1"` is invalid; `saga` or `liblinear` only. An illegal pair inside a grid can abort the whole search.
- Scale your features for any solver except when the matrix is TF-IDF (already normalised) — unscaled columns make gradient solvers crawl and distort the shared penalty.
- `predict` hard-codes the 0.5 threshold; use `predict_proba` when the metric rewards a different one.
- A perfectly separable training set sends L2-free weights to infinity — regularisation is what keeps the fit finite.
