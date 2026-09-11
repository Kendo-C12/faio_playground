# FAIO 2024 — problem index

FAIO = **Fizmat AI Olympiad**, official site https://faio.kz/

Classification of every problem in [`faio-2024/`](../faio-2024), taken from the statements in this repo.

**16 problems total:** 10 qualification + 6 day 1. There is no day 2 in 2024.

| Category | Count |
|---|---|
| `math` | 3 |
| `competitive-programming` | 8 |
| `code (ML implementation)` | 5 |
| `ml-theory` | 0 |

Judge: Yandex.Contest for both rounds — contest 71291 (qualification), contest 69765 (day 1).

## Category legend

| Category | Meaning | Marker in the statement |
|---|---|---|
| `math` | Pen-and-paper. Every number is fixed in the statement, the answer is one value or a short tuple, no program needed. | An "Answer Format" / "Output" section and no input stream |
| `competitive-programming` | Classic CP: stated time and memory limit, stdin/stdout, exact-match auto-judge, algorithmic or closed-form solution written as code. | `Time limit: 1 second`, `Memory limit: 64 MB`, `standard input or input.txt` |
| `code (ML implementation)` | Implement or train a model / data algorithm, graded by a metric on hidden tests. | accuracy or a custom metric, threshold scoring, training data |
| `ml-theory` | Conceptual ML in prose — which model and why, metric trade-offs. No code. | — |

`ml-theory` is empty in 2024 (and in 2025). FAIO grades a numeric answer or a program's output, never prose reasoning.

## Qualification — 10 problems

Single file: [`faio-2024/qualification/README.md`](../faio-2024/qualification/README.md). Every problem is `1 s / 64 MB`, `standard input or input.txt` → `standard output or output.txt`.

| # | Problem | Category | Topic | Input | Output | Scoring |
|---|---|---|---|---|---|---|
| A | Choosing rocket | `competitive-programming` | Combinatorics | `n`, `k` natural | Number of ways | Exact match |
| B | Expected Value | `math` | Probability, geometry | None | Answer as `a/b` | Exact match |
| C | Flight Speed | `competitive-programming` | Calculus | `a`, `b`, `c`, `d` real, each on its own line, `0 < … < 1000` | Speed at time `d` | Exact match |
| D | Flight Speed 2: Data Lost | `competitive-programming` | Algebra, 3×3 solve | `x1 y1 x2 y2 x3 y3` on one line, all in (−1000, 1000) | `a b c` of the parabola | Exact match |
| E | Ray is behaving bad | `math` | Probability on a graph | None | Probability as a float | Exact match |
| F | Gravitational Matrix | `competitive-programming` | Linear algebra | `a b c`, then the two matrices, `1 ≤ a,b,c ≤ 300` | `a` lines of `c` reals | Exact match |
| G | Into Space | `competitive-programming` | Geometry, feasibility | Six natural numbers `a b c d e f` | `yes` or `no` | Exact match |
| H | Meteorites with Gold | `competitive-programming` | Expected value | `x g`, `0 < g ≤ x < 100000` | Expected minutes, float | Exact match |
| I | Ray Is Misbehaving 2: Interference | `competitive-programming` | Line fit, outliers | `n`, then `n` lines of `id x y z` (all `z = 1`) | Deviating ids on one line | Exact match |
| J | Dividing the Stars | `competitive-programming` | Computational geometry | `n`, then `n` neutron-star points, then `n` giant-star points, `1 < n < 10^5` | `a b` of `y = ax + b` | Exact match |

**Rollup:** 8 `competitive-programming`, 2 `math`, 0 `code (ML implementation)`, 0 `ml-theory`.
By topic: probability / expected value 3 (B, E, H), geometry 2 (G, J), algebra & calculus 2 (C, D), combinatorics 1 (A), linear algebra 1 (F), outlier detection 1 (I).

### Solving notes

- **A** — binomial coefficient `C(n, k)`. Use exact integer arithmetic, not factorials through floats.
- **B** — no code reasoning: M is uniform in the triangle, so the expected height over AB is a third of the triangle's height. Answer `1/3`. Print the fraction literally; do not evaluate to a decimal.
- **C** — the speed is the derivative: `y' = 2ax + b`, so the answer is `2ad + b`. The sample confirms it: `a=10, b=4.65116, d=12` → `244.65116`. Watch the input layout: the statement says one number per line.
- **D** — three points give three linear equations in `a, b, c`. The second point is not given directly — it is `(x1 − x2, y3 + y2)`. Read that sentence twice before solving.
- **E** — enumerate by hand. From the first floor, any path of 3 steps reaching floor 4 must go floor 1 → 2 → 3 → 4. At a floor-2 vertex the outgoing stairs are: 2 neighbours on floor 2, plus the floor-3 links, plus back down; count the degrees from the statement before dividing. No input, so the program just prints the constant.
- **F** — plain triple-loop matrix product; `300³ = 2.7·10⁷` fits the 1 s limit in a compiled language, but in Python use numpy. 64 MB is the real constraint on how you store the reals.
- **G** — six given distances must be realisable by 4 coplanar points. Check the triangle inequality on every triple and then the planarity condition (the Cayley–Menger determinant of 4 coplanar points vanishes). Brute-forcing coordinates will not pass.
- **H** — expected position of the first gold meteorite among `x` items with `g` gold is `(x + 1) / (g + 1)`. The `1 1` → `1` sample is consistent.
- **I** — fit a line through the meteorite coordinates (all `z = 1`, so it is a 2D fit), then report ids whose deviation exceeds 40 %. Define the deviation against the fitted line, and fit robustly — the outliers you are hunting also drag a least-squares fit.
- **J** — candidate lines pass through one neutron and one giant star; the guarantee that no three stars are collinear makes counting each side well defined. `n < 10^5` means all `n²` pairs is too many — pick a point on the convex hull and rotate.

## Day 1 (main round) — 6 problems

Directory: [`faio-2024/day1/`](../faio-2024/day1). These carry Input/Output formats and a judge, but **no stated time or memory limit**, and they are scored by metric thresholds rather than exact match — which is why they are not `competitive-programming`.

| # | Problem | File | Category | Topic | Input | Output | Scoring |
|---|---|---|---|---|---|---|---|
| A | Math Test Results | [`A_Math_Test_Results.md`](../faio-2024/day1/A_Math_Test_Results.md) | `math` | Statistics | None — 30 students, mean 75, median 78, sd 5, fixed in the statement | Two numbers separated by a space | Exact match; the file already prints the answer `76.00 83` |
| B | The Story of a Meteorologist | [`B_The_story_of_a_meteorologist.md`](../faio-2024/day1/B_The_story_of_a_meteorologist.md) | `code (ML implementation)` | Outlier detection | One line of `N` ice-purity values | All outliers on one line | Match against the expected outlier set; Python 3.12.3 |
| C | Radioactive Decay | [`C_radioactive_decay.md`](../faio-2024/day1/C_radioactive_decay.md) | `code (ML implementation)` | Linear regression | `n`, then `n` lines `m_i t_i`; `1 < n ≤ 1000` | Half-life `T_0` | 100 points if within 1 % of truth, else 0; averaged over tests |
| D | K-means | [`D_Kmeans.md`](../faio-2024/day1/D_Kmeans.md) | `code (ML implementation)` | Clustering | `n` points, then `k`, then `k` initial centroids; `n < 1000`, `k < 10` | `k` final centroid coordinates | Metric `m = 1 − L/L_baseline` in 0–1; 100 points if ≥ 0.95 × reference, else 0 |
| E | Text Classification | [`E_Text_Classification.md`](../faio-2024/day1/E_Text_Classification.md) | `code (ML implementation)` | NLP classification | `train_set.csv` (≈50k rows, 95 tags) offline; hidden set of `N ≤ 10000` texts via `sys.stdin` | One string, tags separated by `", "` | Percent of correctly predicted tags |
| F | SVM | [`F_SVM.md`](../faio-2024/day1/F_SVM.md) | `code (ML implementation)` | Linear classification | `k`, then `k` lines `x_i y_i l_i`; `1000 < k < 4000` | `a b c` of the line | 0 if accuracy < 0.97, else `100 × accuracy`; averaged over tests |

**Rollup:** 5 `code (ML implementation)`, 1 `math`, 0 `competitive-programming`, 0 `ml-theory`.
By topic: classical ML from scratch 3 (C, D, F), statistics 1 (A), outlier detection 1 (B), NLP 1 (E).

### Solving notes

- **A** — new mean is `(75·30 + 2·90) / 32 = 76.00`. For the second part the standard deviation changes once the two 90s are added; recompute it, then the threshold is `median + new sd` and Darkhan needs to exceed it. Answer in the file: `76.00 83`. Good warm-up for checking your own statistics code.
- **B** — `Q1`, `Q3`, `IQR = Q3 − Q1`, flag anything outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`. The trap is the quartile convention: the expected output in the file is a specific set, so match the percentile method (inclusive vs exclusive) that reproduces it. The statement also describes Z-score, but the task asks for IQR.
- **C** — log-transform: `ln m = ln m_0 − (ln 2 / T_0) · t`, so fit a straight line in `(t, ln m)` and read `T_0 = ln 2 / (−slope)`. The file's own hint says to use the natural logarithm. The 1 % tolerance is generous, so ordinary least squares is enough.
- **D** — standard Lloyd iterations, but initial centroids are given, so do not re-seed them and do not shuffle — the grader compares against a reference run. Iterate to convergence and print centroid coordinates, not labels.
- **E** — build the tag → keyword map from `train_set.csv` (tab-separated), then score each hidden text by keyword overlap. 95 classes and a single-line, comma-space-joined output mean an off-by-one in row order destroys the score; keep the order of the input rows.
- **F** — gradient descent on a hinge-style loss to get `a, b, c`. Accuracy below 0.97 scores zero, so normalise the features and check separation before submitting; the data is guaranteed linearly separable, so failing the threshold means a bug or too few iterations.

## Known defects in the archive files

These are faults in the upstream task archive, left as-is in this repo:

- [`faio-2024/qualification/README.md`](../faio-2024/qualification/README.md) repeats the heading `# C. Flight Speed` twice — once before the link, once after.
- Problem **B** has no limit block, while the other 9 do. **A** has the limits as plain text rather than the bold form used from C onward, so a grep for `Memory limit` finds 9 hits across 10 problems.
- **F**'s statement drops into Russian for one paragraph ("Точку $(x, y)$ будем классифицировать…") and **C** in day 1 ends with a Russian hint section.

## See also

- [FAIO 2025 problem index](./faio-2025-problems.md)
- Upstream source of the task files: https://github.com/abglnv/faio-tasks
