# T01 · Descriptive statistics — formula sheet (Day 1)

**Anchor task(s):**
- [`faio-2025/qualification/task1_stats_101.md`](../faio-2025/qualification/task1_stats_101.md)
- [`faio-2025/day1/qaz_letters.md`](../faio-2025/day1/qaz_letters.md)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

- Mean: `x̄ = (1/n) Σ xᵢ`
- Mode: the value with the highest frequency; a set may have several (multimodal) or none distinct.
- Median, sorted `x₍₁₎…x₍ₙ₎`: `n` odd → `x₍₍ₙ₊₁₎/₂₎`; `n` even → `(x₍ₙ/₂₎ + x₍ₙ/₂₊₁₎)/2`
- Population variance: `σ² = (1/n) Σ (xᵢ − x̄)²` · population sd: `σ = √σ²`
- Sample variance: `s² = (1/(n−1)) Σ (xᵢ − x̄)²` · sample sd: `s = √s²`
- Range: `max − min` · RMS: `√((1/n) Σ xᵢ²)` · CV: `σ / x̄`
- Skewness (Fisher, population): `g₁ = (1/n) Σ ((xᵢ − x̄)/σ)³` — 0 symmetric, >0 right tail
- Kurtosis (excess, population): `g₂ = (1/n) Σ ((xᵢ − x̄)/σ)⁴ − 3` — 0 for a normal distribution
- Percentile `p` (linear interpolation): rank `h = (n−1)·p/100`, value `x₍⌊h⌋₊₁₎ + (h−⌊h⌋)·(x₍⌊h⌋₊₂₎ − x₍⌊h⌋₊₁₎)`
- Quartiles: `Q1 = P25`, `Q2 = median = P50`, `Q3 = P75` · `IQR = Q3 − Q1`
- Outlier fence (Tukey): outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`
- Z-score: `z = (x − x̄)/σ` · Min-max: `(x − min)/(max − min)`

## Values worth memorising

- [`task1_stats_101.md`](../faio-2025/qualification/task1_stats_101.md) data `2, 0, 1, 3, 2, 1, 0, 4, 1, 1` → sorted `0 0 1 1 1 1 2 2 3 4`, `n = 10`, `Σ = 15`.
  Mean **1.5** · mode **1** (4 occurrences) · median **1** (5th and 6th both `1`). Answer line: `1.5 1 1` — mean, mode, median, in that order.
- Same data: population σ ≈ **1.2042** (σ² = 1.45), sample s ≈ **1.2693** (s² = 1.6111).
- `numpy` defaults to the **population** denominator (`ddof=0`); `pandas` `.std()`/`.var()` default to the **sample** one (`ddof=1`). `scipy.stats.skew`/`kurtosis` are population, and `kurtosis` returns **excess** (Fisher) by default.
- The feature list in [`qaz_letters.md`](../faio-2025/day1/qaz_letters.md) names mean intensity, standard deviation, min/max, skewness, kurtosis and 16-bin histograms — so these are the jury's own vocabulary for tabularising images.

## One-line traps

- Even `n`: the median is the **average of the two middle values**, not a single element — invisible in task 1 (both are `1`), fatal elsewhere.
- Population vs sample: divide by `n` or by `n−1`; state which you mean, and match the library default (`ddof`).
- Mode can be a tie — report the rule the statement implies, and never report a mean value as a mode.
- Do not round `1.5` to `2`; print the mean as a decimal.
- `std` of a single observation is `NaN` (sample) or `0` (population) — fill it before it reaches a model.
- Skewness and kurtosis need `σ` computed with the **same** `ddof` as the formula you are quoting, or the value shifts.
