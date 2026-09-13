# T01 · Descriptive statistics — formula sheet (Day 1)

**Anchor task(s):**
- [`faio-2025/qualification/task1_stats_101.md`](../faio-2025/qualification/task1_stats_101.md)
- [`faio-2025/day1/qaz_letters.md`](../faio-2025/day1/qaz_letters.md)

Rating 3 · **exam-probability rank 1** · ~10 min

## Formulas

Mode is the value with the highest frequency; a set may have several (multimodal) or none distinct. Everything else, for a sample $x_1, \dots, x_n$ with sorted order $x_{(1)} \le \dots \le x_{(n)}$:

$$
\text{Mean } \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i \qquad \text{Median} = \begin{cases} x_{((n+1)/2)}, & n \text{ odd} \\ \dfrac{x_{(n/2)} + x_{(n/2+1)}}{2}, & n \text{ even} \end{cases}
$$

**Spread and shape** — population divides by $n$, sample by $n-1$; Fisher skewness $g_1$ is 0 for a symmetric sample and $>0$ for a right tail, excess kurtosis $g_2$ is 0 for a normal distribution:

$$
\begin{gathered}
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2 \quad \sigma = \sqrt{\sigma^2} \qquad s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2 \quad s = \sqrt{s^2} \\
\text{Range} = \max - \min \qquad \text{RMS} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2} \qquad \text{CV} = \frac{\sigma}{\bar{x}} \\
g_1 = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{\sigma}\right)^{3} \qquad g_2 = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{\sigma}\right)^{4} - 3
\end{gathered}
$$

**Position and scaling** — percentile $p$ by linear interpolation at rank $h = (n-1)\,p/100$, with $\lfloor h \rfloor$ the integer part:

$$
\begin{gathered}
P_p = x_{(\lfloor h \rfloor + 1)} + (h - \lfloor h \rfloor)\bigl(x_{(\lfloor h \rfloor + 2)} - x_{(\lfloor h \rfloor + 1)}\bigr) \\
Q_1 = P_{25} \quad Q_2 = \text{median} = P_{50} \quad Q_3 = P_{75} \qquad \text{IQR} = Q_3 - Q_1 \qquad \text{Tukey fence: outside } [\,Q_1 - 1.5\,\text{IQR},\; Q_3 + 1.5\,\text{IQR}\,] \\
z = \frac{x - \bar{x}}{\sigma} \qquad \text{Min-max} = \frac{x - \min}{\max - \min}
\end{gathered}
$$

## Values worth memorising

- [`task1_stats_101.md`](../faio-2025/qualification/task1_stats_101.md) data `2, 0, 1, 3, 2, 1, 0, 4, 1, 1` → sorted `0 0 1 1 1 1 2 2 3 4`, $n = 10$, $\sum_i x_i = 15$.
  Mean **1.5** · mode **1** (4 occurrences) · median **1** (5th and 6th both `1`). Answer line: `1.5 1 1` — mean, mode, median, in that order.
- Same data: population $\sigma \approx$ **1.2042** ($\sigma^2 = 1.45$), sample $s \approx$ **1.2693** ($s^2 = 1.6111$).
- `numpy` defaults to the **population** denominator (`ddof=0`); `pandas` `.std()`/`.var()` default to the **sample** one (`ddof=1`). `scipy.stats.skew`/`kurtosis` are population, and `kurtosis` returns **excess** (Fisher) by default.
- The feature list in [`qaz_letters.md`](../faio-2025/day1/qaz_letters.md) names mean intensity, standard deviation, min/max, skewness, kurtosis and 16-bin histograms — so these are the jury's own vocabulary for tabularising images.

## One-line traps

- Even $n$: the median is the **average of the two middle values**, not a single element — invisible in task 1 (both are `1`), fatal elsewhere.
- Population vs sample: divide by $n$ or by $n-1$; state which you mean, and match the library default (`ddof`).
- Mode can be a tie — report the rule the statement implies, and never report a mean value as a mode.
- Do not round `1.5` to `2`; print the mean as a decimal.
- `std` of a single observation is `NaN` (sample) or `0` (population) — fill it before it reaches a model.
- Skewness and kurtosis need $\sigma$ computed with the **same** `ddof` as the formula you are quoting, or the value shifts.
