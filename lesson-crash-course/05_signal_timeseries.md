# 05 · Sensors, signals and audio

The whole trick: **turn the signal into a table, then use [02](./02_tabular_models.md).** Almost nothing here requires a model that understands time.

## The decision table

| Situation | Approach | Takes |
|---|---|---|
| **Sensor readings, one label per recording** | window summary features + gradient boosting | 20 min |
| The pattern repeats at some frequency | add FFT features | +10 min |
| Audio, predict a category | mel-spectrogram statistics + gradient boosting | 30 min |
| Audio, need higher accuracy | pretrained audio embeddings + `LogisticRegression` | 45 min |
| One long series, predict the future | lag features + gradient boosting | 30 min |
| Deep learning on raw waveforms | **skip it** | — |

## 1. Window features — the core technique

Many rows describe one example. Collapse them into a single row of summary numbers.

```python
import numpy as np, pandas as pd

SIG = ["ax", "ay", "az", "wx", "wy", "wz"]

def featurise(df):
    d = df.copy()
    # Magnitude does not change when the sensor is mounted at a different angle
    d["amag"] = np.sqrt(d.ax**2 + d.ay**2 + d.az**2)
    d["wmag"] = np.sqrt(d.wx**2 + d.wy**2 + d.wz**2)
    cols = SIG + ["amag", "wmag"]

    g = d.groupby("id")[cols]
    agg = g.agg(["mean", "std", "min", "max", "median"])
    agg.columns = ["_".join(c) for c in agg.columns]    # flatten immediately

    agg["n_rows"] = d.groupby("id").size()              # duration
    for c in cols:
        agg[f"{c}_rms"]   = g[c].apply(lambda s: np.sqrt((s**2).mean()))
        agg[f"{c}_range"] = g[c].max() - g[c].min()
        agg[f"{c}_zcr"]   = g[c].apply(lambda s: np.mean(np.diff(np.sign(s)) != 0))

    for a, b in [("ax","ay"), ("ax","az"), ("wx","wy")]:
        agg[f"corr_{a}_{b}"] = d.groupby("id").apply(lambda s: s[a].corr(s[b]))

    return agg.fillna(0)
```

Which features carry signal, and why:

| Feature | Captures |
|---|---|
| `mean` | the resting level — often the weakest feature |
| `std`, `range` | how much movement there was — usually the strongest |
| `rms` | energy of the signal |
| `zcr` (zero-crossing rate) | how often it changes direction, i.e. vibration |
| `n_rows` | duration, at a fixed sampling rate |
| `corr` between axes | coordination between directions |
| magnitude columns | the same motion regardless of sensor orientation |

**The validation trap:** if one example spans many rows, its rows must never be split across folds — consecutive samples are near-identical and a random split scores the model against copies of its own training data. Use `StratifiedGroupKFold(groups=id)`, or aggregate first as above.

Docs: [pandas groupby](https://pandas.pydata.org/docs/user_guide/groupby.html)

## 2. FFT features — when something repeats

```python
from scipy.fft import rfft, rfftfreq

def spectral_features(signal, fs=200):
    """fs = sampling rate in Hz, e.g. 200 readings per second."""
    n = len(signal)
    mag = np.abs(rfft(signal - signal.mean()))     # remove the constant offset first
    freq = rfftfreq(n, 1 / fs)

    total = mag.sum() + 1e-12
    bands = [(0, 3), (3, 8), (8, 20), (20, 50)]    # energy per frequency band, Hz
    feats = {f"band_{lo}_{hi}": mag[(freq >= lo) & (freq < hi)].sum() / total
             for lo, hi in bands}
    feats["dominant_freq"] = freq[np.argmax(mag)]
    feats["spectral_centroid"] = float((freq * mag).sum() / total)
    return feats
```

**Use when:** the label depends on rhythm, vibration or pitch rather than magnitude.

Two facts worth memorising:

- **Nyquist limit:** at a sampling rate $f_s$, the highest measurable frequency is $f_s/2$. At 200 Hz, nothing above 100 Hz exists in the data.
- **Bin width:** $\Delta f = f_s / N$. Longer recordings give finer frequency resolution.

Docs: https://docs.scipy.org/doc/scipy/tutorial/fft.html

## 3. Audio — mel-spectrogram statistics

```python
import librosa

def audio_features(path):
    y, sr = librosa.load(path, sr=22050, mono=True)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    parts = [mel_db.mean(axis=1), mel_db.std(axis=1),
             mfcc.mean(axis=1), mfcc.std(axis=1),
             [librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
              librosa.feature.zero_crossing_rate(y).mean(),
              float(len(y) / sr)]]
    return np.concatenate([np.asarray(p).ravel() for p in parts])
```

Feed the result to gradient boosting like any table. **MFCCs** are the standard compact summary of a sound's character; mel bands describe loudness per pitch region.

**For pitch specifically:** the spacing between harmonics identifies a note even when the lowest tone is absent, so look at the *pattern* of peaks rather than the lowest peak.

```python
f0 = librosa.yin(y, fmin=50, fmax=2000, sr=sr)      # pitch estimate per frame
```

Docs: https://librosa.org/doc/latest/feature.html
Article: [librosa audio feature tutorial](https://librosa.org/doc/latest/tutorial.html)

## 4. Forecasting one long series

```python
def lag_features(df, col="value", lags=(1, 2, 3, 7, 14), windows=(3, 7, 14)):
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[col].shift(L)
    for W in windows:
        out[f"roll_mean_{W}"] = out[col].shift(1).rolling(W).mean()
        out[f"roll_std_{W}"]  = out[col].shift(1).rolling(W).std()
    out["dow"]   = out.index.dayofweek
    out["month"] = out.index.month
    return out.dropna()
```

**Every feature must use `.shift(1)` or more.** A rolling mean that includes today leaks the answer.

**Split chronologically, never randomly:**

```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)
```

Docs: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split

## 5. Pretrained audio embeddings — if accuracy matters more than time

```python
from transformers import pipeline
extractor = pipeline("feature-extraction", model="facebook/wav2vec2-base")
```

**Use when:** the handcrafted route has plateaued and a GPU is available. Otherwise the features above are faster and easier to debug.

## What to skip

- LSTM and GRU networks on raw sequences — slow to write, slow to train, rarely better than boosting on summary features at contest scale
- Transformers for time series
- ARIMA and SARIMA — more assumptions and diagnostics than a contest deadline allows
