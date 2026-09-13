# T26 · FFT and spectral features — Day 4

**Anchor task(s):**
- [`faio-2025/day1/Missing_Fundamental_Puzzle.md`](../faio-2025/day1/Missing_Fundamental_Puzzle.md)
- [`faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md)

Rating 0 · **exam-probability rank 2** · ~50 min

## [concept-first]

**Sampling rate and Nyquist.** A digital signal is the amplitude measured $f_s$ times per second. The **Nyquist–Shannon** limit says a sampled signal can only represent frequencies below the Nyquist frequency

$$
f_{\text{Nyq}} = \frac{f_s}{2}
$$

Anything higher folds back (*aliases*) and masquerades as a lower frequency. At `Missing_Fundamental_Puzzle`'s $f_s = 22050$ Hz the usable band is therefore **0 – 11025 Hz**. Task 5's 200 Hz IMU gives 0 – 100 Hz, which is plenty for human motion.

**The FFT.** `np.fft.rfft(x)` decomposes $N$ samples into complex coefficients, one per frequency bin:

$$
X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i k n / N}
$$

`rfft` returns only the $N/2 + 1$ non-negative-frequency bins, which is all a real signal has. Take `np.abs(...)` for the **magnitude spectrum** — how much energy sits at each frequency. Phase is discarded and rarely matters for pitch.

**Bins and resolution.** Bin $k$ of an $N$-sample FFT sits at frequency

$$
f_k = \frac{k f_s}{N}
$$

so the spacing between bins — your frequency resolution — is

$$
\Delta f = \frac{f_s}{N}
$$

At $f_s = 22050$ and $N = 2048$ that is 10.8 Hz. That number decides what you can see: two harmonics 30 Hz apart are resolvable, a low note's 20 Hz spacing is not. Longer window means finer frequency resolution but coarser time resolution. That trade-off is unavoidable. `np.fft.rfftfreq(N, 1/fs)` gives the bin frequencies; never index bins by hand.

**Windowing.** The FFT assumes the $N$ samples repeat periodically. If the waveform does not line up end to end, the implied discontinuity smears energy across every bin (*spectral leakage*), burying weak harmonics. Multiply by a taper first — `x * np.hanning(N)` — which forces the ends to zero. Always window before an FFT of real audio.

**Spectrogram and log-mel.** A spectrogram slices the signal into overlapping windows and stacks their magnitude spectra into a time $\times$ frequency image. A **mel** spectrogram rebins the frequency axis onto the mel scale, which is roughly logarithmic — fine resolution low down, coarse high up, matching human hearing and compressing 1025 bins to ~128. Taking `log` of the magnitudes compresses the dynamic range so quiet harmonics are visible. A log-mel spectrogram is a 2-D array: feed it to a CNN (T30), or summarise each row into mean/std for a tabular model.

**Harmonics and the fundamental.** A played note at fundamental $f_0$ produces energy at integer multiples of it:

$$
f_k = k f_0, \qquad k = 1, 2, 3, \ldots
$$

Pitch is $f_0$. The crucial structural fact: the **spacing** between consecutive harmonics is also $f_0$. So $f_0$ is encoded twice over — once as the lowest peak, and once as the comb spacing of every peak above it.

**MIDI note numbers.** Pitch labels are integers with 69 = A4 = 440 Hz and 12 semitones per octave:

$$
f = 440 \cdot 2^{(n - 69)/12}
$$

$$
n = 69 + 12 \log_2\!\left(\frac{f}{440}\right)
$$

So $n = 60$ (C4) is $440 \cdot 2^{-9/12} \approx 261.6$ Hz, and one semitone is a ratio of $2^{1/12} \approx 1.0595$ — only ~6 %. With 61 classes spanning roughly five octaves, your $f_0$ estimate must be accurate to a few percent, and it is far safer to estimate $f_0$, convert with the formula, and round to the nearest integer $n$ than to ask a classifier to learn 61 unordered labels from scratch.

## [problem-first]

Open [`Missing_Fundamental_Puzzle.md`](../faio-2025/day1/Missing_Fundamental_Puzzle.md). Mono 16-bit PCM WAV, 22050 Hz, 1–4 seconds, 61 classes — one per MIDI note present in the data — submission `Path, Pitch_ID`, metric plain accuracy, dataset a variant of TinySOL.

Then the sentence that is the entire problem: "The audio has been preprocessed and **some of the frequencies have been removed from the signal**", and later, the model "must infer pitch from the spectral and temporal structure of the remaining harmonics, **without access to the fundamental frequency**".

Derive the consequence:

1. The naive pitch estimator — "take the lowest or strongest spectral peak" — returns $2f_0$ or $3f_0$, because $f_0$ itself was deleted. You will land an octave or a fifth too high, *systematically*, on every file. That is not noise you can average away.
2. But the remaining harmonics are still spaced by $f_0$. The spectrum is a comb with teeth at $2f_0, 3f_0, 4f_0, \ldots$; the gap between neighbouring teeth is $f_0$. Recovering the **spacing** recovers the pitch. This is exactly why a human still hears C4 when the C4 component is gone — the psychoacoustic "missing fundamental" the task is named after.
3. Three ways to measure that spacing, in increasing robustness: differences between detected peak frequencies; **autocorrelation of the magnitude spectrum**, whose first strong lag is $f_0$ in Hz; and the **cepstrum** (inverse FFT of the log magnitude spectrum), whose peak lag is the period $1/f_0$ in seconds. Autocorrelation of the spectrum is the one to write first.
4. Convert the estimated $f_0$ to a MIDI number with $69 + 12\log_2(f/440)$, round, and map to the label ids given in `train.csv`. Then — because 2,330 training files with known labels are available — measure how often you are exactly right, and how often you are off by exactly 12 or 19 semitones. Those two error modes are diagnostic: $+12$ means you locked onto $2f_0$ (one octave), $+19$ onto $3f_0$ (an octave and a fifth, since $12\log_2(3) \approx 19.02$).
5. A CNN over a log-mel spectrogram also works and learns the comb pattern implicitly, with 61 classes and 2,330 training files. Use it as the second model, and blend only if the explicit estimator disagrees with it often.

The same toolbox carries to [`task5`](../faio-2025/qualification/task5_Can_You_Become_AI_Yoga_Instructor.md): per-window spectral features of the IMU signals — dominant frequency, spectral centroid, band energy ratios below and above a few Hz — extend T25's battery, since tremor and jerky motion are high-frequency events that `mean` and `std` cannot see.

## [code-first]

```python
import numpy as np
import scipy.io.wavfile as wav

fs, x = wav.read("note.wav")                # fs = 22050 -> Nyquist = 11025 Hz
x = x.astype(np.float32) / 32768.0          # 16-bit PCM -> [-1, 1]

N = 8192                                    # resolution = fs/N = 2.7 Hz; long window for fine pitch
seg = x[:N] * np.hanning(N)                 # 1-4 s at 22050 Hz is always > N samples
                                            # taper first, or leakage buries weak harmonics
S = np.abs(np.fft.rfft(seg, N))             # magnitude spectrum, N//2 + 1 bins
freqs = np.fft.rfftfreq(N, 1 / fs)          # never index bins by hand

# WRONG for this task: the biggest peak is 2*f0 or 3*f0 because f0 was deleted.
naive_f0 = freqs[S.argmax()]

# RIGHT: the harmonics are still SPACED by f0. Autocorrelate the spectrum and read the lag.
A = np.correlate(S - S.mean(), S - S.mean(), mode="full")[len(S) - 1:]
df = freqs[1] - freqs[0]                    # Hz per bin = fs/N
lo, hi = int(50 / df), int(2000 / df)       # search 50-2000 Hz, the plausible f0 range
f0 = (lo + int(np.argmax(A[lo:hi]))) * df   # first strong lag = harmonic spacing = f0

def to_midi(f):                             # 69 = A4 = 440 Hz, 12 semitones per octave
    return int(round(69 + 12 * np.log2(f / 440.0)))

print("naive", to_midi(naive_f0), "| spacing", to_midi(f0))
# Off by +12 -> you locked onto 2*f0; by +19 -> onto 3*f0. Check the pattern on train.csv.

# Second model: log-mel spectrogram as a 2-D CNN input (see T30).
# import librosa; M = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=fs, n_mels=128))
```

## [drill]

1. $f_s = 22050$. What is the highest representable frequency, and what happens to a 13 kHz tone?
2. $N = 4096$ at $f_s = 22050$. Give the frequency resolution and the frequency of bin 100.
3. Why window before an FFT, and what artefact does skipping it produce?
4. A note's $f_0$ is deleted and the spectrum shows peaks at 440, 660, 880, 1100 Hz. What is $f_0$ and what MIDI number?
5. Your predictions are off by exactly $+12$ semitones on most files. What did your estimator lock onto?
6. Why prefer estimating $f_0$ and converting, over training a 61-way classifier directly?
7. You want finer frequency resolution. What do you lose?

<details><summary>Answers</summary>

1. 11025 Hz ($f_s/2$). A 13 kHz tone aliases, appearing as $22050 - 13000 = 9050$ Hz.
2. $22050 / 4096 \approx 5.38$ Hz per bin; bin 100 sits at $100 \cdot 5.38 \approx 538$ Hz.
3. The FFT treats the window as periodic, so mismatched ends create a discontinuity whose energy leaks across all bins and hides weak harmonics. A Hann taper forces the ends to zero.
4. The spacing is 220 Hz, so $f_0 = 220$ Hz (the peaks are harmonics 2, 3, 4, 5). $69 + 12\log_2(220/440) = 57$ — A3.
5. $2f_0$ — the lowest *surviving* harmonic. One octave is exactly 12 semitones, which is the signature of this mistake.
6. Pitch is ordered and physically determined, so one continuous estimate plus a formula generalises across all 61 classes; a 61-way classifier must learn each label separately from a few dozen examples each and cannot exploit the octave structure.
7. Time resolution. Resolution is $f_s/N$, so a longer window smears everything that changes during it — the unavoidable time-frequency trade-off.

</details>

**Rep:** take any training WAV, plot its magnitude spectrum, mark the peak frequencies, compute both `naive_f0` and the autocorrelation `f0`, convert both to MIDI, and compare against `train.csv`. Then tabulate the error distribution over 100 files and confirm the naive estimator's errors pile up at $+12$ and $+19$.

## Traps & 60-second recall

- Nyquist is $f_s/2$: 11025 Hz at 22050 Hz sampling. Above it, frequencies alias.
- Bin $k$ is at $k f_s/N$; resolution is $\Delta f = f_s/N$. Longer window = finer frequency, coarser time.
- Window (Hann) before every FFT of real audio.
- `rfft` + `np.abs` for the magnitude spectrum; `rfftfreq` for the axis.
- With the fundamental removed, the lowest peak is $2f_0$ — never read pitch off the biggest peak.
- Harmonic **spacing** is $f_0$: autocorrelate the spectrum, or use the cepstrum, whose peak lag is $1/f_0$.
- $f = 440 \cdot 2^{(n-69)/12}$, $n = 69 + 12\log_2(f/440)$; 69 = A4 = 440 Hz; one semitone $\approx 1.0595\times$.
- Errors of exactly $+12$ or $+19$ semitones are diagnostic: you locked onto $2f_0$ or $3f_0$.
- Log-mel spectrogram + CNN is the learned alternative; spectral features also extend T25's IMU battery.
