# T34 · Speed and memory on large inputs — Day 5

**Anchor task(s):**
- [`faio-2025/qualification/task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md)
- [`faio-2025/day2/Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md)
- [`faio-2025/day1/host-author-s-baseline.ipynb`](../faio-2025/day1/host-author-s-baseline.ipynb)

Rating 0 · **exam-probability rank 1** · ~40 min

In a 4-hour round, wall-clock is a scoring term. A correct pipeline that needs five hours scores zero.

## [concept-first]

**Do the arithmetic before you launch.** [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md) says "Your solution will be tested on **8000** images", each "$2048 \times 2048$ pixels".

- $0.4 \text{ s/image} \times 8000 = 3200 \text{ s} = 53$ min — a fifth of the whole round
- $1.0 \text{ s/image} \times 8000 = 8000 \text{ s} = 2$ h 13 min — round over
- $0.1 \text{ s/image} \times 8000 = 800 \text{ s} = 13$ min — fine

So the per-image budget is roughly 0.1–0.2 s, and you know that *before* writing the loop. Measure ten images, multiply by 800, and decide. This one calculation is the most valuable habit in this lesson.

**Vectorise; never loop over pixels.** A $2048 \times 2048$ RGB image is 12.6 million numbers. A Python `for` loop over them costs tens of seconds; the same operation as a numpy expression costs milliseconds, because the loop runs in C over a contiguous buffer. Rule: if an index variable walks over pixels, you have already lost.

```python
mask = (np.abs(gray.astype(np.int16) - bg) > 20)     # whole image, one expression
```

**Read grayscale when colour is unused.** Task 6 only needs to separate strokes from background; the statement says "Background is distinguishable from figure borders", not that colour matters, and "you only need to count the figures — not classify them by type". `cv2.imread(p, cv2.IMREAD_GRAYSCALE)` loads one channel instead of three: a third of the bytes, a third of the decode work, and every downstream operation three times cheaper.

**`dtype` is a memory decision.** [`Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md) wants "exactly 20,000 rows" of D features, D recommended 256 or 512:

- $20000 \times 512 \times 8$ bytes (float64) $= 82$ MB
- $20000 \times 512 \times 4$ bytes (float32) $= 41$ MB

Float32 halves it, and cosine similarity does not care. The 20,000 images themselves, at even $224 \times 224 \times 3$ uint8, are 3 GB if you hold them all — so you do not: you hold embeddings, not pixels. Also note uint8 pixels become float64 the instant you divide by 255 without care; write `img.astype(np.float32) / 255`.

**Batch, then free.** The host baseline in [`host-author-s-baseline.ipynb`](../faio-2025/day1/host-author-s-baseline.ipynb) is explicit about this: it defines `memory_usage_gb()` on `psutil.Process(os.getpid()).memory_info().rss`, prints RAM after loading each JSON file, converts each flattened dict to a DataFrame and then `del players, teams, league; gc.collect()`. Inside the LLM loop it does `del inputs, generation; torch.cuda.empty_cache(); gc.collect()` every iteration. That is the pattern: process a batch, write the result, release the batch. The reward is concrete — its prefix filter takes the players corpus "from 35 million rows to 3.8 million", which is a memory fix as much as an accuracy one.

**Avoid `DataFrame.apply` on hot paths.** `df.apply(f, axis=1)` is a Python loop with DataFrame overhead per row. Use vectorised string methods (`df.paths.str.split(".").str[0]`) or operate on the underlying numpy array. The baseline uses `.str.endswith` and `.str.startswith` for its path filters precisely because those run in C; it also uses `.apply(lambda x: x.split(".")[0])` in one spot, which is the slower spelling of the same thing.

**Measure, then optimise.** `%%time` on a cell, `tqdm` around a loop so you can see the rate and extrapolate, `time.perf_counter()` around the one function you suspect. Optimising an untimed pipeline means optimising the wrong line.

**Never grow a DataFrame in a loop.** `bm25_df = pd.concat([bm25_df, temp_df])` inside a 220-iteration loop — as the baseline does — copies everything each time, making it quadratic. Append to a list and `pd.concat` once at the end.

## [problem-first]

Open [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md) and plan the run, not the algorithm.

1. $8000 \times 2048^2$ → decode time alone is significant. Load grayscale; never load all images at once.
2. "each figure is a single connected component when considering edge pixels" → the work per image is one threshold plus one `findContours`. Both are C-level and fast, so your runtime is dominated by *file reading*, not by logic. That means the optimisation target is I/O, not cleverness.
3. Output is "one integer in each row" of `submit.csv` → results are tiny. Append each count to a list and flush the CSV every few hundred images, so a crash at image 7000 does not cost the run.
4. Nothing in the statement limits tooling — it is the only task in the round that states the opposite, "You may use any programming language and libraries (OpenCV, scikit-image, PIL, etc.)". So use OpenCV's compiled primitives without hesitation.

Then [`Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md): 20,000 images through a pretrained encoder. Here per-image cost is GPU forward-pass, so the lever is **batch size** and **float precision**, not algorithm choice. Encode in batches of 64–256, write embeddings into a preallocated `np.empty((20000, D), np.float32)` rather than appending to a list of arrays, and L2-normalise once at the end.

## [code-first]

```python
import time, gc, os, numpy as np, pandas as pd, cv2, psutil
from tqdm.auto import tqdm

mem_gb = lambda: psutil.Process(os.getpid()).memory_info().rss / 1024**3

def count_figures(path):
    # grayscale: 1 channel instead of 3 -> a third of the decode and the bytes
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    bg = np.bincount(g.ravel()).argmax()                  # dominant value = background
    # vectorised over 4.2M pixels; int16 avoids uint8 wraparound without going float64
    mask = (np.abs(g.astype(np.int16) - int(bg)) > 20).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in cnts if cv2.contourArea(c) >= 20)

# 1. MEASURE FIRST on 10 items, then extrapolate to the full job.
files = sorted(os.listdir("test_images"))
t = time.perf_counter()
for p in files[:10]:
    count_figures(os.path.join("test_images", p))
per = (time.perf_counter() - t) / 10
print(f"{per:.3f} s/image -> {per*len(files)/60:.1f} min for {len(files)} images")
# 0.4 s/image x 8000 = 53 min. If the estimate is unacceptable, fix it NOW.

# 2. Run in batches, flush partial results, release memory. tqdm shows the true rate.
rows, FLUSH = [], 500
for i, p in enumerate(tqdm(files)):
    rows.append((i, count_figures(os.path.join("test_images", p))))
    if (i + 1) % FLUSH == 0:                      # crash-proof: partial file on disk
        pd.DataFrame(rows, columns=["id", "count"]).to_csv("submit.csv", index=False)
        gc.collect()
        tqdm.write(f"{i+1} done, RAM {mem_gb():.2f} GB")
pd.DataFrame(rows, columns=["id", "count"]).to_csv("submit.csv", index=False)

# 3. Embeddings: preallocate float32, never append-and-stack (which doubles peak RAM)
D, N, B = 512, 20000, 128
emb = np.empty((N, D), np.float32)
for s in range(0, N, B):
    emb[s:s+B] = model.encode(names[s:s+B], batch_size=B).astype(np.float32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)      # L2 once, at the end

# 4. Hot paths: vectorised string ops, not apply
df["prefix"] = df.paths.str.split(".", n=1).str[0]      # C loop
# df["prefix"] = df.paths.apply(lambda x: x.split(".")[0])   # Python loop, slower
```

## [drill]

1. One image takes 0.4 s. How long for the 8000 test images of task 6, and what fraction of a 4-hour round is that?
2. You have a 0.9 s/image solution at 14:00. Name two changes, in order of expected win.
3. Why does `cv2.IMREAD_GRAYSCALE` speed up more than just the thresholding step?
4. A $20{,}000 \times 512$ embedding matrix: float64 versus float32 in MB, and does cosine similarity care?
5. What is wrong with `out = pd.concat([out, row_df])` inside a loop over 8000 images?
6. `img.astype(np.int16)` before subtracting the background — why not leave it `uint8`?
7. You must estimate total runtime but the first image is slower than the rest. How do you measure honestly?

<details><summary>Answers</summary>

1. $0.4 \times 8000 = 3200$ s $= 53$ min, about 22% of the 240-minute round — a fifth of everything you have.
2. First: read grayscale instead of RGB and drop any per-pixel Python loop (usually the bulk of the time). Second: downscale the image before thresholding if the strokes survive it, since cost scales with pixel count. Only then consider parallel processes.
3. Decoding is a third of the work, the array is a third of the bytes, so every later operation touches a third of the memory — cache behaviour improves along with raw byte count.
4. $82$ MB float64 versus $41$ MB float32. Cosine similarity is unaffected at this precision; float32 is the correct default.
5. It copies the whole accumulated frame every iteration, so cost is quadratic in the number of images. Collect tuples in a list, build the DataFrame once.
6. `uint8` arithmetic wraps around: `5 - 250` becomes a large positive number, so the mask is wrong. `int16` is wide enough and half the size of float64.
7. Time 10–20 images after a warm-up item and use the median per-image time, or read the rate `tqdm` prints once the loop is a few hundred items in.

</details>

**Rep:** time `count_figures` on 10 images of any size you have locally, extrapolate to 8000, and state whether that fits the 12:00–16:00 plan in [`day05_t32_craft_time_budget_for_a_4_hour_round.md`](./day05_t32_craft_time_budget_for_a_4_hour_round.md).

## Traps & 60-second recall

- Time 10 items, multiply by the real count, decide. Always before launching.
- $0.4$ s $\times\ 8000 = 53$ min. Per-image budget for task 6 is ~0.1–0.2 s.
- No Python loop over pixels. One numpy expression over the whole array.
- Grayscale when colour carries nothing: a third of the bytes and a third of the decode.
- float32 for embeddings and features; float64 doubles memory for no gain.
- `int16` before differencing `uint8` pixels, or subtraction wraps around.
- Batch, flush partial output to disk, `gc.collect()`. A crash at image 7000 must not cost the run.
- Preallocate `np.empty((N, D))`; never append-and-stack at the end.
- No `concat` inside a loop, no `DataFrame.apply` on a hot path — use `.str` methods.
- `%%time`, `tqdm`, `perf_counter`: measure the line before you rewrite it.
