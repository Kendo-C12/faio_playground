# T23 · Pretrained transformers: tokenisation, features, fine-tuning — Day 3

**Anchor task(s):**
- [`faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)
- [`faio-2025/day1/host-author-s-baseline.ipynb`](../faio-2025/day1/host-author-s-baseline.ipynb)

Rating 1 · **exam-probability rank 1** · ~55 min

A pretrained transformer is a file of weights that already understands language. You are renting that understanding, not building it.

## [concept-first]

**Tokenisation.** A transformer cannot read characters. A tokeniser splits text into sub-word pieces from a fixed vocabulary and maps each to an integer. `"Сәлеметсіз"` might become `["▁Сәле", "мет", "сіз"]` → `[14221, 903, 1177]`. Two consequences: an unknown word never becomes `<UNK>`, it decomposes; and **token count $\ne$ word count**, so a 512-token limit is not 512 words. For Kazakh, a multilingual tokeniser splits far more aggressively than for English — check `len(tok(text).input_ids)` before assuming anything fits.

**What a pretrained encoder gives you.** Feed token ids into a model like XLM-RoBERTa and you get, per token, a vector that encodes that token *in its context*. Pool those (mean, or the `[CLS]` position) into one vector per document and you have a dense 768-dimensional embedding where semantically similar sentences are close. That is the product: a feature extractor you did not have to train.

**Two ways to use it.**

*Feature extraction (frozen).* Run the encoder once in inference mode, cache the embeddings, fit a small classifier — logistic regression, SVM, LightGBM — on top. Cheap, no GPU training, no hyperparameters that can blow up, and the embeddings are reusable across experiments. Do this first, every time.

*Fine-tuning.* Attach a classification head and backpropagate through the encoder with a small learning rate (2e-5 to 5e-5), 2–3 epochs. Usually a few points better than frozen features. Costs GPU time, needs a real validation split, and can collapse to one class if the learning rate is wrong.

**Multilingual models.** Standard BERT is English. `xlm-roberta-base` (~279M params) is trained on 100 languages including Kazakh and Russian, sharing one SentencePiece vocabulary — which is precisely the task 4 setting. `xlm-roberta-large` is ~560M and better but three times slower. `Kaz-RoBERTa-Conversational` is Kazakh-specific and conversational, matching the support-chat domain. All three are named in the task 4 notebook's closing cell as routes to a higher score.

**Quantisation.** Weights are normally fp32 (4 bytes each). Cast to bf16/fp16 and VRAM halves with almost no quality loss. `BitsAndBytesConfig(load_in_4bit=True)` goes further — roughly an eighth the memory, some quality cost — and is how a 4B+ model fits alongside your data on one card.

Worked example. IOAI 2026's individual contest runs on **one ~16 GB GPU, no internet, 5 GB storage**. `xlm-roberta-base` in bf16 is about 0.6 GB of weights: fine-tuning it fits comfortably. A 7B chat model in fp16 is ~14 GB of weights alone and will not fine-tune there — 4-bit inference only. That arithmetic, done in the first ten minutes, decides your whole approach.

## [problem-first]

Open the closing markdown cell of [`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb). Having shipped a TF-IDF plus logistic regression solution, the author suggests, to go further: fine-tuned BERT variants (XLM-RoBERTa base/large, Kaz-RoBERTa-Conversational), and SVM / LogReg / CatBoost / LightGBM over TF-IDF **or FastText** features.

Read that ordering as the jury's own honest cost/benefit. TF-IDF is the baseline you must have; transformers are the upgrade you attempt only if time remains. In a 4-hour round with six tasks, the TF-IDF path on task 4 takes about 15 minutes end to end and lands in the high 0.9s macro-F1. Fine-tuning XLM-R takes 40–90 minutes including debugging, for maybe one or two points. **On task 4 specifically, TF-IDF is the correct call.** The transformer is worth it when characters are not enough — long documents, semantics, sentiment, entailment.

Now open [`host-author-s-baseline.ipynb`](../faio-2025/day1/host-author-s-baseline.ipynb) to see the jury actually do it. Three things to notice:

1. It imports `AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig` from `transformers` and loads with `device_map="cuda:1", torch_dtype=torch.bfloat16`. The `BitsAndBytesConfig(load_in_4bit=True)` line and its `quantization_config=` argument are both present but **commented out** — bf16 was enough on their hardware. The 4-bit path is written down for you to uncomment when it is not.
2. `model_id` is a **local path**, `/kaggle/input/gemma-3/transformers/gemma-3-4b-it/1`. Nothing is downloaded at runtime. Same for the `SentenceTransformer('/kaggle/input/bge-large-en-v1.5/...')` used to rerank BM25 candidates. This is what "no internet" looks like in practice.
3. The transformer is the *last* stage. `bm25s` + `Stemmer` retrieves, `sentence_transformers` reranks, and only then does the generative model pick an answer. Cheap retrieval first, expensive model on a short list.

## [code-first]

```python
# Frozen feature extraction — the version that fits in a timed round.
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

MODEL = "xlm-roberta-base"          # offline: point this at a local directory
tok = AutoTokenizer.from_pretrained(MODEL)
enc = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval().cuda()

@torch.no_grad()                     # no gradients: halves memory, doubles speed
def embed(texts, bs=64, maxlen=128):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                max_length=maxlen, return_tensors="pt").to("cuda")
        h = enc(**b).last_hidden_state                 # (batch, tokens, 768)
        m = b["attention_mask"].unsqueeze(-1)          # mask out padding...
        v = (h * m).sum(1) / m.sum(1)                  # ...then mean-pool
        out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
    return np.vstack(out)

train = pd.read_csv("train.csv")
X = embed(train.text.values)
print(cross_val_score(LogisticRegression(max_iter=1000), X, train.label,
                      cv=5, scoring="f1_macro").mean())

# Check the token budget before trusting max_length=128.
print(pd.Series([len(tok(t).input_ids) for t in train.text[:2000]]).describe())
```

To fine-tune instead, swap `AutoModel` for `AutoModelForSequenceClassification(num_labels=3)`, use `Trainer` with `learning_rate=2e-5, num_train_epochs=3, per_device_train_batch_size=16, fp16=True`, and keep a real held-out split. If VRAM is tight, add `BitsAndBytesConfig(load_in_4bit=True)` for inference, or cut `max_length` and batch size for training.

## [drill]

1. Why is token count not word count, and where does that bite on a 512-token model?
2. Frozen features vs fine-tuning: name one advantage of each in a 4-hour round.
3. Which of `bert-base-uncased`, `xlm-roberta-base`, `Kaz-RoBERTa-Conversational` can handle `ru`/`kaz`/`eng` in one model?
4. Roughly how much VRAM do the weights of a 7B model take in fp16, and in 4-bit?
5. In the day 1 baseline, is the generative model doing the retrieval? What does, and why that order?
6. Your fine-tune predicts one class for everything. Name the two likeliest causes.

<details><summary>Answers</summary>

1. Sub-word tokenisers split rare words into several pieces, and split non-English text more aggressively, so 512 tokens can be only 150–250 Kazakh words. Anything past the limit is silently truncated.
2. Frozen: no training, embeddings cached once and reused, nothing to tune — it cannot fail expensively. Fine-tuning: a few points more accuracy, because the encoder adapts to your labels.
3. `xlm-roberta-base` — trained on 100 languages with one shared vocabulary. `bert-base-uncased` is English; Kaz-RoBERTa is Kazakh-focused.
4. About 14 GB in fp16 ($2 \text{ bytes} \times 7 \times 10^{9}$), roughly 3.5–4 GB in 4-bit — the difference between not fitting and fitting on a 16 GB card.
5. No. `bm25s` with `Stemmer` retrieves candidates and a `SentenceTransformer` reranks them; the model only picks from a short list. Cheap retrieval narrows the field so the expensive model runs on few inputs.
6. Learning rate too high (the head collapses to the majority class), or a label-encoding mismatch between `num_labels` and the actual label ids.

</details>

**Rep:** embed task 4's `train.csv` with frozen `xlm-roberta-base`, cross-validate logistic regression on the embeddings with `f1_macro`, and compare against the char-TF-IDF number from T20. Write both down together with the wall-clock cost of each.

## Traps & 60-second recall

- Do the VRAM arithmetic first: $\text{params} \times \text{bytes-per-param}$. 16 GB is the IOAI 2026 budget.
- Assume no internet. Load from local paths, and pre-download weights while you still can.
- `torch.no_grad()` and `.eval()` for inference, every time.
- Mean-pool with the attention mask, not a plain `.mean(1)` — padding otherwise dilutes every vector.
- `fit` the tokeniser nothing: it is fixed. But check the token-length distribution before setting `max_length`.
- Frozen features first, fine-tune only with time to spare.
- TF-IDF beats a transformer on wall-clock-per-point for task 4. Choose by the clock, not by prestige.
