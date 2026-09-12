# T21 · Text normalisation and augmentation — Day 3

**Anchor task(s):**
- [`faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb)
- [`faio-2025/qualification/task4_Who_Speaks_What.md`](../faio-2025/qualification/task4_Who_Speaks_What.md)

Rating 0 · **exam-probability rank 1** · ~50 min

This is the actual trick that wins task 4. The model is a plain logistic regression; the score comes from what was fed to it.
## [concept-first]

**Normalisation** removes variation you have decided is meaningless. **Augmentation** adds variation you expect to meet at test time. They pull in opposite directions and you need both.

*Normalisation*, in rising order of aggression: **case** (`text.lower()`, free unless capitalisation is the signal); **whitespace** (collapse runs, strip ends); **punctuation and digits** (for language ID, keep them — `?` and `!` differ in frequency across languages); **unicode** (`unicodedata.normalize("NFKC", s)`, because identical-looking characters can have different code points and NFKC folds them, so your vocabulary does not split one letter into two features).

*Augmentation* makes new `(text, label)` pairs from existing ones. The rule that decides whether one is legal:

- **Label-preserving** — the transform cannot change the true answer. Stripping Kazakh diacritics from a Kazakh sentence leaves a Kazakh sentence. Safe.
- **Label-destroying** — the transform changes the true answer, or could. Machine-translating a Russian message to English and keeping the label `ru` is a lie to the model. Unsafe.

The general principle, worth memorising: **augment toward the distribution shift you expect at test time.** Not random noise — the specific failure you predict.

Worked example, the shipped notebook's reasoning. Kazakh and Russian share most of the Cyrillic alphabet; Kazakh adds `ә і ң ғ ү ұ ө қ`. A lazy model learns "contains `ә` → `kaz`" and nothing else, then fails on two real test cases: (1) a Kazakh message typed **without** the special letters, and (2) a Kazakh or Russian message typed in **Latin script** — translit, the normal way people type without a Cyrillic keyboard. Neither appears in `train.csv` in quantity, so the notebook manufactures them.

## [problem-first]

Open [`task4_solution_Who_Speaks_What.ipynb`](../faio-2025/qualification/task4_solution_Who_Speaks_What.ipynb) and read cells 1–3.

**Cell 1 — the two transforms.** `KAZAKH_TO_RUS_REPLACEMENTS` is a 16-entry dict mapping each Kazakh-specific Cyrillic letter, both cases, to its nearest Russian one: `Ә→А`, `ә→а`, `І→И`, `і→и`, `Ң→Н`, `ң→н`, `Ғ→Г`, `ғ→г`, `Ү→У`, `ү→у`, `Ұ→У`, `ұ→у`, `Ө→О`, `ө→о`, `Қ→К`, `қ→к`. `kazakh_to_russian(text)` applies it character by character. `translit_text(text)` is `translit(text, "ru", reversed=True)` from the `transliterate` package — Cyrillic out, Latin in.

**Cell 3 — the augmentation recipe**, split by class:

```
kaz:  50% -> kazakh_to_russian       (of those, 50% -> also transliterate)
      50% -> untouched
ru:   50% -> transliterate
      50% -> untouched
eng:  untouched
```

Every split is `sample(frac=0.5, random_state=42)` with the complement taken by `drop(index)`, then the pieces are `pd.concat`-ed back into the training set. Read the code carefully: the transformed rows **replace** the originals inside their own half, so the final set is the same size, with `kaz` ending up 25% de-diacriticised, 25% de-diacriticised-and-Latinised, 50% original.

Why each piece earns its place. `kazakh_to_russian` on `kaz` forces the model to find Kazakh in the *word shapes and endings* (`-дар`, `-мен`, `-ға`) rather than in one diacritic — this is the piece that matters most. Transliterating both `kaz` and `ru` teaches the model that Latin script ≠ `eng`; without it, every Latin-script message gets labelled `eng`. And `eng` is left alone because no shift is expected for it — you do not augment a class whose test distribution already matches training.

The closing markdown cell confirms the target: the test set contains **code-switched** texts, English grammar carrying Kazakh or Russian words and vice versa. The augmentation is a cheap synthetic approximation of that; manually labelling the real code-switched test rows and adding them to training improves macro-F1 further.

## [code-first]

```python
import pandas as pd
from transliterate import translit          # pip install transliterate

KAZAKH_TO_RUS_REPLACEMENTS = {
    "Ә": "А", "ә": "а", "І": "И", "і": "и", "Ң": "Н", "ң": "н",
    "Ғ": "Г", "ғ": "г", "Ү": "У", "ү": "у", "Ұ": "У", "ұ": "у",
    "Ө": "О", "ө": "о", "Қ": "К", "қ": "к",
}

def kazakh_to_russian(text: str) -> str:
    # Strip the giveaway letters. Label stays 'kaz' — label-PRESERVING.
    return "".join(KAZAKH_TO_RUS_REPLACEMENTS.get(ch, ch) for ch in text)

def translit_text(text: str) -> str:
    # Cyrillic -> Latin. Label stays 'kaz'/'ru' — script is not language.
    return translit(text, "ru", reversed=True)

train_df = pd.read_csv("train.csv")
kaz, eng, ru = (train_df[train_df.label == L] for L in ("kaz", "eng", "ru"))

# kaz: half loses its diacritics; half of THAT half is then Latinised.
kaz_a = kaz.sample(frac=0.5, random_state=42)
kaz_b = kaz.drop(kaz_a.index)
kaz_a["text"] = kaz_a.text.apply(kazakh_to_russian)
kaz_a1 = kaz_a.sample(frac=0.5, random_state=42)
kaz_a2 = kaz_a.drop(kaz_a1.index)
kaz_a1["text"] = kaz_a1.text.apply(translit_text)
kaz = pd.concat([kaz_a1, kaz_a2, kaz_b])

# ru: half Latinised, so Latin script stops being a proxy for 'eng'.
ru_a = ru.sample(frac=0.5, random_state=42)
ru_b = ru.drop(ru_a.index)
ru_a["text"] = ru_a.text.apply(translit_text)
ru = pd.concat([ru_a, ru_b])

train = pd.concat([kaz, eng, ru])        # eng untouched: no shift expected
print(train.label.value_counts())        # class balance must not have moved
```

Augment **after** the train/validation split, training half only — otherwise a sentence and its transliteration land on both sides and the validation score is fiction.

## [drill]

1. Why is `kazakh_to_russian` label-preserving while machine translation is not?
2. What does transliterating half the `ru` rows prevent the model from learning?
3. Why is `eng` left untouched?
4. You augment the full dataset and then split 80/20. What is wrong with the validation number?
5. `unicodedata.normalize("NFKC", s)` — what failure does it prevent in a TF-IDF vocabulary?
6. What is the notebook's own suggestion for the code-switched rows, and why is synthetic augmentation only an approximation of them?

<details><summary>Answers</summary>

1. Replacing `ә` with `а` removes evidence but does not make the sentence Russian — the words, grammar and endings are still Kazakh. Translation replaces the language itself, so the old label becomes false.
2. "Latin script → `eng`". Without Latinised `ru`/`kaz` rows, script alone perfectly separates `eng` in training, and the model never looks further.
3. No shift is expected — English in the test set looks like English in training, so augmenting it adds noise and costs fit time.
4. Near-duplicates straddle the split: a validation row is a transform of a training row, so you measure memorisation. Split first, augment the training side only.
5. Two visually identical characters with different code points become two separate features, splitting one letter's evidence across two columns.
6. Manually label them and add them to training. Synthetic augmentation models script and diacritic shift, not real mixed-grammar sentences — those have word-level structure no character transform produces.

</details>

**Rep:** reproduce the cell-3 recipe on task 4's `train.csv`, fit the notebook's word-level TF-IDF plus logistic regression twice — once on raw, once on augmented — and report both macro-F1 values. The gap is the whole lesson.

## Traps & 60-second recall

- Split first, augment the training half only. Always.
- Label-preserving or do not do it. Ask "could this change the true answer?" before every transform.
- Augment toward the shift you expect at test time, not toward random noise; leave classes alone where no shift is expected.
- `frac=0.5` with a fixed `random_state` and `drop(index)` for the complement — reproducible halves, no overlap.
- Normalise (case, NFKC, whitespace) *identically* for train and test; augment only train.
- Check `label.value_counts()` after augmenting: a shifted balance silently changes what macro-F1 means.
