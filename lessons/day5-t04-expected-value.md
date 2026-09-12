# T04 · Expected value: linearity, indicators, metric reasoning — Day 5

**Anchor task(s):**
- [`faio-2025/day2/HearMe_Personalized_Music_Recommender.md`](../faio-2025/day2/HearMe_Personalized_Music_Recommender.md)
- [`faio-2025/day2/Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md)

Rating 2 · **exam-probability rank 2** · ~35 min

Both of these metrics *are* expectations. Once you see that, what to optimise stops being a guess.

## [concept-first]

**Definition.** For a discrete random variable `X` taking value `xᵢ` with probability `pᵢ`:

```
E[X] = Σ xᵢ · pᵢ
```

A weighted average, weights summing to 1. Nothing more.

**Indicators.** Let `1[A]` be 1 when event `A` happens and 0 otherwise. Then

```
E[1[A]] = 1·P(A) + 0·(1−P(A)) = P(A)
```

*The expectation of an indicator is its probability.* This single line converts "what fraction of queries hit?" into "what is the probability one query hits?" and back.

**Linearity.** For any random variables, dependent or not, and any constants:

```
E[a·X + b·Y] = a·E[X] + b·E[Y]      and      E[Σ Xᵢ] = Σ E[Xᵢ]
```

No independence required. This is the workhorse: a metric that sums per-item scores has an expectation equal to the sum of per-item expectations, so you can reason item by item even when your 50 recommendations clearly interact.

**Worked example — the HearMe score.** [`HearMe_Personalized_Music_Recommender.md`](../faio-2025/day2/HearMe_Personalized_Music_Recommender.md) defines, for one user, `Sᵤ = Σ_{j=1..50} fⱼ` where `fⱼ ∈ {0, 0.25, 0.5, 0.75, 1.0}` is the fraction of recommended track `j` the user actually listened to. By linearity,

```
E[Sᵤ] = Σ_{j=1..50} E[fⱼ]
```

So each of the 50 slots contributes independently-in-expectation, and the best possible slate is simply *the 50 tracks with the highest `E[fⱼ]`*. Each `E[fⱼ]` is itself an expectation over the five-point scale:

```
E[f] = 0·P(0) + 0.25·P(.25) + 0.5·P(.5) + 0.75·P(.75) + 1·P(1)
```

Call that the track's **expected listening depth** for this user. A track with `P(1.0) = 0.3` and `P(0) = 0.7` scores `E[f] = 0.30`. A track with `P(0.25) = 0.9, P(0) = 0.1` scores `0.225` — less, despite being listened to nine times out of ten. Probability of engagement is not the objective; depth × probability is.

**Worked example — Hit@3.** [`Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md) scores `Hit@3 = (number of queries with correct match in Top-3) / 1000`. That is the mean of 1000 indicators, so it estimates `P(true match lands in the top 3)` for a random query. Nothing else about the ranking is in the metric: rank 1 and rank 3 are worth the same, rank 4 and rank 19,000 are worth the same.

## [problem-first]

Open [`HearMe_Personalized_Music_Recommender.md`](../faio-2025/day2/HearMe_Personalized_Music_Recommender.md) and read the evaluation section as a formula rather than as prose.

1. "Sum these fractions across the 50 tracks → a score for that user (range 0 to 50)" → per-user score is a **sum of per-item scores**. Linearity applies; optimise slots independently.
2. "Average per user… then normalized score between 0 and 1" → the final number is `E[Sᵤ]/50` over a random user, i.e. the **mean expected listening depth per recommended track**. The statement's own reading confirms the scale: "~0.5 → Users listen to about half of your recommendations."
3. "replays don't count extra" → `f` is capped at 1. A superfan track cannot subsidise 49 bad slots. Expectation per slot is bounded by 1, so there is no jackpot strategy, only 50 independent bets.
4. "no cold-start users" → every test user has history, so every `E[fⱼ]` is estimable from that user's own behaviour. There is no excuse for falling back to a global prior.

Now the payoff question: **why does ranking by expected listening depth beat ranking by popularity?** Popularity ranks by `P(anyone interacts)`, marginalised over all users. The metric pays `E[f | this user]`. A globally popular track the user has already heard to death, or in a genre they skip, has high popularity and low `E[f]` for them — you spend a slot worth ~0.1 where a personalised pick was worth ~0.6. Across 50 slots and a normalised 0–1 scale, that is the whole gap between "0.25, users are starting to engage" and "~0.5, clearly interesting". Popularity is only the right answer when you have no per-user signal, which this task explicitly denies you.

Contrast with Hit@3 in [`Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md): there the per-query contribution is an indicator, so *margin buys you nothing*. Pushing a correct match from cosine rank 3 to rank 1 adds exactly 0. Pushing one from rank 4 to rank 3 adds 1/1000. All your effort belongs at the decision boundary — the queries that are currently just outside the top 3 — not on the ones already comfortably inside.

## [code-first]

```python
import numpy as np, pandas as pd

# --- HearMe: E[f] per (user, item), then take the top 50 per user -------------
# listened fraction, clipped at 1 because "replays don't count extra"
inter = pd.read_csv("interactions.csv").merge(
    pd.read_csv("item_metadata.csv")[["item_id", "track_duration"]], on="item_id")
inter["f"] = (inter.listened_duration / inter.track_duration).clip(0, 1)
# quantise onto the grader's five-point scale: 0, .25, .5, .75, 1
inter["f"] = (inter.f * 4).round() / 4

# E[f] is an average of past f values -> the per-slot expected contribution.
# Shrink toward the item mean so a single lucky play does not outrank real evidence.
GLOBAL = inter.f.mean()
item_mean = inter.groupby("item_id").f.mean()
ui = inter.groupby(["user_id", "item_id"]).f.agg(["mean", "count"])
K = 5                                    # pseudo-counts: strength of the prior
ui["Ef"] = (ui["mean"] * ui["count"] + GLOBAL * K) / (ui["count"] + K)

# Linearity of expectation: E[sum over 50 slots] = sum of the 50 largest E[f].
# So "best slate" = argsort on Ef, no interaction term needed.
top = (ui.reset_index()
         .sort_values(["user_id", "Ef"], ascending=[True, False])
         .groupby("user_id").head(50))
top["rank"] = top.groupby("user_id").cumcount() + 1        # 1..50, unique per user
top.insert(0, "id", range(len(top)))                       # statement's own recipe
top[["id", "user_id", "item_id", "rank"]].to_csv("submission.csv", index=False)

# --- Lost in the Museum: Hit@3 is the mean of indicators ---------------------
def hit_at_3(q, gallery, truth_idx):            # q, gallery already L2-normalised
    sims = gallery @ q                          # normalised -> dot product IS cosine
    top3 = np.argpartition(-sims, 3)[:3]
    return float(truth_idx in top3)             # an indicator; E[1[A]] = P(A)

# mean over queries estimates P(correct match in top 3) -- margin is never scored
score = np.mean([hit_at_3(Q[i], G, t[i]) for i in range(len(Q))])
```

Note what the HearMe code does *not* do: it never models "does this slate as a whole look good". Linearity says it does not have to.

## [drill]

1. `X` is the listened fraction with `P(0)=0.5, P(0.5)=0.3, P(1)=0.2`. Compute `E[X]`.
2. You recommend 50 tracks, each with `E[f] = 0.4`. What is the expected per-user score, and the normalised final score?
3. Track A: `P(f=1) = 0.25`, else 0. Track B: `P(f=0.5) = 0.55`, else 0. Which slot is worth more under the HearMe metric?
4. Why can you optimise the 50 slots independently even though a user's listening time is obviously shared between them?
5. Under Hit@3, you move a correct match from cosine rank 2 to rank 1 for 200 of 1000 queries. What happens to the score?
6. Write Hit@3 as an expectation of an indicator, in symbols.

<details><summary>Answers</summary>

1. `0·0.5 + 0.5·0.3 + 1·0.2 = 0.35`.
2. `E[S] = 50 × 0.4 = 20` out of 50; normalised `20/50 = 0.4`.
3. B: `E[f_B] = 0.5·0.55 = 0.275` versus `E[f_A] = 1·0.25 = 0.25`. Depth alone does not decide it — the product does.
4. Linearity of expectation needs no independence: `E[Σ fⱼ] = Σ E[fⱼ]` regardless of how the `fⱼ` are correlated. The sum is what the grader computes, so maximising the sum of expectations is exactly right.
5. Nothing — it stays identical. Hit@3 only asks whether the match is inside the top 3; ordering within the top 3 is invisible to it.
6. `Hit@3 = E[1[true match ∈ Top3(q)]] = P(true match ∈ Top3(q))`, estimated by the mean over the 1000 queries.

</details>

**Rep:** for the HearMe metric, write down the expected score of a submission that fills all 50 slots with the globally most-played tracks, given `E[f] ≈ 0.12` for an unpersonalised pick, and compare with `E[f] ≈ 0.45` for a personalised one. That gap is the whole task.

## Traps & 60-second recall

- `E[X] = Σ xᵢpᵢ`. `E[1[A]] = P(A)`. `E[ΣXᵢ] = ΣE[Xᵢ]`, dependence allowed.
- A metric that sums per-item scores is optimised slot by slot — rank by the per-item expectation.
- Probability of engagement ≠ expected depth. Multiply, do not substitute.
- Clip at the cap the statement gives you (`replays don't count extra`) before averaging, or your `E[f]` is inflated.
- Indicator metrics (Hit@k, accuracy) pay nothing for margin: work the boundary cases only.
- Shrink a per-user mean computed from 1–2 observations toward a prior; an unshrunk expectation estimated from one play is noise, not signal.
