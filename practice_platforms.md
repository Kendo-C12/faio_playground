# practice_platforms.md — where to practise with a real submit button

FAIO's qualification runs on **Yandex.Contest** (contest 81908 in 2025). What you want before 20 September is somewhere that gives you the same loop: read a short task, produce an answer or a CSV, submit, get scored immediately. Kaggle's main competitions are the wrong size — they run for months and expect weeks of work per task.

Ranked by how closely each matches the FAIO qualification.

## 1. Тренировки по ML — Yandex ML Training · best match

**https://yandex.ru/yaintern/training** · contests on **contest.yandex.ru**

Free ML intensives run by Yandex, graded on the **same platform FAIO uses**. Tasks are published on a schedule but stay open, so you can work at your own pace, including after the walkthrough is posted. No selection, anyone can register.

Contest IDs visible in search results:

| Contest | ID | Link |
|---|---|---|
| Тренировки по ML 3.0 — ДЗ 1 | 75228 | `contest.yandex.ru/contest/75228/enter/` |
| Тренировки по ML 3.0 — ДЗ 2 | 75229 | `contest.yandex.ru/contest/75229/enter/` |
| Тренировки по ML 2.0 — ДЗ 1 | 67635 | `contest.yandex.ru/contest/67635/enter/` |

**Why this is the closest thing available:** identical submission mechanics, identical verdict display, identical standings table. Getting used to the interface is worth real points on exam day — you will not be learning where the submit button is while the clock runs.

**The catch:** the material is in Russian. Browser translation handles it, and the 2025 FAIO round shipped its own statements in Kazakh and Russian too, so this is the same obstacle you would face anyway. Formulas, code and metric names are language-neutral.

## 2. Kaggle Playground Series · closest by task size

**https://www.kaggle.com/competitions/playground-series**

A new tabular competition every month, deliberately built for practice rather than prize money. Datasets are small and synthetic, the metric is stated plainly, and a reasonable submission takes an hour or two — not the weeks a featured Kaggle competition demands.

This is the right shape for FAIO tasks 4 and 5: read the metric, build features, fit a gradient boosting model, write `submission.csv`, submit, see the score. Public leaderboard updates immediately.

## 3. Kaggle Getting Started · always open, instant scoring

These four never close and never expire, so they work as a dry run at any hour:

| Competition | Matches FAIO task |
|---|---|
| **Titanic** | tabular binary classification — task 5's shape |
| **Spaceship Titanic** | same, slightly larger |
| **House Prices** | regression, good for metric practice |
| **Digit Recognizer** | image classification — task 6's neighbourhood |

Start with Titanic specifically to rehearse the submission format discipline in [`lessons/day01_t31_craft_submission_discipline.md`](./lessons/day01_t31_craft_submission_discipline.md): exact filename, exact columns, exact row count.

## 4. Yandex Cup ML track · too hard, listed so you can skip it knowingly

**https://yandex.com/cup/ml**

Yandex's own championship. The 2025 ML qualification ran 22 days with three tasks: autonomous-driving camera image generation, STEM diagram question answering with vision-language models, and LLM hallucination robustness. Two submissions per day, 1 GPU-hour limit.

This is professional-level work and nothing like a 4-hour school olympiad. Worth knowing it exists; not worth your remaining days.

There is an **archive of past tasks** at https://yandex.com/cup/algorithm/archive if you want to see the format later.

## 5. Other platforms with short contests

- **MachineHack** — hackathons that often run days rather than months, CSV submission.
- **Zindi** — community platform focused on African datasets; runs short hackathons alongside longer competitions, beginner-friendly.
- **CodaBench** — open-source successor to CodaLab, noted as one of the most used platforms after Kaggle; hosts many small academic benchmarks.
- **Analytics Vidhya JobAThon** — short weekend hackathons, tabular, CSV submission.

## Suggested use of your remaining days

You have four study days left and each already carries lessons. Practice platforms are for *rehearsal*, not learning — fit them around the schedule rather than instead of it.

| When | What | Why |
|---|---|---|
| After day 1 (submission discipline) | One Titanic submission, start to finish | Rehearse the format checklist against a real grader |
| After day 2 (gradient boosting) | One Playground Series entry with a boosting baseline | The exact motion tasks 4 and 5 need |
| Any evening | Register for Тренировки по ML and open one task | Get familiar with the Yandex.Contest interface before exam day |
| Day 5 | Timed mock on the 2025 statements, scored with [`grader.md`](./grader.md) | Closest possible simulation of the real round |

One warning drawn from [`lessons/day05_t32_craft_time_budget_for_a_4_hour_round.md`](./lessons/day05_t32_craft_time_budget_for_a_4_hour_round.md): these platforms reward spending days on one leaderboard, which is the opposite of what a 4-hour round with six tasks rewards. Practise *finishing* — one valid submission per task — rather than perfecting a single score.

## What I could not verify

`contest.yandex.ru`, `yandex.com` and `kaggle.com` are all blocked by this environment's network proxy, so none of the links above were opened directly. Contest IDs, dates and task descriptions come from search results, and the ML Training contest IDs may point to past seasons by the time you read this. Open https://yandex.ru/yaintern/training to see the current season.
