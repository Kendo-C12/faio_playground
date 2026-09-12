# FAIO 2025 qualification — solutions

One markdown file per problem of the FAIO 2025 qualification round (Yandex.Contest 81908, problems 1–6). Each file covers three things, then the solution:

1. **What the problem asks** — solve a statistics problem, build a classifier, and so on
2. **Knowledge required** — with knowledge the statement itself supplies marked as *given*
3. **Languages and libraries** — what is allowed, what is forbidden, or *not stated* where no rule could be found

| File | Problem | Shape |
|---|---|---|
| [`task1-stats-101.md`](./task1-stats-101.md) | Statistics 101? | math, answer only |
| [`task2-probability-theory-101.md`](./task2-probability-theory-101.md) | Probability Theory 101 | math, answer only |
| [`task3-medicine.md`](./task3-medicine.md) | Medicine | math, answer only |
| [`task4-who-speaks-what.md`](./task4-who-speaks-what.md) | Who Speaks What? | ML, NLP language ID |
| [`task5-ai-yoga-instructor.md`](./task5-ai-yoga-instructor.md) | AI Yoga Instructor | ML, IMU time series |
| [`task6-simple-objects.md`](./task6-simple-objects.md) | Simple Objects | computer vision, counting |

## What the round gives you

Theory sections appear only in tasks 1–3, the pen-and-paper ones: mean/mode/median, classical probability, and conditional probability with Bayes' theorem. Tasks 4 and 5 supply no theory at all. Task 6 supplies method hints (edge detection, contour finding, shape properties) but no ML theory.

Tooling rules are stated **once** in the whole round, in task 6: "You may use any programming language and libraries (OpenCV, scikit-image, PIL, etc.)". Nothing else in the archive restricts languages or libraries, and faio.kz publishes no reachable rules page, so the other tasks are marked *not stated*. For tasks 4–6 only the prediction CSV is graded — "source code is not required" — so the judge never runs your program.

Statements live in [`faio-2025/qualification/`](../../faio-2025/qualification). Category and format breakdown for every FAIO problem: [`docs/faio-2025-problems.md`](../../docs/faio-2025-problems.md).
