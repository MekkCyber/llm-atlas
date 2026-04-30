# Evaluation

*How model quality is measured — benchmarks, pairwise preference, reward model evaluation, and the feedback loops between eval and training decisions.*

---

## What This Is

Evaluation is the only signal you have during training that says "this is working." Its quality is upstream of every decision you make. This folder covers the benchmark landscape, how reward models are themselves evaluated, and the pitfalls (contamination, Goodharting, reward hacking) that make the whole enterprise hard.

---

## What Belongs Here

- **Capability benchmarks** — MMLU, GSM8K, HumanEval, MATH, GPQA.
- **Holistic evaluation** — HELM, BIG-Bench, multi-axis reporting.
- **Pairwise preference** — Chatbot Arena, pairwise win rates, Elo.
- **Reward model evaluation** — how you evaluate the evaluator.
- **Long-context & agent evals** — needle-in-a-haystack, SWE-bench, WebArena.
- **Contamination** — detection, mitigation, reporting.
- **Goodharting** — when optimizing a metric stops tracking the thing you care about.

## Concept Pages (depth)

### Math reasoning
- [AIME](aime.md) — the live olympiad qualifier (30 problems / year, integer answers, exact-match grading). Now the canonical "hard math" benchmark for reasoning models.
- [MATH-500](math500.md) — 500-problem subset of Hendrycks MATH, introduced by *Let's Verify Step by Step*. Saturated at the frontier (~95–98%); still a standard sanity check.

### Knowledge / instruction-following
- [MMLU](mmlu.md) — 57-subject, 4-way multiple choice. Saturated at the frontier; historical anchor.
- [IFEval](ifeval.md) — verifiable structural instruction-following ("write in > 400 words", "output JSON"). Programmatic grading.

### Code
- [HumanEval](humaneval.md) — 164-problem Python benchmark from the Codex paper, with the unbiased Pass@k estimator. Saturated; still a quick sanity check.
- [LiveCodeBench](livecodebench.md) — continuously-updated, contamination-resistant competitive-programming benchmark. Four scenarios (generation, repair, execution, test-prediction). Not saturated.
- [Codeforces as a benchmark](codeforces-benchmark.md) — how papers operationalize the competitive-programming platform as an LLM eval (Elo rating, percentile).

## Reading Order

1. Capability / knowledge — [MMLU](mmlu.md), [IFEval](ifeval.md).
2. Math reasoning — [MATH-500](math500.md), [AIME](aime.md).
3. Code reasoning — [HumanEval](humaneval.md), [LiveCodeBench](livecodebench.md), [Codeforces](codeforces-benchmark.md).
4. Pairwise preference (Chatbot Arena) — *no depth file yet*.
5. Reward model evaluation — *no depth file yet*.
6. Agent & long-context evals — *no depth file yet*.
7. Contamination & Goodharting — *no depth file yet*.

---

## Related

- [post-training/](../post-training/) — reward model quality drives RL.
- [safety/](../safety/) — safety evals are a distinct axis of evaluation.
- [data/](../data/) — contamination originates in pretraining data.
- [case-studies/](../case-studies/) — DeepSeek-R1 and Kimi k1.5 are evaluated against most of the benchmarks listed here.
