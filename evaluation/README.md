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

## Reading Order

1. Capability benchmarks (MMLU, etc.)
2. Pairwise preference (Chatbot Arena)
3. Reward model evaluation
4. Agent & long-context evals
5. Contamination & Goodharting

---

## Related

- [post-training/](../post-training/) — reward model quality drives RL.
- [safety/](../safety/) — safety evals are a distinct axis of evaluation.
- [data/](../data/) — contamination originates in pretraining data.
