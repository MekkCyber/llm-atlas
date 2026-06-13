# Generative Verifier

*Depth — a model trained to read a candidate proof / solution and emit a verification verdict plus critique, used as a high-precision reward signal at training time and as a search filter at test time.*

**TL;DR:** A generative verifier is an LLM trained to **read a candidate solution and decide whether it's correct**, optionally explaining why. Unlike a scalar outcome-reward model (ORM) that only emits a score, a generative verifier produces structured reasoning — useful for both reward shaping and downstream critique-conditioned repair. The MaxProof recipe (MiniMax-M3) trains generation, verification, and critique-conditioned repair *jointly* into one model with an explicit **defense-in-depth, low-false-positive-rate** verifier engineering target, then uses the same model as generator + verifier + refiner at test time to reach 35/42 IMO 2025 and 36/42 USAMO 2026 — above the human gold-medal threshold.

**Prereqs:** [orm](orm.md), [prm](prm.md), [_rewards](../_rewards.md)
**Related:** [long-cot-rl](long-cot-rl.md), [population-test-time-scaling](population-test-time-scaling.md), [../cot-reward-model](../cot-reward-model.md), [../grpo](../grpo.md)

---

## What it is

A model trained to take a problem $x$ and a candidate solution $y$ and emit:

- A **verdict** (correct / incorrect / unsure), and
- A **critique** — a textual explanation that can be fed into a refinement step.

Two distinctions from related concepts:

- **vs. ORM:** an outcome reward model emits a scalar; a generative verifier emits text and a verdict. The textual output is directly useful for critique-conditioned repair.
- **vs. PRM:** a process reward model scores intermediate steps; a generative verifier judges the full solution.

The signature engineering choice is the **target false-positive rate**: how often the verifier blesses an incorrect solution. In a search loop, a false positive *terminates* the search at a wrong answer — far more damaging than a false negative, which just costs another candidate. The MaxProof verifier is "defense in depth": multiple checks layered to drive the FPR down, accepting more false negatives in exchange.

## How it works

Training (MaxProof recipe):

1. **Joint capability training**: a single base model is trained for three skills — proof generation, proof verification, and critique-conditioned repair — using a shared backbone and skill-specific data mixtures. The end result is one released model that wears all three hats.
2. **Defense-in-depth verifier**: the verification training data is engineered so the model only emits "correct" when multiple layered checks agree (e.g. statement-level checks, step-level checks, holistic consistency). This drives down the false-positive rate at the cost of more false negatives.
3. **Critique-conditioned repair**: the repair head conditions on `(problem, broken solution, critique)` and emits a fixed solution. The critique connects verification to repair — the verifier doesn't just say "no", it tells the refiner what to fix.

Test-time use:

- Generate a population of candidate solutions (see [population-test-time-scaling](population-test-time-scaling.md)).
- Filter each through the generative verifier.
- For "incorrect" candidates, run critique-conditioned repair to recover.
- Select the final answer via tournament among the survivors.

The same model is generator, verifier, refiner, and ranker — different prompts, same weights.

## Why it matters

- Beats the **false-positive trap** that has historically hobbled reward modeling for open-ended reasoning. If the reward is wrong on a substantive fraction of the rollouts, RL training poisons the policy and TTS picks the wrong winner.
- Makes **critique** a usable primitive: structured text from the verifier feeds straight into a repair pass, closing the loop without needing a separate critic model.
- Unifying the roles into one model is a deployment win — no need to keep three separate checkpoints in memory at inference.

## Gotchas & tricks

- **FPR-FNR is the real tradeoff.** Engineering for low FPR usually means accepting higher FNR (the verifier rejects some correct solutions). In a search-then-verify pipeline, you tolerate the FNR by generating more candidates, but you can't tolerate the FPR.
- **Verifier confidence drift.** As the policy improves during RL, the distribution of candidate solutions shifts; the verifier needs to keep up or it becomes the bottleneck.
- **Critique quality matters more than verdict accuracy** for the repair head. A verifier that correctly says "wrong" but with a vague critique gives the refiner nothing to act on.
- **Same model, different prompts** is convenient but the prompt for "verify" and "generate" can interfere — careful prompt engineering and skill-conditioning tokens help.

## Sources

- Paper: MaxProof / MiniMax-M3 — Zhang et al. (2026) — [arXiv:2606.13473](https://arxiv.org/abs/2606.13473)
- Background: generative verifiers as a class (Cobbe et al., 2021; Lightman et al., 2024 PRM; Zhang et al., 2025 GenRM-style work).
