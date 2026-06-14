# Generative-Verifier RL
*Depth — training a verifier as a generative LLM in parallel with the proof/solution generator, then using it for RL and inference search.*

**TL;DR:** In RLVR, the verifier is usually a fixed rule (does the math answer match? do the tests pass?). For long-form outputs like proofs, code, or multi-step plans, no rule-based verifier exists. **Generative-verifier RL** trains a separate LLM as a verifier that produces an *error-localization explanation* alongside its accept/reject judgment, jointly with the generator. The localization signal is the lever — it lets a third *repair* model patch the incorrect step rather than restart. Used in MaxProof (2026) to clear human gold-medal thresholds on IMO 2025 and USAMO 2026.

**Prereqs:** [../grpo.md](../grpo.md), [../rlvr.md](../rlvr.md)
**Related:** [prm.md](prm.md) · [orm.md](orm.md) · [population-test-time-search.md](population-test-time-search.md)

---

## What it is

A three-role training setup for tasks where verifying is easier than generating from scratch but harder than running a rule:

- **Generator** — produces candidate solutions via RL.
- **Verifier** — *generative* LLM that reads a candidate and emits both a verdict and a natural-language localization of the first error.
- **Repair model** — conditions on the verifier's critique and produces a patched candidate.

All three share the same backbone family but train on different signals.

---

## How it works

### Verifier training

The verifier is trained on (candidate, ground-truth label) pairs where the label includes *where* the candidate goes wrong. Training data comes from (a) human-annotated proofs and (b) self-generated solutions with errors injected at known positions. The verifier learns to:
- Accept/reject.
- Localize the failing step.
- Explain the failure in natural language.

This is heavier than training an outcome reward model (ORM), but the explanation channel is what enables repair.

### Generator training

Standard GRPO on the generator with the verifier supplying rewards. Rewards are dense — the verifier scores not just final correctness but step-level validity, so the generator gets gradient signal on partial proofs.

### Repair training

The repair model is fine-tuned on triples `(broken candidate, verifier critique, fixed candidate)`. At inference time, the verifier's localization is fed to repair, which produces a patched candidate that gets re-verified.

### Joint loop

Generator, verifier, and repair improve in tandem: better generators surface harder failure modes for the verifier; better verifiers give cleaner gradients to the generator; better repair extends the population search horizon.

---

## Why it matters

- **Long-form correctness without rule verifiers.** Proofs and complex code don't admit a one-line verifier; generative verifiers fill the gap.
- **Composability with test-time search.** The verifier's localization is the input that makes [population-test-time-search](population-test-time-search.md) cheap — repair candidates locally instead of regenerating.
- **Competition-math gold.** MaxProof reports 35/42 IMO 2025, 36/42 USAMO 2026 — above human gold cutoffs — using exactly this stack.

---

## Gotchas & tricks

- **Verifier reward hacking.** A generator can learn to fool a weak verifier; periodic adversarial co-training (have the generator try to produce subtly wrong proofs the verifier accepts) keeps the verifier honest.
- **Critique quality bottlenecks repair.** If the verifier localizes errors vaguely, repair regresses to "rewrite everything". Train the verifier on tight, step-level localization.
- **Verifier compute matters at inference.** Verifying every candidate in a population is the second-largest inference cost (after generation). Distill verifiers or use a hierarchy (cheap reject + expensive verify).
- **Domain dependency.** The recipe is sharpest where solutions have explicit step structure (proofs, code) — looser-structure tasks need a different localization scheme.

---

## Sources

- Paper: *MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling* — Zhang et al., MiniMax + Fudan, 2026 — [arXiv:2606.13473](https://arxiv.org/abs/2606.13473) — generator + verifier + repair training; IMO/USAMO numbers.
