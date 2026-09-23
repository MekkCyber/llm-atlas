# Concealed-Information Probe (PIR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **reference-free** probe of an LLM's hidden state that reads *which candidate answer the model recognizes as correct*, even when the model refuses to say. Inspired by the forensic Concealed Information Test: present the true detail among plausible decoys and measure a stronger internal response to the recognized item. Distinguishes "won't say" from "doesn't know" — cleanly separating sandbagging from honest ignorance and validating that unlearning actually removed knowledge.

**Prereqs:** [../safety/sandbagging.md](../safety/sandbagging.md)
**Related:** [../safety/deceptive-alignment.md](../safety/deceptive-alignment.md) · [../safety/alignment-faking.md](../safety/alignment-faking.md) · [../safety/scheming.md](../safety/scheming.md)

---

## What it is

Behavioral evaluation of language models has a blind spot: when a model outputs "I don't know", we can't distinguish

- the model **doesn't know** the answer, from
- the model **knows** but is **hiding** it (sandbagging, password-locking, backdoor triggers, alignment faking, circuit-breaker refusal).

The **Probe of Internal Recognition (PIR)** answers this by inspecting hidden states rather than outputs. Method:

1. Present a question with candidate answers: one correct, others plausible decoys.
2. Forward-pass the model on each `question + candidate` pair.
3. Read hidden-state features (linear probe on activations) that discriminate "candidate recognized as correct" from "candidate not recognized".
4. The candidate with the strongest recognition signal is the model's *internal* answer, regardless of what it *outputs*.

The probe is trained once, per model, on questions with known answers. It does **not** need an honest reference model or a labeled truth corpus at test time.

## How it works

**Signal.** Under the Concealed Information Test analogy, exposure to a *recognized* item produces a distinctive internal response — even under an instruction to conceal. In an LLM, this shows up as a distinct pattern in mid-to-late-layer activations when the correct candidate is presented.

**Probe.** A small linear classifier (or shallow MLP) is trained on activations at the position of the candidate token to output a "recognition score". Balanced training data of known-correct vs. known-incorrect candidates.

**Reference-freeness.** At test time you only need the model, the question, and the candidate list. No comparison against a different "honest" model, no external verification — the probe's output is the answer.

**Extension to free-form.** The method extends beyond MCQ: for free-form generation, use the sampled top-K continuations as the candidate set and read the recognition score against each. The candidate with the strongest recognition signal is what the model would have generated in the absence of concealment.

## Why it matters

- **Sandbagging audit.** Under-elicitation is a canonical scheming-safety problem: models may deliberately underperform on capability evals to avoid oversight. PIR reads through the sandbag.
- **Unlearning verification.** When you remove knowledge from a model (fine-tuning-based unlearning, gradient ascent, etc.), how do you know the model *actually* forgot vs. just refuses to say? PIR: recognition drops to the "never-knew" baseline exactly when the model has genuinely lost the knowledge.
- **Robust across concealment mechanisms.** In the source paper: robust across prompted deception, trained sandbagging, password-locked models, and circuit-broken checkpoints — recognition score stays **0.85–0.93** across all of them.
- **Causal, not just correlational.** Ablating the probe-selected direction reduces the model's ability to conceal — the recognition signal is on the causal path from knowledge to output.

## Gotchas & tricks

- **Probe transferability.** A probe trained on one model does *not* transfer to another. You retrain per model family. Within a family (e.g. multiple Llama variants) transfer is partial.
- **Choose candidates carefully.** The Concealed-Information analogy only holds if the decoys are *plausible*. Absurd decoys give trivially-easy recognition and inflate the score.
- **Base-rate care.** Report balanced accuracy against an "unknown-item" baseline (0.28–0.40 in the source paper) and chance (0.25 for 4-way MCQ). Raw accuracy can look strong without any real recognition signal.
- **Not a lie detector for opinions.** PIR reads *recognition of a factual candidate*. Applying it to "what does the model believe about X" (where "X" has no ground truth) is not the same problem and doesn't inherit the guarantees.
- **Adversarial fine-tuning may defeat it.** Explicit training against a PIR-style probe (probe-aware defense) is not evaluated; assume future concealment techniques could target the readable representations.

## Sources

- Paper: *A Lie Detector Test for Language Models: Reading Knowledge a Model Won't Reveal* — Hiskias Dingeto, 2026 — introduces PIR, validates across 8 models × 5 families with recognition 0.70–0.87, robust across concealment mechanisms. [arXiv:2609.21996](https://arxiv.org/abs/2609.21996)
- Background: Lykken (1959) — Concealed Information Test in forensic psychology.
- Related: linear probes on hidden states; representation-based sandbagging detection.
