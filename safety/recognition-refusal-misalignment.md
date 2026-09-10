# Recognition-refusal misalignment
*Depth — instruct-tuned LLMs represent "this question is impossible" as a linear direction, but that direction is near-orthogonal to the trained safety-refusal direction.*

**TL;DR:** LLMs often confidently answer structurally unanswerable questions (`cot(-540°)`, `(1).startswith("1")`) instead of abstaining. A single linear probe on the hidden state cleanly separates answerable from structurally impossible prompts — so the model **does** recognise impossibility. But that recognition direction is nearly orthogonal to the canonical safety-refusal direction that RLHF trains for harmful content. The failure is a routing bug, not a missing signal.

**Prereqs:** *(interpretability folder has only a README today; no depth prereqs yet)*
**Related:** [../interpretability/activation-steering.md](../interpretability/activation-steering.md), [refusal-suppression.md](refusal-suppression.md), [_attacks.md](_attacks.md)

---

## What it is

A mechanistic account of a large class of LLM hallucinations. "Structurally unanswerable" questions are prompts whose type or preconditions preclude an answer (`math out of domain`, `code type errors`, `nonexistent-entity queries`). Instruct-tuned models overwhelmingly answer these instead of abstaining; the paper asks whether the failure is *encoding* (the model doesn't know) or *routing* (it knows but the refusal circuit doesn't listen).

## How it works

Two linear probes are fit on residual-stream activations of instruct-tuned models spanning 1.7B–70B parameters:

- **Recognition direction (`v_rec`)** — separates answerable prompts from structurally impossible math/code prompts.
- **Safety-refusal direction (`v_ref`)** — the canonical direction shown to mediate refusal for trained harmful-content categories.

Findings:

- `v_rec` is present at the pretraining endpoint (base and instruct versions both carry it).
- `cos(v_rec, v_ref) ≈ 0`: the two directions are near-orthogonal, so `v_rec` cannot drive `v_ref`.
- An **in-domain behavior-defined invalidity direction** (fit from prompts where the model chose to abstain) is closer to `v_rec` than to `v_ref`, but only partially aligned with `v_rec` — recognition is not fully wired to abstention.
- **Generation-time steering** along `v_rec` changes invalidity-aware behavior *bidirectionally and dose-responsively*; random directions do not. This is a causal test that `v_rec` is the operative signal for structural impossibility.

Interpretation: RLHF trained a refusal pathway on trained harmful categories, and generalisation to structural impossibility did not propagate to that pathway.

## Why it matters

- **Reframes hallucination-on-unanswerables** as an alignment routing gap, not a knowledge gap. This changes what training data helps.
- **Suggests specific interventions.** Fine-tune with abstention labels *on structurally impossible prompts* to rotate `v_rec` toward `v_ref`, or add an inference-time gate that steers along `v_rec` when the recognition score is high.
- **Mechanistic benchmark.** A dataset of structurally impossible questions gives a controlled, cheap-to-grade capability probe that decouples "does the model know?" from "does the model refuse?".
- **Generalises beyond math/code.** The routing-vs-encoding decomposition is a template for other hallucination classes (nonexistent citations, made-up API arguments).

## Gotchas & tricks

- **Probe direction choice.** Fit `v_ref` on the paper's canonical setup (harmful vs. benign) — reusing model-specific refusal probes can inflate alignment illusorily.
- **Layer-choice sensitivity.** Recognition is strongest in mid-to-late layers; refusal in later layers. Compare `v_rec, v_ref` at matched layers.
- **Not the same as calibration.** A model can be *aware* the question is impossible without producing a low-confidence answer distribution; recognition is a per-hidden-state property, not a token-probability property.
- **Instruction tuning doesn't create the signal — it fails to route it.** The recognition direction is already at the base-model endpoint.

## Sources

- Du, Hu — *Recognition-Refusal Misalignment in LLMs: Why Models Answer Structurally Unanswerable Questions* — 2026 — https://arxiv.org/abs/2608.29109
- Arditi et al., *Refusal in LLMs is Mediated by a Single Direction* — 2024 — https://arxiv.org/abs/2406.11717
