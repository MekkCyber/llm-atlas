# Activation Explanation Faithfulness (Decodability + RECAP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Natural-language autoencoders judge activation explanations by whether the activation can be **reconstructed** from the explanation. The 2026 RECAP paper shows this test is passed in two unfaithful ways: (1) reconstruction tracks the input's **gist** rather than specific claims (only ~2% of claims in a released Qwen-2.5-7B verbalizer are reconstruction-dependent), and (2) under exact synthetic ground truth, the standard recipe develops **co-adapted private codes** — false wording the reconstruction depends on — in 5/5 runs. RECAP proposes two audit protocols and a training-side alternative: linear heads trained alongside the target model to keep designated content decodable.

**Prereqs:** [README.md](./README.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

A **natural-language autoencoder** for interpretability works like: (encoder) hidden activation → natural-language explanation; (decoder / reader) explanation → reconstructed activation. The **reconstruction loss** trains the encoder to produce explanations that preserve enough information for the decoder to invert. The community's assumption has been that a low reconstruction loss ⇒ the explanation is *faithful* to the activation.

RECAP shows the assumption is broken two ways:

1. **Gist-tracking.** On a released Qwen-2.5-7B verbalizer, only ~2% of the specific claims in an explanation are actually needed for reconstruction. Flipping the other 98% of claims doesn't change what the decoder produces. The faithfulness score tracks paraphrases and general topic, not individual factual assertions.
2. **Co-adapted private codes.** Under an exact synthetic ground truth, the standard recipe develops **false wording** that the reconstruction depends on — a hidden channel between encoder and decoder that has nothing to do with the underlying activation. Happens in 5/5 runs.

## How it works

**Audit protocols.**
- **Grounded-vs-true cross.** Systematically flip specific claims in an explanation and measure whether reconstruction changes. If not, that claim is not reconstruction-dependent — it was never being tested.
- **Evaluator swap.** Swap the trained decoder for a different one. If reconstruction quality collapses, the two models were sharing a private code. If it holds up, the explanation is decoder-agnostic and closer to faithful.

**RECAP (Readable Encodings via Co-trained Auxiliary Predictors).** Instead of training the reader to reconstruct activations, add **linear heads to the target model** that are trained jointly to keep specified content decodable *directly from the activation*. The auxiliary heads have to succeed with linear probing only — no room for a private channel. The verbalizer then has to describe what those linear heads have shown to be decodable, which grounds the faithfulness claim in a checkable property of the model itself.

## Why it matters

Reconstruction-scored explanations underpin much of modern SAE-and-verbalizer interpretability. If the reconstruction can be fooled by gist or private codes, most existing "faithful explanation" claims need re-evaluation. RECAP moves the burden of proof onto the *target model's* linear-decodability, which is a much harder property to spoof.

This also directly affects **CoT monitoring** and safety verbalizers: if a model's "explanation of what it just did" is really tracking gist plus a private code, monitoring based on that explanation gives false assurance.

## Gotchas & tricks

- The audit protocols are cheap and should be run on *any* natural-language autoencoder before trusting its faithfulness score.
- RECAP requires access to the target model's weights (to add linear heads) — not applicable to black-box models.
- Linear heads are the strictest decodability test; nonlinear probes can decode information linear heads can't. RECAP's claim is stronger *because* of the linearity constraint.
- Doesn't solve the general "what should an explanation say" problem — it changes how faithfulness is measured, not what content is worth explaining.

## Sources

- Paper: *Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations* — 2026 — [arXiv:2607.20379](https://arxiv.org/abs/2607.20379)
- Prior art: Natural-language autoencoder verbalizers for SAE features (the recipe RECAP critiques).
