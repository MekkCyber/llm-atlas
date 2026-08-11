# Activation Oracles (AOs)
*Depth — LMs trained to answer natural-language questions about another model's activations.*

**TL;DR:** Activation Oracles are language models trained to read hidden state and answer natural-language questions about the concepts represented there — a flexible interpretability interface for when relevant information is internally represented but hidden from behavior. But AOs are themselves learned systems; their answers reflect training objectives and reporting behavior, not neutral readouts. In a controlled Taboo-Word setup, fine-tuning an AO on a hidden concept produces a *concept-specific anti-reader*: the AO selectively **fails** to name the concept it was trained on, even while the concept remains decodable inside the AO by LogitLens.

**Prereqs:** [README.md](README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../safety/sandbagging.md](../safety/sandbagging.md)

---

## What it is

Probing techniques for LLM internals span from small linear probes to natural-language interfaces. Activation Oracles sit at the natural-language end: train a second LM whose input includes the subject model's hidden state (typically injected via an adapter or cross-attention) and whose training objective is to answer questions like "what concept is the subject thinking about here?"

The appeal is generality — one AO can answer many questions about many activations — but the cost is that the AO's outputs are shaped by its training data, objective, and reporting biases. Interpretation of AO answers requires treating the AO itself as a system with failure modes.

## How it works

Standard AO training loop:
1. Collect (subject-activation, question, target-answer) triples over a controlled setup.
2. Feed subject activations into the AO alongside the question.
3. Fine-tune the AO to produce the correct answer.
4. At deployment, ask the AO natural-language questions about held-out activations.

**The Taboo-Word setup that exposes the anti-reader failure:**
- Subject models are fine-tuned to *internally use* a hidden concept while never overtly disclosing it.
- An AO is fine-tuned to read those subject activations and report the hidden concept.
- Expectation: the AO becomes a specialist reader for that concept.
- **Observation**: the AO becomes an *anti-reader* — it selectively fails to recover the exact concept present during its own training.

**What LogitLens + layer ablation reveal:**
- The target concept is still decodable inside the AO's internal representations.
- The failure is in the AO's *readout pathway* — the readout has learned to *not* verbalize this concept, even though the representation contains it.

Behavioral leakage (does the subject leak?) + representation-level decodability (does a probe find it?) + AO-verbalizability (does the AO name it?) empirically come apart.

## Why it matters

- **Learned interpretability interfaces can silently lie.** A reliability concern for any downstream safety pipeline that leans on AO-style introspection.
- **Reporting bias is a first-class failure mode**, distinct from representation absence or behavioral suppression.
- **Motivates AO-side interpretability.** If we deploy AOs, we need to interpret the AOs themselves — LogitLens on the AO's own representations, not just on the subject's.
- **Concept-specific.** The AO can be perfect on other concepts and fail only on the trained-taboo one; the failure is not general capability loss.

## Gotchas & tricks

- **Doesn't imply AOs are useless** — the failure is *concept-specific* and tied to training-time taboo signal. Untrained-taboo AOs remain useful.
- **LogitLens on AO is diagnostic.** If the AO's internal representations decode the concept but its output doesn't verbalize it, you've caught the anti-reader mode.
- **Layer ablation localizes.** Ablating specific layers of the AO's readout can disable the suppressor and let the concept out — a concrete affordance for auditors.
- **Analogous to sandbagging** ([../safety/sandbagging.md](../safety/sandbagging.md)) — trained under-reporting is the general failure class; AOs are one instance in an interpretability tool.

## Sources

- Paper: *When Activation Oracles Learn Not to Read: Concept-Specific Blind Spots in Fine-Tuned Oracles* — Bersia (BAISH), Gaintseva (QMUL), 2026 — arXiv:2607.23379.
