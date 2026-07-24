# Hypernetwork LoRA for Knowledge Injection
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a hypernetwork end-to-end that maps a set of facts to a fixed LoRA adapter which, when inserted into a target LLM, lets the LLM answer questions about those facts. Decouples "learn the injection procedure" from "learn each fact" — new fact sets get a fresh adapter without further gradient descent on the target model. The 2026 scaling-laws paper introduces MegaWikiQA (tens of millions of multi-hop QA examples across 39 Wikidata5M domains) and reports clean scaling in hypernet depth, width, target model size, and fact count.

**Prereqs:** [README.md](./README.md), [../_post-training.md](../_post-training.md)
**Related:** [../../data/_data-curation.md](../../data/_data-curation.md), [../../pre-training/mid-training.md](../../pre-training/mid-training.md), [../rejection-sampling.md](../rejection-sampling.md)

---

## What it is

Continual pretraining is the standard route to inject new factual knowledge into an LLM: mix new tokens into the training data, keep training. It works but is expensive and mostly irreversible. LoRA fine-tuning is cheaper but still requires a gradient descent step per new fact set.

Hypernetwork LoRA reframes injection: **train a network whose output is the LoRA weights**. Given a fact corpus at deployment time, run the hypernetwork once to produce an adapter; insert it into the target model; query the LLM as normal. No per-fact gradient descent on the target model.

Hypernetworks have historically been used for test-time adaptation. The 2026 paper is the first to train them at scale at **train time** with the specific goal of factual injection, and to characterize the scaling behavior.

## How it works

**Training.** Given a large fact corpus (MegaWikiQA: tens of millions of multi-hop Wikidata5M QA examples), for each mini-batch:

1. Sample a subset of facts.
2. Encode the facts into a set-conditioning representation.
3. Hypernetwork maps the representation to a full LoRA adapter (Δ_A, Δ_B for each target-model layer).
4. Insert the adapter into the frozen target LLM and evaluate on QA examples derived from the same fact subset.
5. Backprop the QA loss through the LLM (frozen) into the hypernetwork (trainable). Only the hypernetwork updates.

**Inference.** At deployment, hand the hypernetwork a fresh fact set, get an adapter, plug it in. The target LLM never changes.

## Why it matters

- **Cheap per-slice knowledge injection.** No per-fact-set fine-tuning. Adapter generation is a single forward pass through the hypernetwork.
- **Reversible.** Detaching the adapter restores the base model exactly. Continual pretraining doesn't allow this cleanly.
- **Composable with LoRA-serving infrastructure.** Since the output is a standard LoRA, existing multi-adapter serving stacks (vLLM's `--enable-lora`, S-LoRA) work unchanged.
- **Predictable scaling.** The paper reports clean scaling laws across four axes (hypernet depth, hypernet width, target size, fact count), enabling budget-based capacity planning.

## Gotchas & tricks

- The hypernetwork itself is often large — hundreds of millions of parameters to output a full-model LoRA. The savings come from *not retraining the base LLM per fact set*, not from tiny hypernets.
- Set-conditioning matters. Naïve concatenation of fact strings breaks at scale; the paper likely uses a set encoder to make output invariant to fact ordering (not confirmed in the abstract).
- MegaWikiQA is multi-hop; single-hop-only fact injection is easier and may not need a hypernetwork.
- Cross-adapter interference is unstudied. Combining a hypernet-generated adapter with a task-tuned LoRA is not addressed.

## Sources

- Paper: *Scaling Laws for HyperNetwork-Based Knowledge Injection in Large Language Models* — Dhankhar, Baha, Saparov, 2026 — [arXiv:2607.19604](https://arxiv.org/abs/2607.19604)
- Dataset: *MegaWikiQA* (introduced in the same paper) — tens of millions of multi-hop QA examples across 39 Wikidata5M domains.
