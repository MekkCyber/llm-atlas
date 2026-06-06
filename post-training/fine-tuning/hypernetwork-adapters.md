# Hypernetwork-Generated Adapters
*Depth — a learned network emits LoRA / adapter weights directly from a conditioning signal (a repo, a user, a domain), so adaptation costs zero extra inference tokens and zero per-condition training.*

**TL;DR:** Instead of fine-tuning a LoRA per repo / user / tenant, train a *hypernetwork* that maps the conditioning signal to LoRA weights. At inference, look up the adapter and plug it in — no retrieval context, no per-condition optimizer step. Recent revival: Code2LoRA (2026) for repository-level code understanding, with a static variant (single snapshot → adapter) and an "Evo" variant where a GRU updates the latent state per commit so the adapter tracks an evolving repo.

**Prereqs:** [post-training/fine-tuning/README.md](README.md)
**Related:** [post-training/_post-training](../_post-training.md)

---

## What it is

Three options for "make the model know about this private corpus / repo / user":

1. **RAG / long context** — inject retrieved text into every prompt. Cost: O(retrieved tokens) per call, forever.
2. **Per-condition LoRA / fine-tune** — train a small set of weights per condition. Cost: one training run per condition; stale when the underlying data changes.
3. **Hypernetwork adapters** — train *one* hypernetwork that takes the condition and outputs LoRA weights. Cost: one big training run, then a forward pass per new condition.

Option 3 dominates when there are many conditions, conditions change, or per-call token cost matters.

## How it works

Architecture, in pseudocode:

```
encoder(condition)  →  z  (a compact embedding)
hypernet(z)         →  {A_ℓ, B_ℓ}  for each LoRA-injected layer ℓ
forward(x; θ_base + A_ℓ B_ℓ for ℓ) → standard forward pass
```

Training: pick a base model with frozen weights and LoRA hooks at chosen layers. Sample `(condition, task)` pairs. Run the hypernetwork to materialize the LoRA, forward the task input through the adapted model, backprop end-to-end into the hypernetwork (not the base, not the LoRA — the *hypernetwork* that produced the LoRA). Standard parameter-efficient fine-tuning loss (cross-entropy on completions).

Variants:

- **Static**: encoder sees a snapshot of the condition (whole repo flattened, user profile, …) and outputs a fixed adapter.
- **Streaming / Evo (Code2LoRA-Evo)**: encoder is a recurrent net (GRU); each incremental update to the condition (a commit, a new doc) ingests via a hidden-state update, and the current adapter is read out from that state. The "condition" becomes a *trajectory*, and the hypernetwork tracks it without recomputing from scratch.

## Why it matters

- **Zero inference-token overhead.** Unlike RAG, the conditioning doesn't appear in the prompt.
- **Constant amortized cost per new condition.** A single hypernetwork forward pass, not a fine-tune.
- **Tracks evolution.** With a recurrent encoder, the adapter follows a changing condition (active codebase, growing user history) at the cost of per-update GRU steps.

## Gotchas & tricks

- **Encoder budget dominates.** The hypernetwork still has to *read* the condition. For a 100k-line repo this is non-trivial; Code2LoRA trades RAG's per-call cost for an upfront encoder cost.
- **Coverage of unseen conditions.** Generalization across conditions is empirical; if the hypernetwork hasn't seen repos like the test repo, it underperforms per-repo fine-tuning.
- **Stable training requires a strong condition distribution.** RepoPeftBench (604 Python repos, 40K static / 215K commit-derived tasks) is large because hypernet training is data-hungry.
- **Adapter capacity is the choke point.** A larger LoRA rank gives the hypernetwork more "bits" to encode the condition; too small and conditioning blurs across repos.
- **Not new** — HyperNetworks (Ha et al. 2017) and the parameter-generation literature predate this by a decade. What's new is the combination with LoRA + repo-scale code conditioning.

## Sources

- Paper: *Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution* — Hotsko et al., 2026 — [arXiv:2606.06492](https://arxiv.org/abs/2606.06492).
- Paper: *HyperNetworks* — Ha, Dai, Le, 2017 — original hypernetwork formulation.
- Code & data: `huggingface.co/code2lora`, `anonymous.4open.science/r/code2lora-6857`.
