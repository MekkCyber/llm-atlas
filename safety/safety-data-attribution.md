# Safety Data Attribution (SimpleLoRIF / SafeTulu)
*Depth — attribute post-training safety regressions back to specific benign-looking training examples via low-rank gradient influence.*

**TL;DR:** Fine-tuning on instruction data can quietly weaken safety even when *no individual example looks harmful*. SimpleLoRIF (Simple Low-Rank Influence Function) computes a cheap low-rank gradient embedding for every candidate training example, scores how well its gradient aligns with the direction that produces unsafe behavior at inference, and drops the top-aligned slice. Applied to Tulu 3 → **SafeTulu** removes the top-20% most aligned examples; GCG attack success drops by up to 22% across Qwen / Llama / Olmo with minimal capability loss. Emergent-misalignment detection AUC up to 0.865.

**Prereqs:** [../data/_data-curation.md](../data/_data-curation.md), [../data/quality-filtering.md](../data/quality-filtering.md), [_jailbreaks.md](_jailbreaks.md)
**Related:** [refusal-suppression.md](refusal-suppression.md) · [../data/decontamination.md](../data/decontamination.md) · [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Safety-training literature usually assumes bad behavior comes from bad examples: red-team the SFT set, remove the obvious harms, retrain. **Follow the Gradient** (Reddy et al., 2026) shows this misses a major source: *benign-looking examples* whose gradient happens to point in the same direction as unsafe behavior at inference time. Removing them improves robustness even when no explicit harmful content was present.

SimpleLoRIF is the method that scores each training example by its gradient alignment with target unsafe behavior; SafeTulu is the resulting curated version of the Tulu 3 SFT corpus.

## How it works

### The influence question

Classical influence functions ask: how much would the model's loss on a *test* example change if a *training* example were removed? Direct computation requires the loss Hessian, which is intractable at LLM scale.

### Low-rank approximation

SimpleLoRIF replaces the Hessian solve with a low-rank gradient *embedding*: project each example's loss gradient down to a low-dim subspace, and measure alignment by inner product. Concretely, for a candidate training example $x_i$ and a target unsafe prompt-response $(q, r_{\mathrm{unsafe}})$:

$$
\mathrm{align}(x_i) \;=\; \langle P \nabla_\theta L(x_i),\; P \nabla_\theta L(q, r_{\mathrm{unsafe}}) \rangle
$$

where $P$ is a learned or random low-rank projection. The "simple" is that $P$ avoids both the Hessian solve and the per-example full-gradient storage.

### SafeTulu construction

1. Collect a bank of unsafe target behaviors (e.g. GCG completions that an aligned base refuses but a Tulu-SFT'd variant produces).
2. For each example in Tulu 3, compute its SimpleLoRIF alignment with the unsafe bank.
3. Rank; drop the top-20%.
4. Fine-tune on the surviving ~80%.

### Measured effect

- **GCG attack success rate drops by up to 22%** across Qwen-7B, Llama-3-8B, Olmo-7B after SFT on SafeTulu vs full Tulu 3.
- **Capability metrics (MMLU, GSM8K, IFEval)** largely preserved — the dropped slice is not the slice carrying general capability.
- **Emergent-misalignment detection AUC up to 0.865** on held-out probes.

## Why it matters

- **Data-side alignment, not RLHF patching.** Moves safety work earlier in the pipeline: filter *before* training, not re-align after.
- **Explains the "safety free lunch" of smaller SFT sets.** Smaller SFT sets have long been observed to produce safer models at little capability cost; SimpleLoRIF suggests the mechanism — larger sets carry more gradient-aligned-with-unsafe mass.
- **Reusable attribution tool.** The low-rank gradient embedding is useful beyond safety — anywhere you want to attribute a model behavior to a training-data subset without a Hessian.

## Gotchas & tricks

- **Target bank quality dominates.** The method is only as discriminating as the unsafe-behavior set used to compute the alignment direction. Narrow banks produce narrow fixes.
- **Projection choice matters.** Random projection is cheap but noisier; learned projection (via a small gradient PCA) is better but adds a step. Paper uses a mix.
- **No causal guarantee.** Alignment is correlational: dropping a top-aligned example doesn't *prove* it was the cause. The empirical drop in GCG success is the real evidence.
- **The 20% drop is a knob.** More aggressive drops improve safety further but at eventual capability cost. The paper locates the sweet spot at 20% on Tulu 3; other corpora will differ.
- **Doesn't catch adversarial poisoning.** If attackers craft gradient-null examples, SimpleLoRIF misses them. Pair with provenance-side defenses.

## Sources

- Paper: *Follow the Gradient: Tracing Safety Degradation Back to Training Data* — Saket Reddy, Amro Abbas, Anshuman Suri, Ari Morcos, David Schwab, Pratyush Maini — DatologyAI + UIUC, 2026 — arXiv 2610.37481.
- Related: *TRAK* — Park et al., 2023 — scalable gradient-attribution for training data.
- Related: *Studying Large Language Model Generalization with Influence Functions* — Grosse et al., Anthropic, 2023.
