# Image Tokenizer Evaluation in Unified Multimodal Training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In unified multimodal LMs, an image tokenizer's **reconstruction quality is a weak predictor of downstream capability**. Task-conditioned scaling laws differ meaningfully across tokenizers, and rankings flip depending on which tokens the model has to predict. Picking a tokenizer on FID or PSNR alone is measurably wrong.

**Prereqs:** [../fundamentals/_tokenization](../fundamentals/_tokenization.md), [README](README.md)
**Related:** [README](README.md)

---

## What it is

Unified multimodal models (see [mixture-of-transformers](../architectures/mixture-of-transformers.md), and encoder-free / VAE-free systems generally) put an image tokenizer in a load-bearing position: every pixel goes through it, and the LM's whole capability is expressed over its token space. Picking one has historically been done on reconstruction metrics (FID, PSNR, VAE recon loss).

This work sets up a controlled study of image tokenizers inside a joint text+image LM. The main finding: reconstruction quality tells you about the tokenizer's decoder, not about the *token space* the LM has to reason over.

## How it works

- **Controlled sweep.** Fix the LM backbone; vary tokenizer design axes — discriminator architecture, whether/how semantic supervision is applied, codebook size, code dimensionality.
- **Joint text+image loss decomposition.** Track loss *per task* (text-only, image-only reconstruction, T2I generation, VQA understanding) as parameters and tokens scale.
- **Scaling-law fits per task.** Each tokenizer gets its own scaling curve per task. Rankings among tokenizers are read off these curves rather than a single number.
- **Correlation checks.** Compare each task's loss ordering to the reconstruction-only ordering.

## Why it matters

Native unified models live or die by their tokenizer, and the field's convention of picking on reconstruction metrics is dangerous — a tokenizer that reconstructs beautifully can still be worse for VQA or generation. Task-conditioned evaluation is the right calibration; without it, tokenizer choice is a hidden confounder in every unified-model result.

## Gotchas & tricks

- **Loss curves must be per-task.** A single joint loss averages away the disagreements between tokenizers.
- **Semantic supervision helps understanding but can hurt generation.** The two objectives pull the tokenizer in different directions; balance is task-dependent.
- **Codebook size interacts with LM size.** Bigger codebooks help large LMs more; small LMs saturate.
- **Reconstruction is not useless — it's a *lower bound* on token expressiveness.** A tokenizer that can't reconstruct won't reason either. Just don't rank on it alone.
- **Redo the study when the LM backbone changes.** Findings tied to a specific backbone (attention pattern, MoE routing) may not transfer.

## Sources

- Paper: *Studying Image Tokenizers as Visual Languages in Unified Multimodal Models* — Li, Wang, Du, Chen, Liu (University of Washington), 2026 — https://arxiv.org/abs/2609.09143
