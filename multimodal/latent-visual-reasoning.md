# Latent Visual Reasoning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reasoning mode for multimodal LLMs where intermediate steps happen in **continuous visual tokens** instead of words. Avoids verbal chain-of-thought's decoding cost, but introduces a *latent evidence-credit gap*: latent tokens barely respond to image perturbations that change the answer. ReaLVR (2026) closes this gap with a contrastive objective on correct vs. generated-wrong trajectories, grounding the latent chain in visual evidence.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../fundamentals/attention.md](../fundamentals/attention.md).
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md).

---

## What it is

Verbal chain-of-thought generates reasoning tokens one at a time through the LM head, decoding every step. For multimodal tasks the sequences grow long and the decoding cost dominates. **Latent CoT** keeps intermediate reasoning as continuous vectors in the model's hidden space, never decoding until the final answer. The visual flavor supervises those vectors to track the image.

The failure mode ReaLVR names: when the image is perturbed in a way that flips the ground-truth answer, the latent reasoning tokens barely move. The chain is cosmetic — reasoning doesn't actually route through visual evidence.

---

## How it works

1. **Diagnose the gap.** Perturb the image (occlusions, replacements, counterfactual edits). Measure how much latent reasoning-token activations change. For most VLMs, change is small — far smaller than the change in the final answer. This is the *latent evidence-credit gap*.

2. **Contrastive visual-evidence supervision.** For each training sample, use the model to generate a trajectory that produces the correct answer and one or more trajectories that produce wrong answers. Push the latent trajectory distribution toward the correct one and away from wrong ones, conditioning on the image.

3. **Supervise latent tokens directly.** The loss acts on the pre-decode latent sequence; the final answer head is trained separately. This forces the latent chain to encode image-dependent evidence rather than image-independent linguistic habits.

---

## Why it matters

- **Latent CoT is only useful if it works.** Prior latent-reasoning methods were fast but unreliable on vision tasks; this is the first supervision recipe that moves the needle.
- **Peak result:** 63.7% on Qwen2.5-VL-7B; scales up to a 235B-parameter backbone, suggesting the gap-closing recipe is architecture-agnostic.
- **Opens a cheaper reasoning path.** Latent reasoning avoids per-step logit decoding, which matters more as VLM reasoning chains grow longer (spatial queries, multi-image, video).

---

## Gotchas & tricks

- **Needs paired correct/wrong trajectories.** The contrastive signal depends on the model being able to produce both. On tasks where the model is always right (or always wrong), the signal collapses.
- **Perturbation design is load-bearing.** Random noise doesn't help; the perturbations must change the ground-truth answer. Think counterfactual edits, object removal, swapped captions.
- **Still needs a verbal verifier.** The gap diagnosis compares final verbal answers, so an upstream reward signal (RLVR-style) is still required for the correct/wrong split.
- **Doesn't replace all CoT.** For tasks that need auditable reasoning (safety, interpretability, debugging), verbal CoT is still preferable. Latent CoT trades auditability for throughput.

---

## Sources

- Paper: *Rethinking Latent Visual Reasoning: Grounding Latent Reasoning in Visual Evidence* — Xiao, Zhao, Kim, Li, Xu, Wu, Zhang, Xu, Chen, Tejani, Zablocki, Xu, Xing, 2026. arXiv:2609.34563.
