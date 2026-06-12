# MTP-Accelerated RL Rollouts (Bebop)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Multi-Token Prediction (MTP) heads, originally designed as a pretraining auxiliary objective, naturally serve as a **speculative-decoding draft model** during RL post-training rollouts. The problem: MTP acceptance rates degrade as RL training progresses — the policy's distribution shifts under exploration, the MTP head's drafts go off-distribution, and the speedup collapses. **Bebop** (Qwen Team, 2026) fixes this by adding **entropy-bounded rejection sampling**: drafts that fall outside a target entropy budget are rejected, keeping the accepted set on-distribution as the policy moves. Result: MTP speedup is preserved throughout large-scale RL.

**Prereqs:** [../pre-training/mtp.md](../pre-training/mtp.md), [partial-rollouts.md](partial-rollouts.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) · [../post-training/grpo.md](../post-training/grpo.md) · [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

RL post-training has two big cost centers: **policy update** (forward+backward through the model) and **rollouts** (just forward, but many samples per prompt and long sequences). For GRPO-style algorithms with $G \geq 16$ rollouts per prompt and long-CoT outputs, rollouts dominate. Anything that makes rollouts cheaper translates directly into more RL steps per dollar.

MTP heads were trained alongside the base model to predict multiple tokens at once. At inference, they're a natural draft model for **self-speculative decoding** — the MTP head proposes the next $k$ tokens, the main model verifies in parallel, and accepted tokens are emitted in a single forward pass.

The naive composition (MTP speculation during RL rollouts) initially gives 2–3× speedup, then **collapses to near-baseline after a few hundred RL steps**. Bebop's contribution is diagnosing why and fixing it.

---

## How it works

### Why naive MTP collapses during RL

RL exploration *raises policy entropy*. A high-entropy policy samples from a flatter distribution; the MTP head was trained on the original (low-entropy) pretraining distribution, so its drafts are increasingly poor matches for what the current policy would sample. Speculative-decoding acceptance rate drops to near zero, and the verification overhead now exceeds the savings.

The collapse is monotonic with training step count, not random. It's the predictable consequence of policy-distribution shift outpacing the MTP head's fixed training distribution.

### The entropy-bounded rejection-sampling fix

For each MTP draft, compute the entropy of the proposed token distribution and check against a target budget $H_{\text{max}}$:

```
draft_tokens = mtp_head(context)
draft_entropy = compute_entropy(draft_distribution)
if draft_entropy > H_max:
    reject draft, fall back to single-token sampling
else:
    accept draft, verify in parallel with main model
```

Drafts that come from high-entropy regions of the input distribution are pre-emptively rejected before the expensive parallel verification. The accepted set stays on-distribution relative to a controlled entropy regime, so verification acceptance rate stays high.

The entropy budget $H_{\text{max}}$ is a hyperparameter that trades off draft acceptance rate vs draft-rejection rate. Tuned to keep the *product* (draft × verify accept) maximized.

### Why this composes with GRPO

GRPO samples $G$ rollouts per prompt; each rollout is independent. MTP+RS acceleration applies per-rollout, so the savings scale with $G$. Bebop reports preserved speedup across thousands of RL steps where naive MTP collapses by step ~500.

---

## Why it matters

- **Unblocks RL rollouts at scale.** Rollout cost is the binding constraint of frontier RL post-training. Cutting it 2–3× has the same effect as 2–3× more compute.
- **Reuses MTP heads.** If you trained with the MTP auxiliary loss (DeepSeek-V3, Qwen, etc.), the head is sitting on disk. Bebop turns it into a free rollout accelerator.
- **First MTP-during-RL recipe that doesn't collapse.** Prior work showed MTP works for SFT inference; Bebop is the systematic study of why RL is different and how to preserve the speedup.
- **Composable with other rollout systems.** Partial rollouts (Kimi-style), batched rollouts, and Bebop's MTP+RS are orthogonal — stack them.

---

## Gotchas & tricks

- **Entropy budget must be tuned per model.** A budget that keeps acceptance high on a 7B model may be too tight on a 70B model where the policy has more headroom. Sweep $H_{\text{max}}$ at the start of training.
- **Budget should adapt over training.** As the policy distribution shifts during RL, the optimal $H_{\text{max}}$ drifts. Recompute periodically — fixed budgets eventually re-trigger collapse, just slower.
- **Don't confuse with output-side rejection sampling.** Bebop's rejection sampling is over MTP *drafts*, not over RL rollout *outputs*. Output-side rejection sampling (see [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)) is an orthogonal use of the same name.
- **Verification cost still scales with $k$.** The main model still has to verify the proposed $k$ tokens in parallel. Picking $k$ too large wastes verification compute on long rejected drafts.
- **Works best for verifiable-reward RL.** When rewards are sparse and reliable (RLVR), the policy's entropy stays meaningful and the entropy bound is informative. For dense reward-model-based RL with noisier rewards, the entropy signal is weaker.

---

## Sources

- Paper: *Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling* (Bebop) — Qwen Team, Alibaba, 2026 — [arXiv 2606.12370](https://arxiv.org/abs/2606.12370).
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — pretrained MTP heads used as inference accelerators.
- Concept: Multi-Token Prediction as a pretraining objective — see [../pre-training/mtp.md](../pre-training/mtp.md).
