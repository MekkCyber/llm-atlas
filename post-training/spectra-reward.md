# SpectraReward
*Depth — training-free reward for text-to-image RL using the log-likelihood of the prompt reconstructed from the generated image.*

**TL;DR:** SpectraReward turns any pretrained multimodal LLM into a zero-shot reward model for text-to-image generation. Given a prompt–image pair, it feeds the image + prompt through the MLLM in a **single teacher-forced forward pass** and reads the average log-likelihood of the prompt tokens. If the image encodes the prompt, the prompt is easy to predict; if it doesn't, the log-likelihood collapses. No preference labels, no reward-model fine-tuning, no separate judge.

**Prereqs:** [_rewards](_rewards.md), [grpo](grpo.md)
**Related:** [cot-reward-model](cot-reward-model.md), [rlvr](rlvr.md)

---

## What it is

A new reward-family entry for image-generation RL. Existing options: preference reward models (need labels, hackable), MLLM-as-judge with generated critiques (expensive, biased by the judge's language priors), decomposed VQA (brittle question decomposition). SpectraReward instead **reuses the pretrained image-text alignment head of any MLLM directly** as a scoring function.

The signal is `R(prompt, image) = mean log P_MLLM(prompt | image)`. It's training-free at the reward layer — the MLLM is not tuned to be a judge. It's parameter-free — no scalar head, no preference fine-tune.

## How it works

1. Take a pretrained MLLM (`Understand(image, prompt) → prompt-token-logits`).
2. For a generated image `x` and its prompt `p`, run a **single teacher-forced forward pass**: feed the image + prompt into the MLLM as if you were computing the LM loss on `p` conditioned on `x`.
3. Reward is the average per-token log-likelihood of the original prompt tokens:  `R(p, x) = (1/|p|) Σ log P(p_t | x, p_<t)`.
4. Plug this reward into any RL algorithm — the paper uses [GRPO](grpo.md).

**Self-SpectraReward.** For unified any-to-any models (one backbone with both understanding and generation branches), the same model's understanding branch scores its own generation branch. This closes the RL loop with **no external reward model at all** — the policy grades itself.

## Why it matters

- **Cost.** One MLLM forward pass per image. Cheaper than a CoT judge (which decodes hundreds of tokens per score). Comparable to a scalar preference RM but with none of the label/training cost.
- **Alignment.** The reward directly measures whether the pretrained image-text encoder can *see* the prompt in the image. That's exactly what "prompt fidelity" is supposed to mean, so hacking the reward means hacking the encoder — much harder than hacking a scalar-head RM.
- **Unified models get self-improvement for free.** Self-SpectraReward is the closed-loop generalization of "the model is its own judge" — the same tokenizer, the same alignment head, no separate judge to keep in sync.

## Gotchas & tricks

- **Prompt-length normalization matters.** Raw log-likelihood scales with prompt length; the mean normalization is what makes reward comparable across prompts of different lengths.
- **Teacher forcing means the reward reads the *original* prompt.** If the prompt is ambiguous, the reward is inherently ambiguous too — SpectraReward can't beat prompt clarity.
- **Free-form rephrasings can leak reward.** If a related MLLM is trained on the same web captions the T2I model uses, it can score images that look right but miss subtle prompt constraints. Freezing the reward MLLM at a different checkpoint from the generator helps.
- **Self-SpectraReward is subject to shared-error blindness.** If the understanding branch is wrong the same way the generation branch is wrong, the loop reinforces the shared error. Using a *different* MLLM for scoring at least some of the time is a hedge.

## Sources

- Paper: *Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation* — Huang et al., HKU / PKU / ByteDance Seed, 2026 — arXiv 2607.11886.
