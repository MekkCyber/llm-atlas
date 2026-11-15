# Video RLHF with an LLM Physics Judge (PhysLoop)
*Depth — self-improving video diffusion via LLM-as-physics-judge feedback.*

**TL;DR:** Video diffusion models struggle with physical plausibility (object permanence, gravity, contact dynamics). Hand-building a physics simulator or scraping physics annotations is expensive. PhysLoop sidesteps both by using a **frontier multimodal LLM as a physics judge**: inspect a generated clip, return dense verdicts on physical plausibility, and distil those verdicts into reward / preference signals that drive policy updates of the video generator. The whole loop runs closed: generate → LLM critique → preference / reward update → regenerate.

**Prereqs:** [../post-training/cot-reward-model](../post-training/cot-reward-model.md)
**Related:** [../post-training/rlvr](../post-training/rlvr.md), [../post-training/dpo](../post-training/dpo.md)

---

## What it is

The template now familiar from text/code RLVR (use a verifier as the gradient) exported into the video domain. The verifier here is an LLM evaluating physical plausibility — a soft verifier, not a formal check — but one that produces structured, chain-of-thought feedback the video model can train on.

## How it works

- **Generate**: the video diffusion model produces candidate clips for a prompt batch.
- **LLM judge**: a multimodal LLM reads each clip (via a frame sampler) and returns per-axis verdicts: object permanence, gravity consistency, contact dynamics, scene stability. Verdicts are rendered both as scalar scores and as textual critiques.
- **Preference / reward**:
  - For preference-based training: form pairs {better clip, worse clip} within a prompt's batch using the judge's scores and apply a DPO-style loss.
  - For reward-based training: convert the scalar scores into a dense reward and apply a policy-gradient / RLVR-style update.
- **Loop**: the updated model generates again, and the judge re-evaluates. Reflections in the judge's critiques can also prompt the next generation.

## Why it matters

- **Shortcuts physics annotation.** Ground-truth physics labels for video are prohibitively expensive; an LLM physics judge is a cheap proxy that has shown measurable traction on plausibility benchmarks.
- **Reusable template.** The same "frontier LLM as domain judge" pattern can be applied to other hard-to-label video axes (narrative coherence, lighting consistency, 3D consistency).
- **Connects to world-model research.** Video models trained with physics feedback become better world simulators, feeding into agent training and video-based RL.

## Gotchas & tricks

- The LLM judge is **not a ground truth** — it inherits the judge's biases and failure modes (e.g., overrating pretty-but-wrong clips, missing subtle dynamics). The paper reports improvements against strong baselines but the ceiling is the judge's own physics ability.
- Reward hacking shows up quickly if the policy learns to produce visual cues the judge rewards without the underlying physics; standard RL-hacking mitigations (reward-model regularization, KL anchoring) apply.
- Frame-sampling choices in the judge affect which failure modes are detected; too few frames and temporal violations slip through.

## Sources

- Paper: *PhysLoop: Harnessing LLMs' Physics Awareness for Self-Improving Video Generation Models* — Li et al., Peking U. / ByteDance Seed / UCSB / MIT / UHH / HNU, 2026 — arXiv 2611.16748 — https://arxiv.org/abs/2611.16748
- Project page: https://chenyx24.github.io/PhysLoop-site/
