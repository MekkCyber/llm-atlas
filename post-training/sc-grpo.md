# SC-GRPO — Semantic-Consistency GRPO
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO variant for training generative pipelines whose output spans multiple segments (e.g. multi-shot text-to-video plans). On top of the standard preference reward, SC-GRPO adds a **semantic-consistency reward** measured across segments — is shot 3 still faithful to the intent of shot 1? — so the RL update doesn't push the policy toward locally-preferred but globally-drifting outputs. Introduced by Alibaba's Wan team for the WanPE cinematic prompt enhancer (a 397B LLM planner for text-to-video).

**Prereqs:** [grpo](grpo.md), [_rl](_rl.md)
**Related:** [_rewards](_rewards.md), [rlvr](rlvr.md), [av-grpo](av-grpo.md), [rdpo](rdpo.md)

---

## What it is

Long-form generation with a planner-then-generator stack (LLM writes a per-shot screenplay, T2V model renders each shot) fails when the planner paraphrases the user's intent between shots — the video ends up preferred locally but semantically incoherent globally. SC-GRPO targets *the planner*, adding a reward term computed *across shot boundaries* to keep the plan on-intent as the horizon grows.

## How it works

Standard GRPO for a group of $G$ candidate plans $\{o_i\}$ per user prompt $q$:

$$
A_i = \frac{r_i - \mu_G}{\sigma_G}, \quad r_i = r_\text{pref}(q, o_i)
$$

SC-GRPO replaces $r_i$ with a composite:

$$
r_i = r_\text{pref}(q, o_i) + \lambda \cdot r_\text{sc}(o_i)
$$

- $r_\text{pref}$: standard human-preference / downstream-quality reward on the generated video.
- $r_\text{sc}$: **semantic-consistency** — average cross-shot embedding similarity of the plan's per-shot text against the shared user-intent embedding (and/or pairwise across shots), so paraphrasing that drifts from intent is penalized.
- $\lambda$: consistency weight, tuned so preference dominates on short plans and consistency dominates on long-horizon plans.

The rest — clipped-ratio update, KL to reference, group-relative advantages — is standard GRPO.

## Why it matters

- **Scales prompt-enhancement RL to 30-second video.** WanPE-397B beats raw prompts by +50.86 human-preference points at 30 s and +10–19 points at 5–15 s.
- **Attacks a general failure mode.** Applies whenever RL fine-tunes a planner whose local outputs will be assembled into a longer artifact (multi-shot video, multi-turn dialog plan, multi-step agent trajectory).
- **Cheap add-on to GRPO.** No new model — only a new reward term computable with an off-the-shelf text encoder.

## Gotchas & tricks

- Choice of $\lambda$ matters: too high and the planner just repeats the intent verbatim (low preference, high consistency); too low and it drifts.
- $r_\text{sc}$ using a frozen text encoder can be gamed by copying keywords; anchoring on an *intent embedding* rather than pairwise shot similarity is more robust.
- Only useful when the downstream generator actually respects the plan — a T2V that ignores the screenplay makes the SC signal moot.

## Sources

- Paper: *WanPE: Towards Cinematic Prompt Enhancement for Modern Text-to-Video Generation* — Alibaba Wan Team, 2026 — [arXiv:2609.30221](https://arxiv.org/abs/2609.30221).
- Base algorithm: [grpo](grpo.md) — DeepSeekMath, 2024.
