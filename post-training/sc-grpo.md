# Semantic-Consistency GRPO (SC-GRPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO variant for training **prompt-rewriting** LLMs, where each group's reward combines a task-quality signal (video-generation preference score) with a **semantic-consistency** term that penalises rewrites that drift from the user's original intent. Introduced to train WanPE, a 397B cinematic prompt enhancer that sits upstream of Alibaba's Wan3.0 video generator.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [_rewards.md](_rewards.md), [rlvr.md](rlvr.md)

---

## What it is

Standard GRPO for LLMs uses a single scalar reward per rollout — correctness for reasoning, or a preference-model score for helpfulness. When the LLM's job is to **rewrite** a user prompt (into a cinematic screenplay, a search query, a code prompt), a pure quality reward pushes the rewriter to over-embellish or drift toward whatever the downstream evaluator prefers, losing the user's stated intent.

SC-GRPO reshapes the reward as a **weighted sum**:

$$r_i = r_{\text{quality}}(o_i) - \lambda \cdot d_{\text{sem}}(o_i, q)$$

where $r_{\text{quality}}$ is the downstream signal (e.g. a video-generation preference model rating a video produced from rewrite $o_i$) and $d_{\text{sem}}$ is a semantic-drift measure — an embedding distance, an LLM-judge score, or a cross-encoder — between the rewrite $o_i$ and the *original* user prompt $q$. Everything else — group-relative advantage, PPO-clipped ratio, KL to reference — matches GRPO.

## How it works

Training loop per batch:

1. Sample $G$ rewrites $\{o_1, \ldots, o_G\}$ from the current policy for user prompt $q$.
2. Run each rewrite through the downstream system to get $r_{\text{quality}}(o_i)$ (in WanPE: generate a video and score human preference against the raw-prompt video).
3. Compute $d_{\text{sem}}(o_i, q)$ from a frozen judge (embedding cosine or LLM-as-judge on intent preservation).
4. Combine into $r_i$ and z-score normalise within the group to get advantage $A_i$.
5. Apply the standard GRPO clipped-ratio update with a KL penalty to the reference policy.

The consistency weight $\lambda$ trades richness against fidelity. WanPE reports that $\lambda$ is not sensitive; roughly, anything in the same order of magnitude as $r_{\text{quality}}$'s std works.

## Why it matters

- **Prevents intent drift** in prompt-rewriter pipelines that are increasingly common upstream of video, image, and code generators. A rich rewrite that changes the subject is useless.
- **Cleanly composes with any downstream reward** — the consistency term is a plug-in on top of an existing GRPO recipe; no changes to sampling, KL, or optimiser.
- **Scales.** In WanPE, SC-GRPO trains a 397B rewriter over 1M+ videos without divergence; the consistency term dampens reward hacking that pure quality rewards would exploit.

## Gotchas & tricks

- $d_{\text{sem}}$ **is a reward model**, and can itself be hacked. Frozen embedding distances are safer than LLM-judge scores here.
- The task-quality signal usually requires actually **running the downstream pipeline** (rendering a video, scoring an image). This is expensive; batch scheduling and caching matter.
- Ablations in the paper show that adding a consistency term **without** the reverse-construction data curriculum still improves fidelity but loses cinematic richness. The two are complementary.
- Because rewrites are usually short, the KL-to-reference term is small in absolute terms; SC-GRPO tolerates lower $\beta$ than typical reasoning GRPO.

## Sources

- Paper: *WanPE: Towards Cinematic Prompt Enhancement for Modern Text-to-Video Generation* — Zhu et al., Alibaba Wan team, 2026, [arXiv 2609.30221](https://arxiv.org/abs/2609.30221) — introduces SC-GRPO and applies it at 397B scale.
- Paper: *DeepSeekMath* — Shao et al., 2024 — the GRPO baseline SC-GRPO extends.
