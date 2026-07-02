# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In on-policy distillation the *student* rolls out (samples completions), and a *teacher* is queried to supervise the student's own tokens with a dense signal (typically forward-KL on token distributions or per-token log-prob targets). Compared to off-policy KD — where the student is trained on frozen teacher completions — OPD eliminates exposure bias and delivers dense signal exactly on the states the student will visit at inference. DOPD (Dual On-Policy Distillation, 2026) extends OPD with two teacher heads to avoid a failure mode called *privilege illusion*.

**Prereqs:** [_post-training](_post-training.md), [dpo](dpo.md)
**Related:** [_distillation](_distillation.md) · [multi-teacher-on-policy-distillation](multi-teacher-on-policy-distillation.md) · [rejection-sampling](rejection-sampling.md) · [rlvr](rlvr.md)

---

## What it is

Off-policy KD trains the student on teacher-sampled trajectories: cheap, but the student is never trained on its own errors. On-policy distillation reverses the source of trajectories: the *student* generates, the *teacher* only scores. The loss is typically per-token KL:

$$
L_{\text{OPD}} = \mathbb{E}_{q,\ o \sim \pi_{\theta}} \left[ \sum_t \mathrm{KL}\!\left( \pi_{\text{teacher}}(\,\cdot \mid q, o_{<t}) \,\|\, \pi_{\theta}(\,\cdot \mid q, o_{<t}) \right) \right]
$$

That is: sample $o$ from the student, query the teacher at each token position, minimize the KL. The result is a dense, per-token signal computed exactly on the states the student's own policy visits — this is what "on-policy" means in this context.

## How it works

1. **Rollout.** For prompt $q$, sample $o \sim \pi_\theta$ from the current student. Standard temperature / top-p.
2. **Teacher scoring.** For each prefix $(q, o_{<t})$, evaluate $\pi_{\text{teacher}}(\cdot \mid q, o_{<t})$. This is the expensive step — one teacher forward per token.
3. **Loss.** Per-token forward-KL (student toward teacher). Some variants use reverse-KL, JS-divergence, or a mix.
4. **Update.** Standard gradient step; no critic, no reward model, no PPO clip.

### The DOPD extension (privilege illusion)

Distillation setups often give the teacher *privileged inputs* the student can't see: gold answers, retrieval, tool output. DOPD (Li et al., 2026) shows this creates a **privilege illusion** — the student mimics teacher tokens that were actually caused by privileged information, not by capability the student could replicate.

DOPD keeps two heads: a **privileged teacher** and a **privileged student**. For each token, DOPD computes an *advantage gap* and a *probability ratio* and routes supervision between the two heads. Where the capability gap is real (the student would fail even with the extra context), the teacher supervises; where the gap is information-only (the student would match with the same context), the student self-supervises. This decouples transferable capability from mimicked information asymmetry.

## Why it matters

- **No exposure bias.** Student trains on the exact distribution it will decode from — the standard failure mode of off-policy KD (drift on unseen prefixes) disappears.
- **Denser than RL.** RL gives one scalar per response; OPD gives a KL vector per token. Much lower variance, much faster to converge on capability transfer.
- **Composes with RL.** OPD is often the *last* step after RL: RL builds capability in the teacher, OPD compresses it into a deployable student. See MOPD for the multi-teacher extension.
- **Avoids privilege illusion (DOPD).** Grounded / retrieval / tool-use settings — where "just distill" leaks privileged context — get a principled fix.

## Gotchas & tricks

- **Teacher cost per token.** Every rollout token requires a teacher forward. Cache student rollouts and batch teacher scoring; consider vLLM-style prefix caching on the teacher.
- **KL direction matters.** Forward-KL (student → teacher) is mean-seeking and forgiving; reverse-KL is mode-seeking and sharper. Papers report gains from mixing both.
- **Small student can't match a giant teacher.** The teacher's next-token distribution may be unreachable given the student's capacity; capping the number of teacher tokens per query and mixing in SFT can stabilize training.
- **Non-uniform per-token importance.** Only a small subset of tokens carries pivotal capability signal. Advantage-aware routing (DOPD's mechanism) or top-k KL truncation targets this directly.
- **Watch for privilege illusion.** If the teacher sees anything the student doesn't at training time (retrieval context, chain-of-thought scratchpads, gold answers), plan the DOPD-style routing or strip privileged context from teacher inputs.

## Sources

- Paper: *DOPD: Dual On-policy Distillation* — Li et al., 2026 — defines the privilege-illusion failure mode and the two-head routing fix.
- Paper: *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* — Agarwal et al., 2023 — earlier OPD framing for LMs.
- Paper: *Multi-Teacher On-Policy Distillation (MOPD)* — Ma et al., 2026 — multi-teacher extension, deployed in MiMo-V2-Flash.
