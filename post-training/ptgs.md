# PTGS — Posterior-Tempered Group Sampling
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A plug-in sampler for group-based RL (GRPO and relatives) that sets the *per-prompt* temperature from a Bayesian posterior over that prompt's difficulty. Hard prompts get hotter sampling (more exploration); easy prompts get cooler sampling (more exploitation). Pays a smaller [sharpening-tax](sharpening-tax.md) than fixed-temperature RL while also improving pass@1.

**Prereqs:** [grpo.md](grpo.md), [sharpening-tax.md](sharpening-tax.md), [_rl.md](_rl.md)
**Related:** [../evaluation/pass-at-k.md](../evaluation/pass-at-k.md) · [rl-prompt-curation.md](rl-prompt-curation.md)

---

## What it is

Fixed-temperature RL rollouts spend the same exploration budget on every prompt. But easy prompts (verifier passes at $T=0.3$) waste rollouts; hard prompts (nothing passes until $T=1.2$) silently zero out the policy gradient because every rollout fails. PTGS treats each prompt's difficulty as a latent to be estimated and tempers the sampling temperature per prompt accordingly.

## How it works

For each prompt, maintain a Bayesian estimate of difficulty from recent rollout rewards (a Beta-like posterior over "pass probability at nominal temperature"):

1. Pull the prior from the moving posterior over similar prompts (or a flat prior at the start of training).
2. Update the posterior with the current prompt's rollout rewards.
3. Map the posterior mean (and optionally variance) to a temperature: harder → higher $T$; easier → lower $T$.
4. Sample the group of rollouts at that $T$. Feed the resulting advantages into GRPO (or another group-based update) unchanged.

The paper's version is parameterized so that at a nominal pass-probability (e.g. 0.5) temperature equals the baseline fixed value; it drifts above/below for harder/easier prompts respectively.

## Why it matters

- **Better rollout utilization.** Hard prompts get rollouts that actually succeed sometimes (non-zero gradient); easy prompts spend fewer rollouts on redundant pass-everything groups.
- **Lower sharpening tax.** Coverage of hard prompts is retained because the policy continues to sample variants during training; the resulting policy is less mode-collapsed. See [sharpening-tax.md](sharpening-tax.md).
- **Pass@1 also improves.** The paper reports PTGS wins on *both* axes vs the fixed-$T$ baseline in two agentic environments — not a tradeoff.
- **Plug-and-play.** No new network, no auxiliary losses. One extra data-structure per prompt.

## Gotchas & tricks

- **Cold-start requires a prior.** With no history, PTGS falls back to the fixed temperature. Many agentic prompt pools reuse prompts over epochs, which is where the posterior is most useful.
- **Prompt identity matters.** The posterior is per prompt; shuffling prompts across workers needs a shared store, or each worker learns a weak posterior from its slice.
- **Temperature range should be bounded.** Letting $T \to 0$ on trivial prompts cuts entropy to the point that GRPO's $\sigma_r$ collapses (zero advantages). Clip $T \in [0.3, 1.5]$ or similar.
- **Not the same as prompt curriculum.** Curriculum selects *which* prompts to show; PTGS changes *how* they're sampled. They stack cleanly.

## Sources

- Paper: *Sharpening Tax in Post-Training* — Oh et al., 2026 — [arXiv:2610.01509](https://arxiv.org/abs/2610.01509). Introduces PTGS as the plug-in sampler.
- Related: Tempered sampling in classical RL / bandits (Thompson sampling, posterior-aware exploration).
