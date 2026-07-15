# Proxy-guided Update Signal Transfer (PUST)
*Depth — turn post-training into a modular pipeline where cheap proxies produce reusable optimization signals for larger primaries.*

**TL;DR:** Instead of coupling exploration and distribution alignment on the primary model, PUST runs post-training on a **lightweight proxy**, extracts the **relative improvement** between the proxy's initial and optimized states, and transfers that directional signal to the primary. Signals are asynchronous, cacheable, and reusable across primary models and families.

**Prereqs:** [rlvr.md](rlvr.md), [_rl.md](_rl.md), [_post-training.md](_post-training.md)
**Related:** [direct-opd.md](direct-opd.md), [grpo.md](grpo.md), [rl-prompt-curation.md](rl-prompt-curation.md), [dpo.md](dpo.md)

---

## What it is

Classical reward-optimization and distribution-matching methods (PPO, GRPO, DPO applied to the primary) **tightly couple** three things:

1. Exploration — generating rollouts to see what works.
2. Reward evaluation — scoring those rollouts.
3. Distribution alignment — updating the policy.

The primary model pays for all three every step. PUST splits them:

- Do (1) and (2) once, on a cheap proxy.
- Extract a **relative improvement signal** — the direction the proxy moved during optimization.
- Do only (3) on the primary, guided by the cached signal.

The signal is *relative* (a shift), not *absolute* (a distribution), so it can guide primaries much stronger than the proxy — weak-to-strong by construction.

## How it works

### Three-stage pipeline

1. **Proxy exploration.** Take a small proxy $\pi_\text{proxy}^\text{init}$ and post-train it on the target domain using any standard method (RLVR/GRPO, SFT on rejection-sampled data, etc.). Save both the initial and the optimized proxy state $\pi_\text{proxy}^\text{opt}$.

2. **Update-signal extraction.** Extract the relative improvement between the two proxy states — for example, per-token or per-response log-ratios $\log(\pi_\text{proxy}^\text{opt} / \pi_\text{proxy}^\text{init})$. Cache this signal.

3. **Signal transfer.** Update the primary $\pi_\text{primary}$ using the cached signal as a guidance term in its alignment step, applied to the primary's own on-policy states. No verifier calls, no proxy calls at primary-training time — just two frozen forward passes through the proxy pair.

### The relative-vs-absolute distinction

Absolute-distribution methods (imitation, direct distillation) upper-bound the student by the teacher. Relative-shift methods **transfer the *change*** the optimization induced — that change can be beneficial even when the student starts stronger than the proxy did after optimization.

### Modularity properties

- **Asynchronous** — proxy runs happen offline; primary training pulls signals from a cache.
- **Reusable** — one proxy signal serves many primaries.
- **Composable** — multiple signals (math proxy, code proxy, format proxy) can be combined.

## Why it matters

- Turns post-training from a monolithic online optimization into a **modular, cost-efficient factory**. Same shift as batching pretraining runs against precomputed data.
- **Democratizes RL for larger models.** The compute-heavy step is done once on a small model; frontier-scale primaries reuse the signal.
- **Cross-model transfer.** A signal extracted from a Qwen3-family proxy improves a Qwen3-family primary; the paper shows the pattern generalizes across model sizes within a family and (with weaker guarantees) across families.
- Names the same abstraction as Direct-OPD (same-day paper) — evidence a real shift in how the field will structure post-training.

## Gotchas & tricks

- **Signal quality bottleneck.** Whatever the proxy failed to learn during its run is invisible to the primary. Run proxy evals first; treat weak proxy gains as a red flag for the whole pipeline.
- **Proxy-primary mismatch.** Very different tokenizers or architectures can distort the log-ratio signal. Same-family proxies transfer most cleanly.
- **KL anchor needed on the primary side.** As with any policy update, add KL to a primary reference so the signal doesn't push the primary off-distribution.
- **Sign of the signal.** Signals extracted from *degraded* proxy runs (over-training, reward hacking) transfer the bad shift as reliably as they transfer good ones. Validate the proxy delta before caching.
- **Not a replacement for verifiers.** If you have cheap verifiers and can afford primary rollouts, plain [RLVR](rlvr.md) is more direct. PUST wins when primary rollouts are the bottleneck.

## Sources

- Paper: *Proxy Exploration and Reusable Guidance: A Modular LLM Post-Training Paradigm via Proxy-Guided Update Signals* — Fu et al., KnowledgeXLab @ Shanghai AI Lab, 2026 — arXiv:2607.11505.
- Related: [direct-opd.md](direct-opd.md) — same-day companion; a concrete instantiation of the pattern for RLVR-derived shifts.
