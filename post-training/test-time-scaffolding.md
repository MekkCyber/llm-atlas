# Test-Time Scaffolding
*Depth — transfer capability from a strong model to a weak one at inference by having the strong model author the weak one's execution harness.*

**TL;DR:** Instead of updating the weak model's weights, use the strong model to *build the runtime scaffold* the weak model executes through — deterministic code offloads, benchmark-specific routing, strict format enforcement. On four Theory-of-Mind benchmarks this doubled average target performance from **0.49 → 0.91**. Framed as "AI4AI at test-time" by Qian et al. 2026 and extended to embodied agents ("SHAPER") the same week.

**Prereqs:** [rejection-sampling](rejection-sampling.md)
**Related:** [../agents/README.md](../agents/README.md), [../agents/skill-libraries.md](../agents/skill-libraries.md)

---

## What it is

A form of **strong-to-weak capability transfer** where the strong model never trains the weak one — it writes a *harness* the weak one runs through. The harness is code + prompts + routing tables + validators. Iteration happens on a held-out validation slice: the builder writes a harness, evaluates the target through it, gets scores back, revises. Target weights are frozen throughout.

## How it works

Round-based iteration:

1. **Held-out slice.** Reserve ~5% of the benchmark as a validation set. The builder never sees the full test set during iteration.
2. **Builder pass.** The strong model reads the current harness, the validation performance, and a small set of failure examples, and outputs a new harness. Typical harnesses combine: deterministic Python for unstable subroutines, per-benchmark routing that dispatches inputs to different prompts, strict output-format validators.
3. **Target rollout.** The weaker model runs the harness on the validation slice; scores update.
4. **Repeat** for several rounds; then freeze the harness and evaluate on the full test set.

Gains come primarily from **offloading unstable reasoning into deterministic code**, **benchmark-specific routing**, and **strict answer-format enforcement** — not from sampling more or reasoning longer. This is a hard-to-fake observation because those knobs are all *removable* individually.

## Why it matters

- **Capability transfer without training access.** Works for closed-weight strong models (which is most of them right now) and for any target model you can just call an API on.
- **Complements weight-level distillation, doesn't replace it.** For domains where a scaffold can encode the win (structured tasks, tool-use, format-heavy benchmarks), scaffolding is cheaper and immediate.
- **Recasts "how to deploy a smaller model" as a code-generation problem** the frontier model solves once.
- **Extends beyond text.** SHAPER (Wang et al. 2026, [arXiv:2608.11350](https://arxiv.org/abs/2608.11350)) applies the same recipe to embodied agents, evolving reusable skills and a context/code harness via rollouts.

## Gotchas & tricks

- **Weaker targets gain the most.** The stronger the target, the smaller the delta — scaffolding is not going to move the frontier model itself.
- **Builder reasoning effort matters monotonically.** Increasing the builder's inference budget improves harness quality; skimping on the builder is a false economy.
- **Benchmark overfitting is a live risk.** Because the harness is optimized against the validation slice, held-out test sets and adversarial task variants are the only honest signal — report both.
- **Not a substitute for training when the win is truly capability-shaped.** Scaffolding cannot install a capability the target model fundamentally lacks; it reshapes and channels the capability it has.
- **The "unstable reasoning → deterministic code" trick generalizes.** Most gains in practice come from moving fragile intermediate steps into Python, not from clever prompting.

## Sources

- Paper: *AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses* — Qian, Zhao, Yang, Wang, Qiu, Ji, Savarese, Wang, Heinecke, 2026 — [arXiv:2608.12307](https://arxiv.org/abs/2608.12307) — the ToM-benchmark demonstration and 0.49 → 0.91 headline result.
- Paper: *Self-Evolving Embodied Agents via Skill-Harness Evolution* — Wang, Ma, Chang, Luo, Yang, Feng, Yang, Li, 2026 — [arXiv:2608.11350](https://arxiv.org/abs/2608.11350) — same principle applied to embodied agents (SHAPER).
