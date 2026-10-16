# Capability Filtering (for SFT reasoning-trace supervision)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Rule for building an SFT dataset: **check whether the base model already solves a prompt at pass@8 without a reasoning trace. If yes, keep the prompt without CoT. If no, add a verified reasoning trace.** Trace supervision is spent where it buys capability, not where it is redundant. Used in Apertus 1.5 SFT.

**Prereqs:** [quality-filtering.md](./quality-filtering.md), [_data-curation.md](./_data-curation.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) · [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md) · [../case-studies/apertus-1-5.md](../case-studies/apertus-1-5.md)

---

## What it is

Adding reasoning traces to every SFT prompt is wasteful (and sometimes harmful): short, easy prompts don't need CoT, and heavy CoT supervision on them teaches the model to over-reason. On the other hand, hard prompts without traces leave the model guessing.

Capability filtering picks per prompt: **does the base model already handle this?** If yes, keep the prompt bare. If no, generate and verify a reasoning trace, and include that.

## How it works

Per prompt $x$ in the SFT candidate pool:

1. **Capability probe.** Sample $k$ responses from the base model (Apertus 1.5 uses $k = 8$, pass@8). For prompts with a verifier (math checker, unit tests, exact match), compute whether any of the 8 responses pass.
2. **Routing by result:**
   - **If pass@8 = 1 (model succeeds):** keep the prompt in the SFT set *without* a reasoning trace. The target is the gold answer only.
   - **If pass@8 = 0 (model fails):** generate a reasoning trace from a stronger model (teacher), verify it against the gold answer, and keep only verified traces. Include the trace in the SFT target.
3. **Final SFT set:** mixture of trace-free easy prompts and trace-supervised hard prompts.

## Why it matters

- **Spends supervision where it buys capability.** Trace supervision has a cost (longer sequences, slower training, risk of over-reasoning); capability filtering only pays it when there is a return.
- **Measured cost.** Apertus 1.5 reports a 2.2-point drop on thinking-off vision tasks when trace supervision is applied uniformly — a real alignment-vs-capability tradeoff that filtering avoids.
- **Composes with hybrid-reasoning policies.** For models with think / no-think modes, capability filtering gives per-prompt labels for which mode to teach.
- **Cheap to run.** Pass@8 probes are one extra inference per prompt in the candidate pool; the saved SFT compute from skipping unnecessary traces usually exceeds the probe cost.

## Gotchas & tricks

- **Verifier dependency.** Capability filtering needs an automatic correctness check per prompt. For verifiable domains (math, code, exact-match QA) this is cheap; for open-ended prompts it is not.
- **Pass@$k$ threshold is a knob.** $k = 8$ is Apertus 1.5's choice; smaller $k$ is cheaper but noisier (false negatives route easy prompts to the trace-training pool). Larger $k$ is more accurate but increases probe cost.
- **Stronger teacher required.** If the base model fails pass@8 but no stronger model succeeds, you cannot generate a verified trace. Those prompts should be dropped or sent to RL.
- **Trace quality controls still apply.** Even verified traces may be rambling, mixed-language, or formatted poorly; downstream filters (readability, length, language consistency) remain necessary. R1 Stage 3 is a related recipe — see [rejection-sampling](../post-training/rejection-sampling.md).
- **Not a reasoning-only tool.** The same probe-and-route logic applies anywhere you'd otherwise supervise with extra context (tool-use traces, chain-of-notes, retrieval citations).
- **Interaction with no-think RL.** Prompts the base already solves without CoT may regress under naive no-think GRPO; see [thinking-leakage](../post-training/reasoning/thinking-leakage.md) for the related failure mode.

## Sources

- Paper: *Apertus v1.5 Technical Report* — 2026 — [arXiv:2610.15542](https://arxiv.org/abs/2610.15542). Capability filtering is used in production SFT with ~3.7M text samples; pass@8 is the cutoff for routing prompts to trace supervision.
