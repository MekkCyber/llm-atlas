# Local Preference Learning (LoCo)
*Depth — prefix-matched DPO for reasoning compression: pair the same reasoning prefix with a long vs short continuation, train for the short one.*

**TL;DR:** A self-contained method for compressing long-CoT reasoning traces. Detect candidate **overthinking states** inside a model's own traces, freeze the reasoning prefix at that state, **regenerate shorter verified continuations** from it, and train with DPO on the resulting pairs. Trained only on self-generated math traces, LoCo reduces completion length by **41.7–45.4%** on math and **23.4–36.3%** on code, while maintaining or improving accuracy. No external teacher, no RL phase.

**Prereqs:** [../dpo.md](../dpo.md), [long-cot-rl.md](long-cot-rl.md)
**Related:** [long2short.md](long2short.md) · [length-penalty.md](length-penalty.md) · [../rejection-sampling.md](../rejection-sampling.md)

---

## What it is

Long-CoT models over-reason on easy problems: they emit unnecessary verification loops, duplicate strategy comparisons, and "let me double-check" detours that don't change the final answer. Existing long2short methods (model merging, shortest-rejection-sampling SFT, DPO on whole-response length, long2short RL) optimize at the **response level**: compare $y_w$ and $y_l$ as full strings.

LoCo optimizes at the **state level**: find the specific reasoning step at which the model began to over-reason, and generate competing continuations from exactly that state. The preference pair is **prefix-matched** — same opening reasoning, different continuations.

---

## How it works

Three steps, all inside one self-generated loop:

1. **Overthinking-state detection.** Sample long-CoT rollouts from the base policy on math problems with verifiable answers. For each trace, identify candidate overthinking states — typically states where the model has already produced a correct intermediate result and then begins to re-derive or verify it unnecessarily. Heuristics in the paper include duplicate strategy markers, length-past-correct-answer-token, and re-computation signatures.

2. **Shorter-continuation regeneration.** For each candidate state, **freeze the reasoning prefix up to that state** and resample multiple continuations. Keep only continuations that (a) reach the correct final answer and (b) are shorter than the original continuation from that state.

3. **DPO pair construction.** For each state, form the DPO pair:
   - $y_w$ = shortest verified continuation.
   - $y_l$ = original continuation (long).
   - Both share the same prefix; the loss operates only on the continuation tokens.

   Train with standard DPO (see [../dpo.md](../dpo.md)), applying the loss only over the continuation span.

Only the base model's own rollouts are used — no teacher model, no larger model, no RL.

---

## Why it matters

- **Localized reward signal.** The DPO loss at a prefix-matched pair isolates the decision "continue short vs continue long" from all other sources of preference signal. Global-length DPO conflates length with every other quality axis.
- **Avoids the DPO length-bias failure mode.** Because both $y_w$ and $y_l$ share the same prefix, the standard DPO length bias (preferring longer preferred completions) is neutralized — length only varies in the continuation, where "shorter correct" is the explicit positive.
- **Self-contained.** No teacher model, no RL, no reward model, no human labels. The only requirement is verifiable correctness (math: symbolic check; code: unit tests).
- **Transfers across domains.** Trained only on math, improves code too — the compression pattern generalizes.

---

## Gotchas & tricks

- **Overthinking-state detection is the load-bearing piece.** Noisy state detection produces pairs where the "long" continuation is actually necessary — you end up training against correctness. The paper's heuristics (duplicated strategies, re-derivation after correct intermediate, length past the first correctly-boxed answer) are starting points; tune per model.
- **Correctness verification is per-continuation.** Both $y_w$ and $y_l$ must reach the correct final answer. If $y_l$ was also wrong, the signal is contaminated — drop such pairs.
- **$1.5\times$ length ratio as a filter.** The paper keeps pairs where $|y_w| / |y_l| < 0.667$ roughly; too-close pairs give weak signal, too-far pairs can be trivially distinguishable without learning general compression.
- **DPO $\beta$ and length.** Because the loss span is the continuation only, the effective KL term is smaller — use a larger $\beta$ than for whole-response DPO.
- **Doesn't replace long-CoT RL.** LoCo is a *post-processing* step on a trained long-CoT model, not an alternative to long-CoT RL. Apply after the main reasoning training.

---

## Sources

- Paper: *Think Less, Locally: Local Preference Learning for Reasoning Compression* — Jiao, Li, Zhang, Zhou (XJTU MOE KLINNS + WeChat AI, Tencent), 2026, arXiv 2611.04814 — primary source.
- Related: [long2short.md](long2short.md) — Kimi k1.5's four-method comparison; LoCo slots in as a prefix-matched variant of the DPO method.
- Related: [../dpo.md](../dpo.md) — base algorithm; LoCo is DPO applied over prefix-matched pairs.
