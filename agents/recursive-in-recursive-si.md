# Recursive-in-Recursive Self-Improvement (RiR-SI)
*Depth — coupled RSI: an inner harness-evolution loop nested inside an outer model-RL loop.*

**TL;DR:** RiR-SI runs two nested loops. The *inner* loop evolves the agent harness with the model frozen; the *outer* loop then trains the model with RL under the (now improved) harness. Alternation lets each stage raise the other's ceiling: harness edits shape better training experience for RL, and better model weights create new opportunities for harness adaptation. Introduced with ScienceBuddy.

**Prereqs:** [harness-evolution](harness-evolution.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [_recursive-self-improvement](_recursive-self-improvement.md), [modular-harness](modular-harness.md), [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

A coupled RSI scheme that formalises how model-side and scaffold-side improvements interact. Prior work usually froze one and optimized the other. RiR-SI acknowledges that each freeze wastes potential: a smarter model exposes new harness improvements, and a better harness makes model-RL sample-efficient. The paradigm ships as the training core of the ScienceBuddy interactive research workspace (Xue et al., 2026).

## How it works

Two nested loops:

**Inner (harness) loop, model frozen:**
1. Run the agent on the current benchmark-disjoint task pool.
2. Attribute recurring failures to modules of the harness (prompts, tools, memory).
3. Propose and validate harness edits (see [harness-evolution](harness-evolution.md)).
4. Terminate when validation gains plateau.

**Outer (model) loop, harness frozen at the newly evolved version:**
1. Sample trajectories under the improved harness — richer prompts, tighter tools, better context management yield cleaner signal.
2. Apply RL (typically GRPO / rubric- or verifier-based rewards) to the model on those trajectories.
3. Terminate when the model's improvement plateaus under this harness.

Return to the inner loop with the new model. The alternation forms the "recursive-in-recursive" pattern: each recursion improves the environment of the other.

## Why it matters

Coupled RSI is what makes production agent training compound. In practice, teams have been stuck picking either (a) prompt-hack the harness or (b) fine-tune the model, and neither alone keeps improving. RiR-SI states the alternation explicitly and shows in the ScienceBuddy case studies that it drives sustained gains across four scientific task families — the first large-scale packaging of the pattern as a released research product.

## Gotchas & tricks

- **Termination criteria matter.** Running the inner loop too long overfits harness to the current model; too short leaves gains on the table. Use validation gain slope, not fixed epoch counts.
- **Harness-frozen RL still overfits to the harness.** If the harness has a quirk (e.g. a particular tool schema), the model will bake it in. Rotate harness variants across outer rounds when possible.
- **Reward-hacking cross-pollinates.** A weakness in the reward that the inner loop learned to exploit becomes a weakness the outer RL amplifies. Certificate-graded rewards (see [../evaluation/impossible-rubrics.md](../evaluation/impossible-rubrics.md)) mitigate.
- **Compute is asymmetric.** Inner rounds are cheap (no gradient step); outer rounds are expensive. Front-load inner iterations before spending an outer round.

## Sources

- Paper: *ScienceBuddy: Recursive-in-Recursive Self-Improvement for Interactive Scientific Agents* — Xue et al., 2026 — [arXiv:2609.17523](https://arxiv.org/abs/2609.17523).
- Related paradigm: *The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement* — Duan et al., 2026 — [arXiv:2609.11873](https://arxiv.org/abs/2609.11873).
