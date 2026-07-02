# Evolution Fine-Tuning (EFT)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** LLM-driven evolutionary search (FunSearch, AlphaEvolve) has produced state-of-the-art solutions to open mathematical conjectures, GPU-kernel design, and combinatorial optimization — but the "evolution logic" lives in the search *scaffold*, not the model. Evolution Fine-Tuning (Kim et al., 2026) is a **mid-training** paradigm that converts evolutionary-search *trajectories* into supervision, teaching the LLM itself which parts to mutate, when to backtrack, and how to iterate. Trained across 371 optimization tasks so the skill transfers across problem classes.

**Prereqs:** [mid-training](mid-training.md)
**Related:** [../post-training/fine-tuning/README](../post-training/fine-tuning/README.md) · [../agents/README](../agents/README.md)

---

## What it is

Modern LLM-driven optimization runs look like:

```
scaffold:
  while budget remaining:
    pick a candidate from the archive
    ask the LLM to mutate it
    evaluate the mutation
    add to archive
```

The LLM here is a *mutator* — it doesn't decide which candidate to mutate, when to backtrack, or when to escalate. All of that lives in the scaffold. When a new problem arrives, the scaffold is redesigned; the LLM starts from scratch and forgets whatever it "learned" during the previous search.

EFT's premise: **evolutionary search is a general skill**. Instead of leaving it in the scaffold, distill it into the model weights, so the LLM itself decides mutation strategies, mutation magnitude, when to backtrack, when to give up. Then future scaffolds can be much thinner, and the model transfers evolutionary skill across problem classes.

## How it works

### Training data

Run LLM-driven evolutionary search on **371 optimization tasks** spanning math conjectures, GPU-kernel discovery, scientific-law discovery, and combinatorial puzzles. Log the full search tree per task: candidate → mutation → outcome → next candidate. This gives a corpus of successful and unsuccessful mutation trajectories.

### The EFT objective

Convert trajectories into a supervised sequence:

```
[problem statement]
[current best]
[mutation decision: what part to change, how to change it]
[outcome: better / worse / neutral]
[next mutation ...]
```

Train (mid-train) the LLM to predict the next mutation given the problem + trajectory-so-far. This teaches the model to reason about *which mutations tend to work* — which is exactly what an evolutionary search benefits from.

### Cross-task transfer

Because training spans 371 tasks, the model learns cross-task evolutionary skill: patterns like "if the last three mutations all made it worse, backtrack two steps" or "if the objective is combinatorial, try structural mutations before parameter mutations". These transfer to held-out tasks — the paper's main experimental claim.

## Why it matters

- **Moves the smart part into the weights.** Task-specific scaffolds are brittle; a model with internalized evolutionary skill is far more portable.
- **Mid-training as a paradigm.** EFT is one of the clearest examples of *mid-training* as a distinct phase between pretraining and post-training: it uses a specialized data distribution (search trajectories) with a next-token objective, but on a scale much smaller than pretraining and much broader than post-training fine-tuning.
- **A general recipe.** Any domain where an LLM-in-the-loop search produces breakthroughs (GPU kernels, math conjectures, chemistry, program synthesis) can generate EFT training data.

## Gotchas & tricks

- **Task-mix quality matters.** 371 tasks give strong transfer; a narrow task suite over-fits the model to that suite's mutation vocabulary.
- **Successful trajectories aren't enough.** The failed mutations are where backtracking-behavior is learned; keep them in the training data with appropriate labels.
- **Scaffold still helps.** EFT reduces scaffold complexity but doesn't eliminate it — you still need to score candidates and maintain an archive. Think of EFT as "learned mutation policy" rather than "learned search algorithm".
- **Overlaps with agent RL.** Once trajectories have outcome labels, you can also RL-fine-tune on them (RLVR-style); the paper focuses on supervised mid-training, but the substrate is compatible with both.

## Sources

- Paper: *Evolution Fine-Tuning: Learning to Discover Across 371 Optimization Tasks* — Kim, Kang, Cheong, Chen, Han, Jung, Kang, 2026 — UMinn / CMU / KAIST / Cambridge / Hanyang / Amazon.
- Related: FunSearch (Romera-Paredes et al., 2024) and AlphaEvolve (2025) — the scaffolds EFT distills into weights.
