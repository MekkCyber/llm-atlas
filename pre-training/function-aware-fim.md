# Function-Aware Fill-in-the-Middle
*Depth — a mid-training objective that masks whole functions selected by program-dependency-graph analysis to teach coding agents the action→observation→continuation pattern.*

**TL;DR:** Ordinary code contains the same conditioning structure a coding agent uses at inference — a caller binds arguments, a callee returns a value computed elsewhere, downstream code consumes the return. Function-aware FIM exploits this by treating every function call site in real code as a training example: mask the function body (the "observation"), keep the caller and downstream context, and train the model to fill in a body consistent with what its callers expect. Masks are selected via **program-dependency-graph analysis** with a **complexity + inferability double criterion**. Applied as a mid-training pass (2.6B tokens, 968 repos), improves SWE-Bench Verified by **+2.8 to +3.2** on top of Qwen2.5-Coder and Qwen3 backbones.

**Prereqs:** [mid-training](mid-training.md)
**Related:** [../data/decontamination](../data/decontamination.md)

---

## What it is

A fill-in-the-middle variant tailored to the *shape* of coding-agent inference. Standard FIM masks random spans; function-aware FIM masks entire function bodies, chosen so the masked span is (a) non-trivial (complexity above a threshold) and (b) inferable from the surrounding caller and downstream code (inferability above a threshold). This turns each training example into a small end-to-end agent-loop simulation on real GitHub code.

Sits inside the mid-training curriculum — after Stage 1 pretraining, before agentic post-training (SFT + RL on trajectories).

## How it works

### The isomorphism

A coding agent's inference loop:

```
[reasoning] → [tool call: run_python(x)] → [observation: 42] → [continued reasoning]
```

A function call site in ordinary code:

```
def caller():
    x = ...
    y = callee(x)      # ← the "observation"
    return process(y)  # ← the "continuation"
```

Callers "bind arguments" (analogous to writing a tool call), callees "return values computed elsewhere" (analogous to a tool result), downstream code "consumes the return" (analogous to reasoning over the observation). This pattern exists at internet scale.

### Mask target selection

For each repo, build the **program dependency graph** (PDG) and pick candidate functions to mask. Two thresholds:

- **Complexity floor.** Mask only functions whose body has non-trivial computation. Trivial one-liners provide no training signal.
- **Inferability floor.** Mask only functions whose caller + downstream context contains enough constraint that the body is recoverable. Otherwise the label is arbitrary and the loss is noise.

The double criterion selects "the sweet spot" — hard enough to teach, easy enough to be a valid label.

### Training

- Corpus: 2.6B decontaminated tokens from 968 GitHub repos.
- Objective: standard FIM prefix/suffix/middle format, with the middle span always being a masked function body.
- Applied as **mid-training** on top of an already-code-pretrained base (Qwen2.5-Coder-Instruct 7B/14B, Qwen3-8B).
- Followed by the same agentic post-training pipeline as the baselines — this isolates the mid-training contribution.

## Why it matters

- **Aligns pretraining with deployment structure.** Standard FIM teaches the model to fill in random spans; the deployment pattern is more specific. Aligning the two picks up 3-point SWE-Bench gains that architecture and post-training left on the table.
- **Cheap.** 2.6B tokens is <1% of the base model's pretraining. Reusable — the same corpus can be re-masked for other backbones.
- **Generalizes.** The action-observation-continuation structure isn't specific to code. Tool-use trajectories, RAG contexts, and multi-agent handoffs all have the same shape; the same PDG-style selection could be adapted to each.

## Gotchas & tricks

- **Decontamination is required.** The corpus overlaps with SWE-Bench repos; the paper decontaminates against Verified + Lite before mid-training. Without this the gains are inflated.
- **Complexity + inferability must be *jointly* satisfied.** Filtering on complexity alone gives many uninferable masks (bodies whose callers don't constrain them); filtering on inferability alone gives trivial completions.
- **PDG construction is language-specific.** Paper is Python-focused; extending to Java/Go/Rust requires per-language PDG tooling.
- **Doesn't replace agentic post-training.** Mid-training is a prior; SFT + RL on real trajectories is still needed to teach interaction-specific behaviors (multi-turn, error recovery).

## Sources

- Paper: *Function-Aware Fill-in-the-Middle as Mid-Training for Coding Agent Foundation Models* — Wang et al., Waterloo / Vector / NVIDIA / Verdent, 2026 — arXiv 2607.12463.
- Related: [mid-training](mid-training.md) — the curriculum slot this objective fills.
