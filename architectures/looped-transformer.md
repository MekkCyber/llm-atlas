# Looped Transformer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A transformer design that **reuses the same block parameters across logical depth**, trading resident parameter count for inference FLOPs. Logical depth is a hyperparameter: the same weights can be unrolled 2×, 8×, or 32×. Modern recipes decompose the stack into a **prelude → recurrent body → coda** ("sandwich") with explicit **input injection** at each loop step and careful recurrent-state design. Enables adaptive-depth inference (stop early for easy inputs) and dramatic parameter savings vs equivalent feedforward depth.

**Prereqs:** [transformer-block.md](transformer-block.md), [attention.md](../fundamentals/attention.md)
**Related:** [_moe.md](_moe.md) · [mla.md](mla.md) · [multi-head-attention.md](multi-head-attention.md)

---

## What it is

A standard transformer stacks $N$ independently-parameterized blocks. A looped transformer shares one block (or a small set) across $K$ logical steps, so **resident parameters stay small while effective compute scales with $K$**. Depth becomes a knob you can turn at inference instead of a fixed training-time choice. The design is also a natural substrate for **dynamic-depth** inference — stop looping when the output stabilizes.

Prior incarnations (Universal Transformer, Albert, Ouro, Huginn) differed in which pieces repeat, whether the input is re-injected each iteration, and how recurrent state is organized. Recent ablations isolate those axes.

## How it works

**The sandwich topology.** Three contiguous regions:
- **Prelude** — a few non-shared blocks that tokenize context and set up representations.
- **Recurrent body** — one block (or a small group) replayed $K$ times. This is where the parameter sharing lives.
- **Coda** — a few non-shared blocks that read the final recurrent state and prepare it for the LM head.

**Input injection.** At each loop iteration, re-mix the embedded input into the recurrent state (an additive or small-projection fuse). Without re-injection, deep loops forget earlier context; with it, retrieval and context-use benchmarks improve.

**Recurrent-state design.** Keep a single tied state across iterations rather than separate high/low states. Random initialization of the recurrent state at step 0 gives mixed results — zero-init or the prelude's output as init is more stable.

**Depth as a dial.** Train with a *range* of $K$ values (random $K$ per batch, or curriculum) so the model learns a depth-robust solution. At inference, pick $K$ by budget — or by an online **stopper head** that halts when its confidence crosses a threshold.

## Why it matters

- **Parameter efficiency.** Under matched train/infer FLOPs, a looped-sandwich beats a feedforward MoE with the same compute budget but ~**4× more resident parameters** (reported: 75% resident reduction at matched FLOPs, parity-or-better on DROP/GSM8K/MATH500).
- **Adaptive compute.** Harder inputs get more loop iterations without changing weights — a cheap alternative to CoT text for decision-shaped tasks.
- **Composable with MoE.** The sandwich transfers from dense 730M to an 8B-resident / 0.8B-active MoE (effective 32B when unrolled), with more balanced expert routing as a side effect.

## Gotchas & tricks

- **Loop depth and position don't commute.** RoPE/ALiBi phases need re-application at each loop step, not once.
- **Train with varied $K$**. Fixed-$K$ training produces a model that collapses at other depths.
- **Random high/low recurrent splits hurt.** Keep one tied state; the paper's ablations are explicit about this.
- **Input injection is cheap but necessary.** Omitting it costs several points on retrieval-flavored benchmarks even when model and tokens match.

## Sources

- Paper: *Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design* — Shi et al., 2026 — [arXiv:2610.23036](https://arxiv.org/abs/2610.23036)
- Paper: *LEV: Thinking Fast and Deep with Looped Decision Models* — Yan, Zhang, Sun, 2026 — [arXiv:2610.20951](https://arxiv.org/abs/2610.20951)
- Related: Universal Transformer (Dehghani et al., 2018), Ouro, Huginn — cited as prior art.
