# Self-Pruning of Tool Output (SWE-Pruner Pro)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Coder agents drown in tool-output context — test logs, file dumps, error traces — and that context inflates token bills and pushes useful signal out of the effective window. Prior context-pruners bolt on a *separate* code classifier. SWE-Pruner Pro's observation is that the coder agent's own internal representations, computed while reading the tool output, already encode line-level relevance. A small head reads those representations and outputs a keep/prune label per line, with a length-aware embedding keyed to the tool-output's line count. Self-pruning saves tokens *and* raises task success — the pruning is more accurate than an external classifier because it uses the agent's own view of what's useful.

**Prereqs:** [_post-training.md](../post-training/_post-training.md)
**Related:** [rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

A lightweight per-line pruning head trained on top of a frozen coder agent's internal representations. It runs at inference time, per tool response, and edits the context before the agent's next turn.

## How it works

1. When the agent reads a tool response containing $N$ lines, capture the agent's per-line hidden state $h_i$ at the point of ingestion (a specific layer / position derived by the paper).
2. Concatenate a **length-aware embedding** $\ell(N)$ that tells the head roughly how much context there is — the same line can be more or less prunable depending on how much of a token budget is under contention.
3. Feed $(h_i \Vert \ell(N))$ into a small MLP head that outputs keep/prune per line.
4. Pruned lines are removed from context before the next agent step; kept lines remain intact.

Training is standard supervised classification on labeled tool-output/keep-mask pairs. The head is tiny relative to the base agent, so inference overhead is bounded.

## Why it matters

Long-context context management is the operational bottleneck for SWE-bench-flavored agents. The paper reports up to 39% prompt + completion token savings across four multi-turn benchmarks and two backbones, with bounded overhead. More striking: on MiMo-V2-Flash, SWE-Bench Verified resolve rate goes *up* by +3.8% and long-context Oolong accuracy by +2.2 points — pruning improves quality on top of saving tokens. "The agent already knows what to prune" is a general lens: internal-representation-driven inference-time control is likely to show up in other agent settings.

## Gotchas & tricks

- The right layer to tap for $h_i$ is empirical — the paper's choice for its backbones may not transfer. Ablation before trusting the same layer on a new model.
- Length-aware embedding is load-bearing. Without it, the head over-prunes on small outputs and under-prunes on huge ones.
- Pruning at the line level assumes the tool output is meaningfully line-structured. For JSON blobs or free-form prose, a chunk-based variant is needed.
- The improvement on resolve rate implies that some pruned lines were actively *distracting*, not just neutral — echoing lost-in-the-middle findings. Self-pruning is partly a distractor removal.

## Sources

- Paper: *SWE-Pruner Pro: The Coder LLM Already Knows What to Prune* — equal-contribution team (Shanghai Jiao Tong University), 2026 — [arXiv:2607.18213](https://arxiv.org/abs/2607.18213) · [HF](https://huggingface.co/papers/2607.18213)
