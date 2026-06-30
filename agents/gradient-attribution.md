# Gradient-Based Connections for Multi-Agent Systems (GBC)
*Depth — token-level gradient attribution across an LLM multi-agent computation graph.*

**TL;DR:** Treat an LLM multi-agent system (MAS) as a computational graph with **gradient-based connection weights** between agent outputs and downstream agents at the token level. Build an attribution graph from the connection weights, back-propagate a task-specific loss, and use the result for fine-grained credit assignment and targeted prompt optimization. Implemented efficiently via prefix-based gradient computation in *AgentChord*.

**Prereqs:** [agents README](../agents/README.md)
**Related:** [_post-training](../post-training/_post-training.md), [_rl](../post-training/_rl.md)

---

## What it is

A differentiable layer over the MAS graph. Each edge from agent A to agent B carries a learned scalar weight per *token* of A's output, quantifying how much that token influences B's downstream behavior. Compose edges into an attribution graph and you can ask "which agent / which token caused this final-task error?"

## How it works

**Graph construction.** Encode the MAS as a DAG of agents and their message-passing edges. For each edge `A → B`, parameterize a per-token connection weight `w_{A→B, i}` over A's output tokens `i`.

**Prefix-based gradient.** AgentChord runs B with a prefix that includes a weighted version of A's tokens; back-prop through this weighted prefix yields ∂L/∂w cheaply (no need to re-run downstream agents from scratch for every gradient).

**Attribution & optimization.** After computing per-token connection weights, propagate task loss backward through the attribution graph. Use the resulting per-agent / per-step blame signal to **target prompt updates** (rewrite the high-blame agent's instructions, not the entire MAS).

## Why it matters

- **Differentiable credit assignment** is the missing piece in prompt-graph optimization (TextGrad/DSPy-style). Without it, end-to-end loss provides only a global signal, and prompt updates are blind.
- **Empirical link** between attribution quality and optimization gains: higher attribution → bigger MAS improvement. That correlation is evidence the gradient is genuinely informative, not just a regularizer.
- **Beats single-agent and multi-agent baselines** on MultiWOZ and τ-bench.

## Gotchas & tricks

- Per-token connection weights blow up parameter count if naively applied to long agent outputs; prefix-based gradients keep the runtime tractable but the *number* of weights still scales with output length.
- The attribution is only as good as the connection-weight parameterization; over-flexible parameterizations overfit, under-flexible ones miss the signal.
- Best used for prompt optimization in production MAS — not for weight updates inside frontier LLMs, where standard RLHF/RFT remains the right tool.

## Sources

- Paper: *GBC: Gradient-Based Connections for Optimizing Multi-Agent Systems* — Xiaocheng Yang et al., UIUC — arXiv:2606.28187 — https://arxiv.org/abs/2606.28187
