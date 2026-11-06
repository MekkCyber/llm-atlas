# TAC-CoT — Tool-Empowered Implicit CoT
*Depth — latent chain-of-thought interleaved with explicit tool-use tokens.*

**TL;DR:** Explicit chain-of-thought inside a tool-using loop is slow — the model emits a long natural-language reasoning segment, then a tool call, then another reasoning segment. TAC-CoT keeps the *surface* tokens only where they are needed for routing — the **tool call**, the **tool observation**, and the **final answer** — and replaces the reasoning text between them with **latent passes** (hidden-state-only, no token output). A curriculum trains the model to compress reasoning text into latent segments progressively.

**Prereqs:** [long-cot-rl.md](./long-cot-rl.md), [../ppo.md](../ppo.md)
**Related:** [length-penalty.md](./length-penalty.md) · [long2short.md](./long2short.md) · [../../agents/README.md](../../agents/README.md)

---

## What it is

Implicit (latent) CoT, popularized by COCONUT / Quiet-STaR, runs the model's "thinking" entirely in hidden states — no reasoning tokens are emitted, only the final answer. It is fast but has been restricted to self-contained tasks. Tool-using agents can't do that cleanly: the model has to emit *some* text to call a tool, and it has to read *some* text back from the tool. TAC-CoT is the hybrid: latent for reasoning, explicit for *interfaces*.

```
[implicit]  →  <tool_call name="calc" args="…">  →  <obs>42</obs>  →  [implicit]  →  <answer>…</answer>
```

## How it works

**Latent segments.** Between visible-token boundaries, the model runs forward passes that update a latent state but do not sample tokens. These latent passes replace what would have been reasoning text.

**Explicit anchors.** Tool-call tokens, observation tokens, and final-answer tokens stay fully visible — they need to be parsed by the tool executor and the final-answer reader.

**Curriculum.** Training proceeds in stages: initially the model trains on fully-explicit traces; progressively more of the inter-tool reasoning is *masked* and the model must reproduce the same tool calls and final answers given only the hidden-state continuation. By the end, most reasoning tokens are gone.

**Backbone unchanged.** No new architecture — just latent forward passes between visible tokens. A gate decides at each step whether to sample a token or run another latent pass, trained as part of the curriculum.

## Why it matters

Agent loops amplify inference cost — every tool call wraps 100s–1000s of reasoning tokens. Cutting those by the ratios the paper reports (**up to +8.8 pts accuracy with up to 6.8× fewer forward passes**) is a significant inference-cost win, and unlike pure token-compression tricks it keeps the explicit interfaces intact so existing tool infrastructure still parses correctly. Also makes **latent CoT compatible with every agent benchmark** that already runs on tool-formatted traces.

## Gotchas & tricks

- Interfaces that are *themselves* ambiguous (a tool call that depends on reasoning context the tool itself can't see) break harder when the surrounding reasoning is latent — the gate must still emit enough context.
- Latent CoT is **not interpretable** to human readers. For safety-critical tool use (e.g. filesystem writes) keep a readable summary line before each tool call.
- Curriculum pace matters — mask too aggressively too early and the model loses both the reasoning text and the tool-call structure.

## Sources

- Paper: *TAC-CoT: Tool-Empowered Implicit Chain-of-Thought via Interleaving Explicit Tokens* — Do et al., 2026 — https://arxiv.org/abs/2611.01926
- Prior art: *COCONUT* — Hao et al., 2024; *Quiet-STaR* — Zelikman et al., 2024 (background on latent-token reasoning).
