# Constrained Decoding
*Depth — restrict the decoder's token choices to those compatible with a formal grammar (JSON, regex, CFG) by masking out forbidden tokens at each step.*

**TL;DR:** Constrained decoding compiles a grammar (regex, JSON Schema, context-free) into a per-step *token mask* that zeros the logits of any token that would violate the grammar. The result is guaranteed-valid output without fine-tuning. The cost: the mask can also zero out tokens the model would have used for other purposes — including tool calls. *Tool Suppression* (Li, Zhang, Lv, 2026) reproduces this failure across multiple open-weight models when JSON Schema and tool-calling are jointly enabled. The fix is to decouple: run tool decisions in an unconstrained pass, then a schema-constrained pass for the user-facing reply.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [_agent-memory](../agents/_agent-memory.md), [context-management](../agents/context-management.md)

---

## What it is

A decoding-time intervention that restricts the support of $\pi_\theta(\cdot \mid q, o_{<t})$ to a grammar-allowed subset at every step:

```
allowed_tokens(t) = { v : prefix(o_<t) ++ v is a valid grammar prefix }
mask(v, t)        = -∞ if v ∉ allowed_tokens(t) else 0
logits'(t)        = logits(t) + mask(t)
output(t)        ~ softmax(logits'(t))
```

The grammar is compiled to a finite-state machine or pushdown automaton; at each step the engine looks up the set of token IDs that keep the parser in a valid state.

Used by every modern serving stack (Outlines, vLLM `guided_decoding`, llama.cpp grammar, XGrammar, lm-format-enforcer) to enforce JSON Schema, regex patterns, or arbitrary CFGs.

## How it works

Three implementation flavors:

- **Regex / FSM-based.** Fast, supports any regular grammar. JSON Schema with bounded depth compiles to a regex.
- **Context-free grammar (CFG)-based.** Slower, supports arbitrary nested structures (full JSON, code grammars).
- **Constrained logit warping at the decoding layer.** The mask is applied to the post-softmax probabilities or pre-softmax logits depending on engine; per-step lookup must keep up with token generation, so engines pre-compute reachability tables.

### The Tool Suppression failure mode

When a serving stack is asked to both **call tools** and produce **JSON-schema-conforming output**, many open-weight models stop calling tools — even though they call tools fine in isolation and produce valid JSON fine in isolation. Mechanism:

1. The JSON Schema constraint compiles to a token mask.
2. Tool-call tokens (often special tokens like `<|tool_call|>`) are not in the schema's allowed alphabet at any state.
3. The mask zeros the tool-call branch at every step. The model has no way to invoke a tool.

Li et al. (2026) name this **Constraint Priority Inversion**: schema satisfaction dominates action-selection because it kills entire decoding branches. The fix is *Transparent Two-Pass Execution*: run pass 1 unconstrained (model decides whether to tool-call), then pass 2 schema-constrained on the final response only.

## Why it matters

- **Production agents rely on it.** JSON-mode is the dominant API contract; without grammar masking, the model produces unparseable strings ~5–20% of the time at small sizes.
- **Mechanically guarantees validity.** No fine-tuning, no retry, no fragile prompt engineering.
- **Has nontrivial interactions with the rest of the agent stack.** Tool suppression is one example; structured-output JSON also subtly biases the model's reasoning (truncated freedom in intermediate scratchpads).
- **Free quality at scale.** Constrained decoding closes the gap between small models and large ones for many structured tasks — a 1B model with grammar masking can hit the same JSON-validity numbers as a 70B without.

## Gotchas & tricks

- **Beware the joint-constraint failure.** If you must enforce JSON and expose tools, decouple the passes (Transparent Two-Pass Execution) or use a schema that explicitly includes the tool-call structure as a permitted variant.
- **Grammar compilation cost.** Complex schemas (deeply nested, recursive) take seconds to compile. Cache.
- **Per-step lookup latency.** With large vocabs (256k) and deep parsers the per-step mask lookup can dominate decode latency. Use packed-bit reachability tables; FlashInfer and XGrammar are state-of-the-art.
- **The probability-renormalization issue.** Masking shifts probability mass to the allowed support. If the model wanted to produce a forbidden token, the mask doesn't restore "correct" behavior — it picks the model's second-best within the grammar, which can be silently wrong.
- **Helps small models more than large.** Frontier models usually produce valid JSON without help; constrained decoding's value is highest for sub-7B models.
- **Tokenizer-tied.** A grammar compiled against one tokenizer doesn't transfer to another (BPE merges change which IDs cover which strings).
- **Speculative decoding interaction.** Constrained decoding requires per-step validation; speculative drafts must be verified against the grammar, not just against the target distribution.

## Sources

- Paper: *Constraint Tax in Open-Weight LLMs: An Empirical Study of Tool Calling Suppression Under Structured Output Constraints* — Li, Zhang, Lv (Focus Technology / NJUST), 2026 — [arXiv 2606.25605](https://arxiv.org/abs/2606.25605). Defines Tool Suppression and CPI; proposes Transparent Two-Pass Execution.
- Library: *Outlines* — Willard & Louf, 2023 — [arXiv 2307.09702](https://arxiv.org/abs/2307.09702).
- Library: *XGrammar* — Dong, Lai, Chen et al., 2024 — high-throughput grammar-constrained decoding.
- Background: *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning* — Geng et al., 2023.
