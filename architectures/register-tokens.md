# Register Tokens
*Depth — fixed-position tokens whose hidden states carry reasoning state across generation chunks in diffusion LMs.*

**TL;DR:** Masked-diffusion LMs (dLLMs) normally lose working memory at chunk boundaries because the model has to see previously generated *text* to keep reasoning. Register tokens (Ge et al., 2026) are a small number of dedicated positions whose continuous hidden states are trained to carry reasoning progress across chunks: the emitted text can be cleared, but the registers persist, giving the model a bounded, learned scratchpad. On math and code they beat discrete-text carry by 8.5 and 19.5 points respectively.

**Prereqs:** [masked-diffusion-lm](masked-diffusion-lm.md), [multi-head-attention](multi-head-attention.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [transformer-block](transformer-block.md)

---

## What it is

A learned, fixed-size working memory built into a diffusion LM. `R` register tokens sit at reserved positions of the input; their hidden states are optimized to encode reasoning progress that can be reused after the emitted text has been discarded. The model treats registers like any other token during attention but *never emits them as text*.

## How it works

1. **Reserve R positions.** These are always present, never masked-in-content, never decoded to text.
2. **Post-training loop.** The model is post-trained to:
   - Decode a chunk of text conditioned on the prompt and current register values.
   - **Clear the emitted text** while preserving the register values.
   - Continue decoding from `(prompt + registers)` alone.

   The gradient signal is the cross-entropy of the *next* chunk after the clear, so the registers are pushed to carry whatever information is useful for the continuation.
3. **Optional RL.** After post-training, RL on long-horizon reasoning tasks further sharpens what registers hold — e.g. bounded intermediate results for multi-chunk code generation.

The registers are continuous vectors, not tokens with a vocab index; they live in the hidden-state space of the transformer and are carried across chunks as `(K, V)` cache-like state.

## Why it matters

dLLMs' long-horizon reasoning was previously bounded by the context window they had to keep emitted text in. Registers turn that soft context into a **learned compressed state**, letting the model reason over programs and math derivations that span many chunks without the compute of keeping all prior text.

- **+8.5 pts** on math benchmarks on LLaDA and Dream.
- **+19.5 pts** on code benchmarks.
- Especially strong on **bounded-length code generation**, where a correct program often spans several chunks and the intermediate state (partial AST, variable names) matters.

## Gotchas & tricks

- **`R` is small.** Typical values are 8–32; larger registers help until they saturate available update signal.
- **Registers need explicit training.** A pretrained dLLM's register positions carry noise; post-training is what makes them useful.
- **Clearing must be exact.** Any leaked information from cleared text through the KV cache defeats the "bounded state" claim.
- **Interpretability is open.** What registers encode is not currently well understood — the paper's ablations suggest they mix content and structure.
- **Compatibility with RL.** Registers are compatible with policy-gradient RL and preference-based RL, but reward-shaping across register clears requires care to avoid the model "hiding" reward-hacking state in registers.

## Sources

- Paper: *Register Tokens for Bounded-State Reasoning in Diffusion Language Models* — Ge, Singh, Zhuang, Liu, Gao, Sala — 2026 — [arXiv:2609.16372](https://arxiv.org/abs/2609.16372).
