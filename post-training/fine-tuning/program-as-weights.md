# Program-as-Weights (PAW)
*Depth — compiling a natural-language function specification into a small weight patch that a fixed interpreter model executes.*

**TL;DR:** Prompting a large model at inference time pays the large model's inference cost every call. **Program-as-Weights** flips this: a compiler LLM (4B) reads a natural-language spec of a fuzzy function once and emits a compact *weight artifact*; a fixed small interpreter model (0.6B Qwen3) then executes the artifact at deployment. On the paper's FuzzyBench evaluations, the 0.6B interpreter + PAW artifact matches direct prompting of Qwen3-32B while using ~1/50th the inference memory and running at 30 tokens/s on a MacBook M3.

**Prereqs:** [../rejection-sampling.md](../rejection-sampling.md), [../_post-training.md](../_post-training.md)
**Related:** [../../case-studies/qwen2-5.md](../../case-studies/qwen2-5.md)

---

## What it is

A "fuzzy function" is any function whose input/output relation is under-specified by types alone — extract sentiment, summarize a paragraph in a style, decide whether an email is urgent. Traditionally you either write brittle rules or you prompt an LLM at every call.

PAW proposes a third option: treat the natural-language *specification* as source code, compile it into weights, and execute those weights on a fixed small model at deployment. The compile-time cost is paid once; every subsequent call runs on the small interpreter.

## How it works

**Two-model architecture.**
- **Compiler** — a 4B model trained to map (natural-language spec, examples) → weight artifact. The artifact is a set of weight perturbations (in practice, LoRA-style low-rank updates, though the paper's exact form is bespoke) targeting specific layers of the interpreter.
- **Interpreter** — a fixed small model (0.6B Qwen3 in the paper). At deployment, it loads the artifact and runs. Its own weights never change per user; only the artifact does.

**Training the compiler.** FuzzyBench — a new 10M-example dataset of NL spec → target-behavior pairs — is used to train the compiler with a distillation-style objective: for each spec, the compiler emits an artifact, the interpreter runs with the artifact on eval inputs, and the loss is the distance to the target behavior (with a strong reference-model teacher). Rejection-sampling and iterative refinement bootstrap higher-quality artifacts.

**At inference.** The user provides an NL spec; the compiler is queried once to emit the artifact; the artifact is cached; all subsequent calls are handled entirely by the 0.6B interpreter with the cached artifact loaded.

## Why it matters

- **Amortizes prompt cost into cacheable weights.** Prompts are re-processed at every call — long prompts especially — and don't cache across users. Weight artifacts are computed once and cached forever.
- **On-device deployment of task-specialized models.** A 0.6B interpreter with a PAW artifact fits on a MacBook M3 at 30 tok/s and matches Qwen3-32B on the target task. That's plausible offline-first product territory.
- **A different point in the "specialize a model" design space.** SFT retrains the whole model per task. LoRA adds a per-task adapter that must be trained per task. PAW learns the *meta-map* from spec → adapter, so a new task requires only a spec, not new training.

## Gotchas & tricks

- **Scope is fuzzy functions**, not arbitrary programming. If the target task has hard structural constraints (formal grammars, deterministic parsing), specification-to-weights is unlikely to capture them.
- **The artifact is spec-specific.** Two related specs get two artifacts; PAW doesn't share structure across artifacts. Whether artifacts *should* share structure is an open question.
- **Compile quality caps output quality.** The 4B compiler is itself a bottleneck; a stronger compiler would presumably produce better artifacts. Scaling the compiler is the natural next step.
- **Depends on a specific interpreter.** The artifact format is tied to the interpreter's architecture. Different interpreter → recompile.

## Sources

- Paper: *Program-as-Weights: A Programming Paradigm for Fuzzy Functions* — Zhang, Hotsko, Kim, Nie, Shieber, Deng (Waterloo / Cornell / Harvard), 2026 — [arXiv:2607.02512](https://arxiv.org/abs/2607.02512).
- Related: *LoRA* — Hu et al., 2021 — the per-task low-rank weight-patch predecessor PAW meta-learns.
- Related: *Hypernetworks* — Ha et al., 2016 — the general "one network generates another's weights" pattern.
