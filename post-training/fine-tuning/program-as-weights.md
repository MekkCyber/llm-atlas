# Program-as-Weights (PAW)

*Depth — compile a natural-language "fuzzy function" spec into a parameter-efficient adapter for a small frozen interpreter.*

**TL;DR:** Everyday tasks like "alert on important log lines," "repair malformed JSON," "rank search results by intent" are hard to express as rules and today get outsourced to LLM APIs — losing locality, reproducibility, and cost control. **PAW** trains a **4B "compiler" LM** on the **FuzzyBench** dataset (10M examples) to emit **parameter-efficient adapters** targeting a small frozen **0.6B Qwen3 interpreter**. The adapter *is* the compiled program; deployment is local. On the FuzzyBench eval, a 0.6B interpreter + PAW adapter matches direct prompting of Qwen3-32B while using **~1/50th the memory** and running at **30 tokens/s on a MacBook M3**.

**Prereqs:** [README.md](README.md)
**Related:** [../rejection-sampling.md](../rejection-sampling.md), [../_post-training.md](../_post-training.md)

---

## What it is

PAW inverts the "distill a big model into a small one" pipeline. Instead of one distillation run per model, one compilation run per **function**: for each fuzzy-function spec, the compiler emits an adapter that specializes the frozen interpreter for that spec. The interpreter is small enough to run locally; the adapters are cheap to store and swap.

## How it works

### FuzzyBench

A dataset of 10M (natural-language spec, input, expected output) triples covering hundreds of everyday fuzzy-function categories: log filtering, JSON repair, intent ranking, text classification, structured extraction, etc. Constructed at scale by seeding categories and generating specs + I/O pairs.

### The compiler

A 4B LM is trained on FuzzyBench with a two-part objective: given a spec, produce (a) a parameter-efficient adapter (LoRA-shaped) targeting the frozen 0.6B interpreter, and (b) an interpreter output that matches the reference for a batch of inputs sampled under that spec. The gradient flows through the (interpreter, adapter) pair back to the compiler.

### The interpreter

A frozen 0.6B Qwen3 with LoRA-style adapter slots. At inference, the compiled adapter is loaded, and the interpreter runs on the fuzzy-function's real inputs. No further training; the compiler is used once per function.

### Deploy semantics

The whole compiled artifact — adapter weights plus a tiny wrapper — is a self-contained "program." Portable, versionable, reproducible. Compare with "call an API endpoint" or "prompt a 32B model": PAW's artifact is smaller, cheaper, and doesn't drift when the API vendor updates their model.

## Why it matters

- **Compiles LLM behavior into an artifact.** Once compiled, a fuzzy function is a file, not an API call. That changes the operational story: version control, offline execution, deterministic replays.
- **50× memory reduction.** 0.6B interpreter + PAW matches 32B direct prompting on the eval — that's a genuine order-of-magnitude improvement on the deployment side of small-task LLM work.
- **Turns adapter generation into a first-class compiler.** Prior LoRA training is one-off, per-user. A trained compiler makes adapter generation a repeatable pipeline, and FuzzyBench establishes a scale where the pipeline is credible.

## Gotchas & tricks

- **Interpreter must be strong enough.** A 0.6B interpreter's ceiling is low; PAW works because the tasks are narrow-scope fuzzy functions. Don't expect PAW to compile "write me an essay" into an adapter — the interpreter can't run it well regardless.
- **Compiler generalization is the limit.** The compiler was trained on FuzzyBench categories. Novel categories will get a plausible-but-off adapter. Evaluate compiled adapters on held-out categories, not just held-out specs.
- **Adapter drift on distribution shift.** A compiled adapter is a *snapshot* of the spec's implementation; when input distribution shifts, expect performance drift. Recompile with fresh I/O samples.
- **Adapter storage adds up.** One adapter per function is small individually but accumulates. Deduplicate similar adapters and use a routing layer if you have thousands.

## Sources

- Paper: *Program-as-Weights: A Programming Paradigm for Fuzzy Functions* — Zhang, Hotsko, Kim, Nie, Shieber, Deng (Waterloo / Cornell / Harvard), 2026 — [arXiv:2607.02512](https://arxiv.org/abs/2607.02512).
- Dataset: FuzzyBench — 10M fuzzy-function examples, released with the paper.
