# Hidden CoT Extraction via Tool Registration
*Depth — externalizing frontier models' internal reasoning through the standard tool-use API.*

**TL;DR:** Closed-source frontier models hide their raw chain-of-thought so users can't inspect reasoning. This paper induces the model to externalize intermediate reasoning by **registering a custom tool through the standard tool-use API** and effectively using its free-text argument slot as a CoT scratchpad. Validates against open-source models with visible native CoT that the extracted traces are genuine (not post-hoc rationalization) by matching task performance, then applies the method to closed-source frontier models including GPT-6 Astra and characterizes how each model structures its reasoning.

**Prereqs:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md)
**Related:** [cot-monitoring](../safety/cot-monitoring.md)

---

## What it is

An extraction technique for the intermediate reasoning of API-only models that don't expose native CoT. Uses the model's own tool-calling capability against itself: a bespoke "reasoning" tool is registered with the API, its schema forces a free-text argument, and prompting nudges the model to call it before answering. The tool-call arguments then contain the reasoning trace.

## How it works

1. **Register a tool.** Define a JSON schema in the API's `tools` parameter with one string argument (`thoughts` or similar). The tool's stated purpose in the description is "make your reasoning explicit before answering."
2. **Prompt to invoke it.** Instruction primes the model to call the tool exactly once per query, with its full reasoning in the argument, before emitting a final answer.
3. **Collect the tool calls.** Even though the model produces the reasoning in service of a tool it can't actually run, the argument text is returned to the client verbatim through the API's function-calling mechanics.
4. **Validate that it's real reasoning, not rationalization.** On open-source models with visible native CoT, the extracted tool-call reasoning is compared to native CoT for the same prompts. If task-conditional performance matches, the extracted trace is reasoning; if it only matches on already-solved prompts, it's rationalization. The paper's finding: it matches.

## Why it matters

Provides a **behavioral lens** on closed-source frontier-model reasoning without needing insider access. Prior work could only look at final answers. With this method, researchers can measure how frontier models structure reasoning — token efficiency, step types, breadth-vs-depth of trees — and compare across providers.

Findings about **Astra** (GPT-6 class): token-efficient directed reasoning — the model commits to a correct trajectory early, resolves elementary steps internally, and externalizes only crucial reasoning. Contrasts with other frontier models that produce longer, more branching traces.

## Gotchas & tricks

- The technique relies on the provider not filtering "reasoning-shaped" tool arguments. If a provider adds active filtering, the surface breaks — but so far providers keep tool-call arguments intact for functional reasons.
- Cross-model comparability requires care: token-efficient reasoning under this extraction protocol is not the same as native short CoT under the model's normal thinking budget. Absolute token counts should be reported per model, not compared across.
- Ethical caveat: providers may consider this an unintended usage pattern. Cite the paper if using in production analysis.

## Sources

- Paper: *Capable yet Parsimonious: Extracting and Characterizing Hidden Chain-of-Thought in Frontier Models* — Luo et al., 2026 — [arXiv:2609.26637](https://arxiv.org/abs/2609.26637)
