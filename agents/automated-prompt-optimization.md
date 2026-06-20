# Automated Prompt Optimization
*Depth — agent-driven optimization of multi-step LLM pipelines via a meta-agent that edits prompts and (when justified) chain structure.*

**TL;DR:** Multi-step LLM pipelines (retrieval → reasoning → formatting) fail through interactions among steps, so prompt-only tuning often misses the actual bottleneck. **FAPO** (Fully Autonomous Prompt Optimization, 2026) wires Claude Code into a standardized pipeline codebase and lets it evaluate the pipeline, inspect intermediate steps, diagnose failures, propose scoped edits, and validate variants — escalating from prompt edits to structural chain edits only when attribution shows a structural bottleneck.

**Prereqs:** *(none — coding-agent literacy helps)*
**Related:** *(none in the graph yet)*

---

## What it is

The natural generalization of older "automatic prompt engineering" (APE, DSPy compilation, OPRO) past their core limitation: they only optimize prompt strings under a fixed pipeline topology. In real systems the topology — number of retrieval calls, chain-of-thought vs direct, whether to add a reflection step — is itself the lever. APO frames optimization as a *coding-agent task*: the agent has access to the pipeline source, can run evaluations, and can change anything inside a scope.

## How it works

The FAPO recipe (2026) is the cleanest published instance:

1. **Standardized harness.** A canonical codebase wraps the pipeline so the agent can run end-to-end evals with a single command and access intermediate step outputs.
2. **Score function.** A task-level objective (accuracy, win-rate, structural correctness) the optimizer maximizes.
3. **Diagnose-then-edit loop.** Each round: run the eval, inspect failing traces, attribute failures to specific pipeline steps, propose a scoped change (edit a prompt, swap a tool, restructure a sub-chain), validate by re-running.
4. **Prompt-first, structure-second.** Try prompt edits first because they are cheaper and lower-risk. Modify chain structure only when attribution identifies a *structural* bottleneck that prompt edits cannot fix.
5. **Sandbox.** All edits happen inside the agent's permitted scope so safety properties of the outer system are preserved.

## Why it matters

- Production LLM pipelines often plateau because the chain shape is wrong (missing reflection step, redundant retrieval call), not because the prompts are wrong. Tools that can only change prompts hit this ceiling.
- Removes the "prompt engineer in a loop" cost from iterating on LLM systems.
- The same harness/agent pattern generalizes to optimization beyond accuracy — latency, cost, safety constraints — by swapping the score function.
- Demonstrates a viable form of *bounded* self-improvement: a coding agent rewrites another LLM system inside a guardrail.

## Gotchas & tricks

- Evaluation cost dominates. Each optimization round runs the pipeline on the eval set; without a small fast-eval suite, the loop becomes infeasible.
- Reward hacking applies: the agent can find score-function exploits the same way RL policies do (degenerate prompts that score high on the proxy but generalize poorly). Use held-out evals.
- Structural changes risk breaking invariants the outer system depends on. The "scoped change" constraint is load-bearing.
- Topology changes interact with prompt edits — a structurally edited pipeline often needs a fresh prompt-edit pass after the restructuring.
- Don't confuse with classical hyperparameter optimization. APO operates on code and natural-language artifacts; Bayesian optimization frameworks generally cannot.

## Sources

- Paper: *Fully Autonomous Prompt Optimization of Multi-Step LLM Pipelines* — Saglam, Zhao, Nelson, Vijay, Priyanshu, Karbasi, Cisco Foundation AI / Yale, 2026 — arXiv 2606.19605.
- Related: *DSPy* — Khattab et al., 2023 — earlier compiled-prompt pipeline framework.
- Related: *Large Language Models as Optimizers (OPRO)* — Yang et al., 2023 — LLM-as-optimizer for prompts.
