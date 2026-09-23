# Agent Harness
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The **harness** around an LLM agent is everything that isn't the model weights: system prompt, few-shot exemplars, tool definitions and schemas, control flow (planner / actor / critic loops), memory (scratchpad, retrieval, long-term store), and context management (truncation, summarization, working-set curation). For a fixed frozen backbone, the harness is often the *dominant* determinant of end-task capability — swapping one for another on the same model can move benchmark scores by tens of points.

**Prereqs:** *(none required — this is a foundational agent concept)*
**Related:** [_agentic-memory.md](_agentic-memory.md) · [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

An agent = **backbone LLM + harness**. The harness typically wraps five distinct concerns:

| Component | What it does | Example knobs |
| --- | --- | --- |
| Prompting | System instructions, exemplars, output-format constraints | Instruction style, few-shot count, format schema |
| Tools | Function definitions, argument schemas, response formatters | Which tools are exposed, argument granularity, retry policy |
| Control flow | The loop that runs the model and dispatches its outputs | Planner-actor, ReAct, plan-and-solve, tree search |
| Memory | Working buffer + persistent store, retrieval | Buffer size, summarization policy, retrieval strategy |
| Context management | What survives across turns and what gets evicted | Truncation, compression, working-set selection |

A model is "an agent" only in the presence of a harness that gives it a loop, tools, and state.

## How it works

At the smallest, a harness is a `while` loop:

```
state = init(task)
while not done(state):
    response = model.generate(prompt(state))
    action = parse(response)
    obs = execute(action)   # tool call, code run, browser step, …
    state = update(state, response, action, obs)
```

Every choice above is a harness design decision: how `prompt` composes state into a prompt, how `parse` turns a response into an action, how `execute` runs it, how `update` records the result and decides when to summarize. Modern harnesses (Devin, SWE-agent, DSPy, AFlow, GEPA) layer critics, planners, and self-repair loops on top.

The harness is **frozen at inference** but is itself an object of optimization. Two families:

- **Manual.** Human-designed prompts and control flow, iterated in-the-loop.
- **Automated harness search.** Treat the harness as a search space and iteratively propose + score edits (DSPy compilers, AFlow, GEPA, RRSI, …). Concrete edits look like *rewrite this prompt*, *add a critic step*, *change tool schema*, *extend memory summarization policy*.

## Why it matters

- **Same model, very different scores.** A well-designed harness routinely doubles benchmark accuracy at fixed compute. The harness is where a lot of the "agent capability" story actually lives.
- **Cheap iteration.** Editing prompts and control flow is orders of magnitude cheaper than fine-tuning weights — so the marginal-quality lever most teams pull first.
- **Now optimizable end-to-end.** Automated harness search means you can put the harness *itself* under an RL-style outer loop (see RRSI). This creates a new failure mode — **harness overfitting** — where the search memorizes the training benchmark.
- **Backbone-portable.** A good harness for one model often transfers to a stronger one; a heavily-overfit harness does not.

## Gotchas & tricks

- **Harness overfitting is real.** If you optimize a harness against a single benchmark you'll see large in-distribution gains and small (or negative) OOD transfer. Always hold out an eval that the search cannot see. RRSI's regularization scheme (edit-budget, novelty-seeking, benchmark-specific-pruning) is a concrete recipe.
- **Token cost compounds.** Every added critic / repair step multiplies per-task token spend. Prune ruthlessly — unused branches survive by accident.
- **Tool interface > tool coverage.** Fewer tools with cleaner schemas usually beats many overlapping ones. The model's parser is your bottleneck.
- **Memory is a harness component, not a separate system.** Retrieval policy, summarization cadence, and working-set curation are all harness edits — see [_agentic-memory.md](_agentic-memory.md).
- **Backbone upgrades reset the design point.** A harness tuned for a 70B model may be over-engineered for a 400B one — retest, don't assume.

## Sources

- Paper: *RRSI: Regularized Recursive Self-Improvement of Agent Harnesses* — Xia et al., Google, 2026 — treats the harness as an optimization target and shows how regularization prevents overfitting. [arXiv:2609.24972](https://arxiv.org/abs/2609.24972)
- Framework: *DSPy* — Khattab et al., Stanford — declarative compilation of prompts and control flow.
- Framework: *AFlow* — MetaGPT team, 2024 — Monte-Carlo tree search over agent workflows.
- Blog: "The bitter lesson for AI agents" — argument that harness matters more than most teams admit.
