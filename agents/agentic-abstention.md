# Agentic Abstention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Abstention for LLM *agents* isn't the single-turn "answer or refuse" of QA abstention — it's a **sequential** decision that plays out over multiple tool-calling turns. At each turn the agent can answer, abstain, or keep gathering information; the right moment to stop is often only revealed after some environment interaction. The main empirical gap is not *whether* agents can abstain but *when*: many over-explore, some never abstain at all. CONVOLVE, a context-engineering method, distills full trajectories into reusable stopping rules and raises timely abstention without any parameter updates.

**Prereqs:** [README.md](./README.md), [../safety/refusal-suppression.md](../safety/refusal-suppression.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Standard LLM abstention benchmarks are **one-shot**: the model sees a question and outputs `answer` or `IDK`. Agent settings break that shape — the agent can *act* (call tools, browse, execute in a shell), and infeasibility often only surfaces after acting. Under-abstention wastes tool budget and can be unsafe; over-abstention breaks otherwise-solvable tasks.

Agentic abstention formalizes the sequential version: at every turn `t`, the agent's action space is `{answer(a), abstain, act(tool_i)}`. A *correct* trajectory abstains when the environment has revealed the task is infeasible or under-specified, and does so with **timely recall** — abstaining after step 40 is worse than step 5 even if the final label is right.

## How it works

Two evaluation dimensions and one training-free intervention.

**Timing-aware evaluation.** Standard precision/recall become *timely* precision/recall: the metric penalises correct abstentions that happen too late. A "should-abstain" task gets full credit only if the abstention fires within the intended window.

**CONVOLVE.** A context-engineering step that operates entirely at inference time:
1. Replay a corpus of interaction trajectories.
2. Extract the *stopping rule* that would have fired at each turn (e.g. "if the search returns no valid results after 2 refinements → abstain").
3. Package rules as a compact prompt context served alongside the agent's task.
4. At runtime the agent conditions on both the current observation and the retrieved rules.

No weights change; the recipe generalises across scaffolds (ReAct, CodeAct, WebShop-style).

## Why it matters

- **Cost + latency lever orthogonal to capability.** A frontier agent that abstains one turn earlier saves tool calls, tokens, and wall-clock, with no drop in success on solvable tasks.
- **Bigger is not better.** More capable models sometimes abstain *later* — capability biases them to keep trying — so abstention timing is its own axis worth training.
- **Composable safety signal.** Feeds directly into deployment guardrails: an agent that reliably signals "I should stop" is one where a supervisor can trust its self-reports.

## Gotchas & tricks

- **Two failure modes look identical.** An agent that never abstains and an agent that abstains too late both waste budget; you need the *timing* metric to tell them apart.
- **CONVOLVE degrades if the extracted rules are stale.** Environments drift (site UIs change); the trajectory corpus needs periodic refresh.
- **"Should abstain" labelling is expensive.** The paper's 28k task set relies on carefully curated ground truth — abstain-labels aren't cheap to bootstrap for new domains.
- **Larger models with reasoning traces show the largest gains from CONVOLVE**, suggesting the stopping rules bind to the CoT rather than replacing it.

## Sources

- Paper: *Agentic Abstention: Do Agents Know When to Stop Instead of Act?* — Luo, Wen, Wang, 2026 — [arXiv:2606.28733](https://arxiv.org/abs/2606.28733).
- Project page: https://lhannnn.github.io/agentic-abstention
