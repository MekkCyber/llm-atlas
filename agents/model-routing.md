# Model & Workflow Routing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of running every request through one heavyweight model or workflow, a **router** picks the right one per prompt. GenRouter (HKUST-GZ, 2026) applies this to agentic image generation: a learned router directs each prompt to one of several image-generation workflows (fast T2I, LoRA-styled, edit-first, multi-model ensemble), reporting **>95% cost reduction and 65% latency reduction** vs the heavyweight static pipeline while improving visual alignment, with a continuous self-evolution loop that updates the routing policy from outcomes.

**Prereqs:** [_agent-harness.md](_agent-harness.md)
**Related:** [subtask-workflows.md](subtask-workflows.md) · [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A **model router** is a policy `router(prompt) → workflow` that dispatches inputs across a library of models/workflows with different cost, latency, and quality profiles. The router is itself typically a small model (classifier, retrieval, or lightweight LLM) that decides *which* pipeline should handle *this* prompt.

Routing is emerging as its own capability across (i) LLM serving (RouteLLM, Portkey), (ii) coding agents (choose fast vs deep model), and (iii) image generation (GenRouter's contribution).

## How it works

At the level of a workflow router like GenRouter:

1. **Workflow library.** A curated set of pipelines — fast text-to-image, LoRA-styled generation with per-style adapters, edit-first (start from a base image), ensembles of two generators voted on by a judge, etc. Each has a known cost/latency/quality profile on typical inputs.
2. **Routing policy.** For each prompt, the router encodes it and picks one workflow. Simple routers are classifiers over prompt embeddings; richer routers use a small LLM to reason about which workflow fits.
3. **Feedback loop.** Each executed prompt/workflow pair produces an outcome — human upvote, judge score, or downstream user action. Those outcomes update the router (contextual bandit, RL, or offline fine-tuning) so it improves at picking.
4. **Self-evolution.** Optionally, the workflow *library itself* is edited by the system — retiring workflows that never win, adding new ones based on gaps.

The economics: most prompts don't need the heaviest pipeline. Reserving the heavy one for the small fraction of prompts that need it, and using cheap workflows for the rest, is where the cost/latency wins come from.

## Why it matters

- Inference cost dominates ops budgets for image and video generation. A router that saves 95% while improving quality on average is a big deal even if it's not the sexiest research artifact.
- Generalizes the ["cheap model / expensive model" fallback](https://arxiv.org/abs/2404.14618) into a many-workflow decision. Same shape, richer action space.
- Turns pipeline design from a monolith into a library — each pipeline can be evolved independently, and the router mediates.

## Gotchas & tricks

- **Router quality caps everything.** A miscalibrated router either overspends (routes to heavy when unnecessary) or undershoots (routes to cheap when a hard prompt needed the strong pipeline). Explicit calibration on held-out prompts is not optional.
- **Cold-start problem.** With no outcome data, the router can't route well. Bootstrap from a static heuristic (rule-based on prompt features) and switch to the learned router once the feedback log is warm.
- **Feedback loops can lock in mediocre workflows.** A workflow that gets routed to a lot accumulates data and looks strong; a good workflow that starts under-selected never gets a chance. Explicit exploration (ε-greedy, Thompson sampling) matters.
- **Cost accounting must be honest.** "Cost reduction" numbers depend heavily on what mix of prompts you evaluate on. Report the mix.
- **Distinct from cascade/mixture models.** A cascade always tries cheap first; a mixture always uses several. A router **picks one up front** based on the input.

## Sources

- Paper: *GenRouter: Unified Workflow Routing for Agentic Image Generation* — Chen, Hou, Shu, Ruan, Xu, Guo, Chen — arXiv:2608.16721 — 2026 (HKUST-GZ).
- Related: *RouteLLM* — Ong et al., 2024 — routing across LLM tiers for cost/quality tradeoffs.
