# Context-as-Action (ConAct)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Make context curation a first-class action the agent emits, instead of letting prompt history accumulate passively. The same autoregressive policy that picks UI taps also emits explicit `fold_history`, `fold_ui_state`, and `update_record` actions. Long-horizon mobile-GUI agents trained this way (MemGUI-Agent, 2026) stay coherent across many app transitions where ReAct-style stacking blows out the prompt.

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [README.md](README.md) · [_gui-agents.md](_gui-agents.md) · [execute-distill-verify.md](execute-distill-verify.md)

---

## What it is

ReAct-style agents append every (thought, action, observation) tuple to the running prompt. On short tasks this works; on long-horizon multi-app workflows it produces prompt explosion and dilutes the cross-app facts that actually matter. ConAct reframes the agent's action vocabulary so that *context management* is part of what the policy outputs — not a wrapper applied around it.

The policy emits one of two action classes at each step:
- **Environment actions** — tap, type, swipe, navigate (what an agent normally does).
- **Context actions** — fold history into a summary, fold the current UI state into salient facts, write a new entry into the running record.

Both classes are produced by the same autoregressive head; training data labels both kinds inline.

## How it works

Three structured context fields are maintained across steps:

| Field | Holds | Updated by |
| --- | --- | --- |
| Folded action history | compressed summary of past actions | `fold_history` |
| Folded UI state | salient elements / locator strings from prior screens | `fold_ui_state` |
| Recent step record | last few raw (action, observation) pairs | implicit, rolling |

At each step the prompt contains: instruction · folded action history · folded UI state · recent step record · current screenshot. The policy decides whether to emit an environment action or a context action; context actions modify the corresponding field in place and consume no observation.

Training is SFT on trajectories annotated with ConAct calls. The MemGUI-3K dataset contains 2,956 such trajectories. The policy learns *when* to compress as a supervised target — fold too rarely and the prompt explodes; fold too often and important facts are lost.

## Why it matters

- Long-horizon multi-app tasks (booking a flight then forwarding the confirmation in a messaging app, etc.) are the realistic target for mobile agents, and they were where ReAct broke down.
- Treating memory management as a learnable target removes a layer of prompt-engineering brittleness — the policy learns its own compression budget instead of being given a fixed sliding window.
- The pattern generalizes: any LLM-driven loop with growing context (research agents, code agents) can in principle adopt ConAct-style first-class compression actions.

## Gotchas & tricks

- The supervised target depends on having ConAct-annotated trajectories. Annotating fold operations is harder than annotating UI actions because there's no ground-truth "right time to compress."
- An ill-conditioned policy can spam `fold_history` to truncate inconvenient parts of its trajectory — needs reward shaping or a verifier (cf. [execute-distill-verify.md](execute-distill-verify.md)) if extended to RL.
- Distinct from RAG and external memory stores: ConAct edits the *prompt* in place, not an external KV store. Less powerful but simpler.

## Sources

- Paper: *MemGUI-Agent: An End-to-End Long-Horizon Mobile GUI Agent with Proactive Context Management* — Liu et al., Kwai, 2026 — [arXiv:2606.19926](https://arxiv.org/abs/2606.19926).
- Code/data: https://memgui-agent.github.io/
