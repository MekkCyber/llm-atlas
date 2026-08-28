# Multi-Teacher On-Policy Distillation (M-OPD)
*Depth — merging domain-specialized RL experts into one generalist student via dense token-level reward supervision.*

**TL;DR:** M-OPD distills several domain-specialized RL experts into a single generalist student using per-token supervision from routed teachers. Vanilla M-OPD leaves >60% of achievable headroom on the floor — not because of gradient conflict, but because of **token-level budget misallocation** across domains. Open-MOPD (Gao et al., 2026) diagnoses the cause and closes the gap with three orthogonal fixes: token-share balancing, gap-aware dynamic budget allocation, and student reward refresh.

**Prereqs:** [rlvr.md](rlvr.md), [_rl.md](_rl.md)
**Related:** [token-share-balancing.md](token-share-balancing.md)

---

## What it is

M-OPD is a post-training paradigm where a pool of domain-specialized teachers (each an RL expert in one domain — math, code, instruction-following) supervises a single student model via **on-policy distillation**: the student generates tokens; for each token, an oracle-routed teacher provides a dense reward. Its promise is a single deployable model with the pooled capabilities of many experts.

Open-MOPD is a controlled, reproducible M-OPD benchmark built on SmolLM3-3B-Base with oracle routing (so routing ambiguity is removed and only capability integration is measured), plus a diagnosis and fix for the capability-integration gap.

## How it works

On a controlled M-OPD run against a domain-routed oracle ensemble, vanilla M-OPD captures only **35.6%** of available headroom. The gap is not gradient conflict (the failure mode you'd naively suspect) but budget misallocation, driven by three orthogonal factors:

| Pathology | What goes wrong |
| --- | --- |
| **Structural sequence-length disparities** | Long-form domains (proofs, code) consume most of the token budget; concise domains (instruction-following) starve. |
| **Dynamic convergence drift** | Non-uniform learning rates cause fast-converging domains to keep updating while slower ones stall. |
| **Multi-step reward staleness** | Asynchronous policy updates mean teacher rewards are computed against an older student — dense rewards go stale. |

Open-MOPD's three-lever fix maps onto the three pathologies:

1. **Token-share balancing.** Equalize per-domain token contribution to the update — see [token-share-balancing.md](token-share-balancing.md).
2. **Gap-aware dynamic budget allocation.** Route optimization budget to domains still far from ceiling; back off from converged ones.
3. **Student reward refresh.** Recompute teacher rewards against the current student, killing staleness from asynchronous updates.

Together these elevate headroom recovery from **35.6% → 83.4%** in a single deployable student, with the recipe, training trajectories, and eval suites open-sourced on an academic hardware budget.

## Why it matters

M-OPD is the operational path many labs actually use behind the scenes to ship a single strong generalist: spin up RL experts per domain in parallel, then distill them into one. Without the three-lever fix it silently loses most of the point. Open-MOPD turns folk knowledge ("distillation eats capabilities") into three diagnosable pathologies with concrete, reproducible fixes — and provides an academically feasible benchmark for iterating on this further.

## Gotchas & tricks

- **Oracle routing is a benchmark simplification.** In production, routing decisions are themselves error-prone; the "capability integration" headroom Open-MOPD measures is an *upper bound* achievable once routing is solved separately.
- **Concise-task collapse is the canary.** Instruction-following (short outputs) is the first thing to degrade under budget misallocation; monitor it early.
- **Sequence-length disparities recur.** Any multi-domain distillation with heterogeneous output lengths needs token-share balancing or a similar knob — it's not M-OPD-specific.

## Sources

- Paper: *Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation* — Gao et al., 2026 — [arXiv:2608.19098](https://arxiv.org/abs/2608.19098)
