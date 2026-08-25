# Environment Synthesis for Agent RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of hand-authoring an environment per benchmark task, synthesize a **persistent executable world** — entities, services, tools, state, and cross-service invariants — from a high-level scenario, and let diverse tasks emerge from the world. Introduced at scale by AgentMercury (Jeong et al., 2026), which generated 4,783 business environments spanning 14 industries and 50 countries and used them as RL training substrate.

**Prereqs:** none (start here for agents/)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md) · [../post-training/grpo.md](../post-training/grpo.md) · [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

Agent RL needs environments the way LLM SFT needs data. The historical bottleneck: environments are hand-crafted per task, so you have as many environments as benchmarks. Environment synthesis inverts the target — build a *world*, and the *tasks* fall out.

## How it works

The construction is scenario → world → tasks:

```
scenario  = "a mid-sized e-commerce company processing returns and refunds
             across 3 regional warehouses, with a legal-hold policy on
             high-value orders"

world     = LLM instantiates: entities (orders, users, warehouses),
            services (returns-API, refunds-API, legal-hold service),
            tools (function signatures the agent can call),
            state (a database seed),
            invariants (executable predicates: "refunded ⇒ not legal-held")

tasks     = sampled from the world: "process return X respecting all
            invariants" — with the invariants providing automatic
            verification.
```

Executable cross-service invariants are the key detail: they give you a *rule-based reward* automatically, without needing a separate reward model. RLVR-style training runs directly on top.

Meta-move: **construction itself is learnable.** Fine-tuning an authoring model on synthesis traces raised held-out authoring success from 3.3% → 83.3% in AgentMercury, so environment throughput can be scaled without more manual scaffolding.

## Why it matters

- **Environment scale unlocks agent RL.** RL on 4,783 auto-synthesized worlds improved Qwen3.5-4B from 12.3 → 15.7 on EnterpriseOps-GYM and — cross-domain — from 45.9 → 56.0 on AIME26, demonstrating transfer beyond the training domain.
- **Not benchmark-targeted.** Because environments are scenario-grounded, they don't teach the specific benchmark; the transfer is via general tool-use / planning / verification patterns.
- **Free verifier.** Invariants that are executable predicates are automatically usable as verifiable rewards — no learned reward model needed.

## Gotchas & tricks

- **Invariant quality is the ceiling.** If invariants are trivially satisfiable, RL exploits them. If they contradict, no policy can succeed. LLM-generated invariants must be checked for satisfiability against the seed state.
- **World diversity vs specialization.** Very diverse worlds transfer better across domains; specialized worlds hit domain benchmarks harder. Which to pick depends on the deployment target.
- **Distinguish from task synthesis.** Prior work synthesizes *tasks* against an existing environment (WebArena, τ-bench). Environment synthesis synthesizes the world itself.
- **Compatible with human-authored envs.** Nothing prevents mixing hand-authored high-quality envs with synthesized bulk — a common pattern going forward.

## Sources

- Paper: *AgentMercury: Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale* — Jeong & Yoon, 2026 — introduces the framework and 4,783-environment corpus.
- Related: *τ-bench* — Yao et al., 2024 — task-centric environment for tool-use.
- Related: *WebArena / VisualWebArena* — Zhou et al., 2023/2024 — hand-crafted agent environments.
