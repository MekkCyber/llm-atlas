# WebAnything
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A framework that turns real website URLs into reusable, executable sandboxed environments for web agents without the per-site rebuilding that WebHarbor and WebArena need. An **Explorer–Builder** records live page structure, resources, and interaction outcomes; an **Independent Verifier** audits each frozen replica against the captured evidence, routing discrepancies back for local repair or targeted re-capture. 244 environments × 14 domains, 12,984 accepted tasks; wins 12 of 14 blind-audit comparisons against human-assisted WebHarbor mirrors.

**Prereqs:** [README.md](README.md)
**Related:** [workfactory.md](workfactory.md), [../data/_data-curation.md](../data/_data-curation.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Web-agent training depends on executable environments. Live sites are unstable, legally risky, and slow; hand-built mirrors (WebArena, WebHarbor) are expensive and incomplete. WebAnything automates the mirroring: an exploration agent captures what a live site does, a builder renders it into a local replica, and a verifier agent audits the replica without human intervention per site.

## How it works

**Explorer–Builder.**
- The *Explorer* interacts with the live site, recording DOM structure, resource graph (images, scripts, API responses), and the outcome of each interaction (clicks, form submits, navigation).
- The *Builder* generates a local replica implementing the same pages and the same state transitions from the recorded evidence.

**Independent Verifier.** A separate agent — not the one that built the replica — compares the frozen replica against the captured source evidence and against the site's observable rules. Discrepancies are classified:
- *Local repair:* the replica implements a page or transition incorrectly; emit a patch to the builder.
- *Re-capture:* the Explorer missed a page or state; emit a targeted re-crawl.
- *Accept:* replica matches source within tolerance.

This closes the loop: no human in the pipeline per site, and the Verifier's independence from the Builder prevents the common failure mode where the same agent that generated the replica also sign-offs on it.

**Scale.** 244 sites across 14 application domains, 12,984 accepted tasks that execute cleanly in the replicas.

**Downstream.** 5,856 interaction trajectories from WebAnything environments fine-tune Qwen3.5-9B, lifting REAL, Odysseys-Easy, and WebArena-Verified scores.

## Why it matters

If the audit-and-repair loop scales, "mine any live site" becomes a reasonable training-data source — competitive with internal web-simulator stacks at the frontier labs. In blind audits WebAnything mirrors beat human-assisted WebHarbor on 3/3 fidelity metrics and win 12/14 head-to-head comparisons with 2 ties and 0 losses, which is a strong statement about the ceiling of automated environment construction.

## Gotchas & tricks

- **Verifier independence is load-bearing.** Reusing the Builder for self-audit produces inflated agreement; the paper uses a separate agent with distinct prompts.
- **Three fidelity metrics**, not one — the audit checks structural match (DOM topology), behavioral match (transitions), and resource match (asset availability).
- **Repair-not-rewrite.** The Verifier's output is a patch to the local replica or a re-crawl request, never a change to the task intent.
- **Composable with WorkFactory** (same digest). WorkFactory makes professional-work environments; WebAnything makes web environments; both share the planner-and-verifier shape.
- **Blind audits were against WebHarbor** specifically. Comparisons against newer human-mirroring stacks are the main open question.

## Sources

- Paper: *WebAnything: From the Wild Web to Scalable Agent Environments* — Cai, Zhang, Guo, Li, Zheng, Yan — Nanyang Technological U. · Shanghai Qiji Zhifeng, 2026 — [arXiv:2610.37786](https://arxiv.org/abs/2610.37786).
- Reference: *WebArena* and *WebHarbor* — the hand-built web-agent benchmarks WebAnything compares against.
- Reference: *REAL, Odysseys-Easy, WebArena-Verified* — the downstream benchmarks lifted by WebAnything-trained Qwen3.5-9B.
