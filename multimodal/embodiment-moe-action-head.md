# Embodiment-MoE action head (DyPES-VLA)
*Depth — shared attention, embodiment-specific FFN experts for cross-embodiment VLA control.*

**TL;DR:** Cross-embodiment vision-language-action policies either (a) force all robots into a common action format (expensive preprocessing, information loss) or (b) train per-robot heads (no sharing). DyPES-VLA's action head keeps a **shared attention stack** across embodiments to capture common temporal action structure, and routes to **embodiment-specific feed-forward experts** for each robot's native kinematics. No manual action-space alignment.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [../case-studies/qwen2-5.md](../case-studies/qwen2-5.md)

---

## What it is

The action-generation head of a cross-embodiment VLA. Sits on top of a shared VLM backbone that has been pretrained with a **future-prediction objective** on cross-embodiment data (so the shared representation captures object motion, contact, and interaction-induced scene changes across robots). The head translates that shared representation into executable controls in each embodiment's native action space.

## How it works

**Two-tier factorization.**

```
VLM backbone (shared across embodiments, pretrained with future-prediction)
        |
        v
Action head:
   ┌──────────────────────────────┐
   │  Shared attention layers     │  ← temporal action structure
   └──────────────────────────────┘
        |
   ┌──────────────────────────────┐
   │  Embodiment-specific FFN     │  ← routed by robot ID
   │  experts (MoE)               │
   └──────────────────────────────┘
        |
        v
   Native action space per embodiment
```

**Routing.** Robot identity picks which FFN expert(s) run — a hard router by embodiment ID rather than a soft learned router. Attention layers see all data across embodiments; FFN experts see only their robot's data.

**Training.** Shared dynamics priors are learned by giving the VLM a future-prediction objective on cross-embodiment trajectories. Action head is trained jointly on all embodiments; the MoE structure makes gradient interference across robots minimal.

## Why it matters

- **No manual action alignment.** Prior cross-embodiment work spent significant preprocessing on converting heterogeneous action spaces into a common format. DyPES-VLA lets each embodiment keep its native action space; the router does the specialization.
- **Positive transfer via shared attention.** Sharing temporal action structure means cross-embodiment data helps every robot, without smearing individual kinematic constraints.
- **Numbers.** As a single generalist policy: 98.0% LIBERO, 59.25% RoboCasa-GR1, 89.02% RoboTwin 2.0.

Likely to become a standard factorization for VLA scaling — the MoE-per-embodiment split cleanly separates the two things the head has to do.

## Gotchas & tricks

- Hard router (embodiment ID) requires that identity to be a known input — for zero-shot new embodiments, you need a default expert or a soft router extension.
- FFN experts can drift out of sync if a robot is under-represented in a batch; balance sampling per embodiment.
- The future-prediction pretraining objective is doing a lot of work — the head design alone without it underperforms.

## Sources

- Paper: *DyPES-VLA: Learning Shared Dynamics Priors and Embodiment-Specific Control for Cross-Embodiment Manipulation* — He et al., 2026 — [arXiv:2608.06374](https://arxiv.org/abs/2608.06374)
