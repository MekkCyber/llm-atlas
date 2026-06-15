# Latent CoT with On-Policy RL (Switch)

*Depth — boundary-token-anchored latent chain-of-thought that is both GRPO-trainable and open to mechanistic probing.*

**TL;DR:** Latent CoT replaces visible reasoning tokens with **continuous hidden-state recurrence**, saving sequence length and compute. Two problems have plagued it: on-policy RL doesn't have a well-defined importance ratio at the latent steps, and the latent block is hard to probe causally. **Switch** solves both at once with a *single pair of discrete boundary tokens* `<swi>` and `</swi>`: the model emits `<swi>` to enter latent mode and `</swi>` to exit, [GRPO](../grpo.md)'s policy ratio is well-defined at the boundaries, and the boundaries also expose the latent computation to direct probing. Trained with a visible-to-latent curriculum and a Switch-GRPO objective.

**Prereqs:** [../grpo.md](../grpo.md), [long-cot-rl.md](long-cot-rl.md)
**Related:** [../../interpretability/README.md](../../interpretability/README.md), [orm.md](orm.md), [../_rl.md](../_rl.md)

---

## What it is

Visible long-CoT RL ([long-cot-rl.md](long-cot-rl.md)) trains the model to emit its reasoning as text tokens, then optimizes with GRPO over the verifiable reward. Latent CoT compresses the reasoning into continuous hidden-state recurrence — the model "thinks" without emitting tokens. The benefit is sequence-length and FLOPs savings; the cost has been that:

- The latent transitions aren't discrete tokens, so the standard PPO/GRPO importance ratio $\pi_\theta / \pi_{\theta_\text{old}}$ isn't well-defined at those steps.
- Without discrete anchors, you can't directly probe or causally intervene on what the latent block is doing.

Switch's insight is that one pair of discrete boundary tokens (`<swi>`, `</swi>`) makes both problems go away at once: at the boundaries the policy is making a discrete choice (whether to enter / exit latent mode), so the ratio is defined; and those same boundaries are concrete intervention points for mechanistic analysis.

## How it works

### The latent block

```
... regular tokens ... <swi>  ⟨k latent recurrent steps⟩  </swi>  ... output tokens ...
                       │       │                          │
                   discrete    continuous                discrete
                   token       hidden-state              token
                               recurrence
```

The model emits `<swi>` (a real vocabulary token) → runs $k$ continuous hidden-state updates internally → emits `</swi>` → resumes normal token generation.

### Switch-GRPO objective

The GRPO ratio at `<swi>` and `</swi>` is the standard discrete-token ratio. For the $k$ latent steps in between, gradients propagate through the recurrence via backprop-through-time — but the policy *update* is anchored at the boundary tokens. Switch-GRPO sums the entry / exit ratio contributions and treats the latent steps as a deterministic computation chained between them.

### Visible-to-latent curriculum

Training proceeds in two phases:
1. **Visible CoT:** the model is rewarded for emitting full visible reasoning, no boundary tokens.
2. **Latent transition:** the model is taught to wrap reasoning in `<swi> ... </swi>` and gradually shift load from visible tokens into the latent block, scored by the same outcome reward.

The curriculum stabilizes the boundary-token policy — without it, `<swi>` is emitted erratically and the latent block doesn't perform useful work.

## Why it matters

- **Latent CoT is now RL-trainable.** Switch-GRPO outperforms prior hidden-state-recurrence latent reasoning at similar scale, on the same outcome reward used by visible long-CoT RL.
- **Mechanistic analysis is now possible.** Probing shows: `<swi>` is a *sharply localized, learned switching policy* (not a stylistic artifact); the latent step is *causally important* (not an inert placeholder); and the computation *concentrates at a single hidden-state transition on entry*.
- **Direct evidence of "how RL improves the model from the inside."** Mechanistic probing of the latent block before / after RL shows what the RL update *changed*, not just that it changed the outcome metric.

## Gotchas & tricks

- **Boundary tokens must be real vocabulary entries.** Special tokens added without proper embedding initialization fail to be learned as switching policies.
- **Curriculum order matters.** Going latent-first (without the visible-CoT warmup) collapses to a no-op latent block.
- **Latent block depth is a hyperparameter.** Too few latent steps and there's no benefit over visible CoT; too many and the gradient flow degrades.
- **Don't overload `<swi>` with shape.** A single pair of boundary tokens is the cleanest interpretability handle; multiple latent modes with their own switch tokens muddy mechanistic analysis.
- **Outcome reward only.** The latent block has no per-step interpretable target — keep the RL reward at the trajectory level (verifier on the final answer).

## Sources

- Paper: *Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy Reinforcement Learning* — Guo et al., HKUST(GZ) / Cambridge, 2026 — [arXiv:2606.13106](https://arxiv.org/abs/2606.13106).
- Related: [long-cot-rl.md](long-cot-rl.md), [../grpo.md](../grpo.md).
