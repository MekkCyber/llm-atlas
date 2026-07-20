# Long-Context RL (Million-Token GRPO)
*Depth — architecture-aware execution stack for million-token GRPO under a fixed GPU budget.*

**TL;DR:** Inference systems have stretched to million-token contexts, but RL post-training is still stuck at ≤256K. **LongStraw** closes that gap: for each GRPO step it evaluates the shared prompt **without autograd**, retains only the model-specific state later tokens need, and replays each short response branch **one at a time**. Peak memory scales with response length and hidden state — nearly independent of group size — trading wall-clock for memory linearity.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [partial-rollouts.md](partial-rollouts.md)
**Related:** [../architectures/mla.md](../architectures/mla.md), [dualpipe.md](dualpipe.md)

---

## What it is

GRPO's memory cost per step is dominated by the *shared prompt* forward-plus-backward, replicated once per group member. At million-token prompts and even modest group sizes, this blows out GPU memory long before the compute is the bottleneck.

LongStraw is an execution stack — not a new algorithm — that reshapes the GRPO step to make million-token prompts affordable. The paper demonstrates it on two contemporary architectures: a hybrid recurrent + full-attention model (Qwen3.6-27B) and a compressed-attention MoE (GLM-5.2).

## How it works

### The three-move recipe

1. **Detached prompt forward.** Run the shared prompt through the model *without* autograd. The prompt is identical across all group members, so its gradient contribution can be reconstructed later — no need to hold the full computation graph in memory.
2. **Architecture-aware state retention.** Cache only the state later response tokens will consume: recurrent hidden states for the recurrent portion of hybrid attention; compressed KV cache for MLA-style compressed attention; expert-routing metadata for MoE. Discard everything else.
3. **Per-response replay.** For each of the $G$ short response branches, replay only that response's forward-plus-backward against the cached prompt state. Peak memory sees one response's worth of activations at a time.

The end-to-end memory profile becomes: `state cache (fixed) + one response's activations (fixed) + optimizer state (fixed)`. Group size $G$ enters only as extra *time* to iterate through the responses.

### Measured capacity

- **8×H20 GPUs, Qwen3.6-27B hybrid:** grouped scoring + response backward at 2.1M positions for $G = 2$ and $G = 8$. Enlarging the group adds only **+0.21 GB peak allocated memory**. A separate stress test reaches **4.46M positions**.
- **32×H20 GPUs, GLM-5.2 MoE:** end-to-end LongStraw execution across all 78 layers for a 2.1M-token prompt.

### The correctness caveat

The paper is explicit that these results are *execution-capacity* demonstrations, not full training-correctness runs. The captured prompt state is detached; some distributed forward and gradient composition paths remain incomplete. LongStraw is the plumbing; validating that end-to-end million-token training converges is future work.

## Why it matters

- **Agentic RL needs deployment-length contexts.** Tool traces, evidence graphs, prior turns pile up — post-training with a 256K cap can't teach behavior over million-token trajectories.
- **The architecture-awareness is the point.** A generic "gradient checkpointing" story doesn't work because different attention flavors need different state retention. LongStraw makes the retention scheme explicit per architecture.
- **Group size becomes free (in memory).** GRPO already prefers larger $G$ for lower-variance advantages; when memory-per-group is ~0.2 GB, going from $G=8$ to $G=64$ is a wall-clock decision, not a hardware one.

## Gotchas & tricks

- **This is not "gradient checkpointing done well."** LongStraw specifically discards autograd on the shared prompt, then reconstructs its gradient contribution through the per-response replays. It's a different execution graph, not a memory-optimized version of the standard one.
- **Recurrent + full-attention hybrids retain differently than MoE.** The state cache for Qwen3.6-27B carries recurrent hidden state; for GLM-5.2 it carries compressed KV. A new architecture needs its own retention scheme.
- **Distributed gradient composition is not fully validated.** The paper flags this openly — reproduction attempts should not assume end-to-end convergence at 4M positions without their own numerical checks.
- **Wall-clock cost grows linearly in group size.** LongStraw trades memory for time. If your rollout budget is tight, the per-response replay adds up.

## Sources

- Paper: *LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget* — Zhou, Liu, Zhou, Qiao, Gao, Zhang, et al., 2026.
