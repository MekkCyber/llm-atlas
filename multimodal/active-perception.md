# Active Perception for Omni-Modal Agents

*Depth — formulate video / omni-modal understanding as a POMDP where the model decides what to look at next.*

**TL;DR:** Most video MLLMs encode the full frame stream uniformly and pay quadratic compute. Active perception recasts the task as a **POMDP**: at each turn the model observes a slice (a small set of frames + audio), thinks, and emits an action — either commit to an answer or specify the next slice to attend to. Persistent textual memory accumulates across turns. Trained with Agentic SFT (synthesized observation–thought–action trajectories) + Agentic RL with turn-aware advantage rescaling (TAURA). A 7B OmniAgent outperforms a 10× larger baseline using significantly fewer frames per query.

**Prereqs:** [grpo](../post-training/grpo.md)
**Related:** [_rl](../post-training/_rl.md)

---

## What it is

A reframing of long-context / long-video understanding: instead of "stuff all frames into context and hope the model attends to the right ones," let the model *act* — pick which frames to read, write what it learned to a textual scratchpad, decide when to commit. Perception becomes an action in the policy's action space.

## How it works

**State** $s$: a hidden world state (the full video stream), partially observable.
**Observation** $o_t$: at turn $t$, the model receives a thin slice — a few selected frames + audio.
**Action** $a_t \in \{\texttt{LOOK(start, end, modality)}, \texttt{THINK(text)}, \texttt{ANSWER(text)}\}$.
**Memory** $m_t$: persistent textual summary, updated by `THINK` actions.

The model loops:
1. Receive observation $o_t$.
2. Update memory $m_t$ given $(o_t, m_{t-1})$ via `THINK`.
3. Either `LOOK` (pick next frame slice) or `ANSWER` (commit).
4. Reward delivered only when `ANSWER` fires; correct answer = $+1$, else $0$.

Training:

- **Agentic SFT.** Synthesize good observation–thought–action trajectories (model + verifier loop, with privileged access to the full video at synthesis time). Fine-tune the policy to imitate.
- **Agentic RL with TAURA.** Standard GRPO-style rollouts, but per-turn advantages are **rescaled by entropy-based uncertainty** at each decision point. High-uncertainty turns get larger effective advantage — credit assignment focuses on the genuine decision points (where to LOOK next) rather than mechanical THINK steps.

## Why it matters

- **Compute scales sub-linearly with video length.** Active perception reads $O(\text{turns})$ frames, not $O(\text{video duration})$. The policy learns which frames matter and ignores the rest.
- **A 7B model outperforms a 10× larger passive baseline** on omni-modal benchmarks. The win comes from compute-efficient sampling, not raw scale.
- **Generalizes the agentic pattern to perception.** Same OTA loop that drives tool-using agents now drives video / audio attention. Probably the right shape for embodied agents too.
- **POMDP framing pays off.** It surfaces the credit-assignment problem (which `LOOK` was the decisive one?) cleanly, which is exactly what TAURA solves.

## Gotchas & tricks

- **Synthesis bottleneck.** Good Agentic SFT needs trajectories where the LOOK choices are actually optimal — generating these with a privileged-access oracle is expensive.
- **Reward sparsity.** Only the final `ANSWER` gets reward; long videos mean many `LOOK`/`THINK` turns with no signal. TAURA's uncertainty rescaling partially fixes this; reward shaping helps further.
- **Audio + video together** are harder than video alone — modality choice (which to LOOK at next) becomes an additional decision dimension.
- **At inference, exposes a knob:** maximum number of LOOKs allowed. Bigger budget = higher accuracy but higher latency. Lets users dial compute–quality on the fly.

## Sources

- Paper: *Native Active Perception as Reasoning for Omni-Modal Understanding* (OmniAgent) — Xu, Wang, He, Ma, Yang, Chu, Xu, Lin, Fu, Heng, 2026 — [arXiv:2606.19341](https://arxiv.org/abs/2606.19341).
