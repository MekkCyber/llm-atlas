# RNG-Bench — Reconstructive Non-Markov Games

*Depth — isolate an MLLM's ability to reconstruct past observations and act on them during multi-step interaction.*

**TL;DR:** MLLMs deployed as closed-loop policies need to act on observations that are no longer visible — past frames, briefly-revealed information, integrated spatial maps. Existing benchmarks either expose the full state, conflate hidden-state reconstruction with other skills, or test recall only after the episode ends. RNG-Bench isolates the skill: two **non-Markov** games where the agent must reconstruct hidden history *during* play. Matching Pairs (briefly-revealed card identities) and 3D Maze (egocentric views → spatial map).

**Prereqs:** —
**Related:** [README](README.md), [ceo-bench](ceo-bench.md)

---

## What it is

A diagnostic benchmark for one specific skill: *hidden-state reconstruction during ongoing interaction*. Not memory at the end of an episode, not recall in isolation — actively-used memory during a multi-step closed-loop task.

## How it works

**Matching Pairs.** A grid of face-down cards. The agent can flip two cards per turn; matched pairs are removed, mismatched pairs are flipped back. Optimal play requires remembering which card identities were revealed at which locations — across many turns, against a partial-observability backdrop.

**3D Maze.** The agent navigates a 3D maze with egocentric first-person views. Goal: reach a target. Optimal play requires integrating successive viewpoints into a coherent allocentric spatial map. The agent only sees the current view; past views must be reconstructed and stitched.

**Why "Non-Markov".** A Markov policy depends only on the current observation. Both games are explicitly non-Markov: the current observation alone is insufficient — optimal action requires history. The benchmark forces the model to act as a non-Markov policy.

**Controllability.** Both games expose difficulty knobs (grid size, maze complexity, revelation duration) so the benchmark scales as models improve.

## Why it matters

- **Cuts memory out as its own metric.** Without a clean isolated benchmark, the field can't tell whether a closed-loop MLLM failure is perception, planning, or memory. RNG-Bench separates them by design.
- **Fills the "IFEval for memory" niche.** Just as IFEval gave the field a clean instruction-following diagnostic, RNG-Bench gives a clean hidden-state-reconstruction diagnostic.
- **Directly relevant to deployed MLLM policies.** Computer-use agents, embodied agents, browser agents all face exactly this problem: the past matters, but it's no longer in the screenshot. RNG-Bench measures what the deployments actually need.

## Gotchas & tricks

- **Context-window confound.** A model with a long enough context window can just keep all past observations in context, sidestepping the reconstruction skill. RNG-Bench mitigates this with longer game durations and image-token costs, but it's an arms race; use the benchmark with a fixed context budget for fair comparison.
- **3D Maze rewards specific 3D-reasoning ability** beyond memory — failure on Maze can mean either memory or 3D spatial integration. Matching Pairs is the cleaner memory-only test.
- **Privileged-access baselines.** An oracle baseline that sees the full hidden state should always be reported alongside the model — without it, "73% on Matching Pairs" is uninterpretable.
- **Stochastic episodes.** Report mean ± variance over multiple seeds per difficulty level.

## Sources

- Paper: *Beyond the Current Observation: Evaluating Multimodal Large Language Models in Controllable Non-Markov Games* — Ding, Wei, Fang, Duan, Lin, Wang, Zang, Shanghai AI Lab et al., 2026 — [arXiv:2606.19338](https://arxiv.org/abs/2606.19338).
