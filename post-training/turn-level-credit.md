# Turn-Level Credit Assignment (TurnWise)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A modification to GRPO for **multi-turn streaming agents** that assigns advantage *per decision* instead of once per trajectory. Two decoupled parts: (1) **silence-completed prefix scoring** credits each turn with its *remaining* return, so a correct "stay silent and keep watching" turn is rewarded when a useful response follows; (2) **turn-mean policy optimization** averages token-level losses inside a turn first, then across turns, so a one-token "wait" carries the same weight as a 500-token reply. No critic, no extra rollouts.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [rlvr.md](./rlvr.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md) · [reasoning/length-penalty.md](./reasoning/length-penalty.md) · [_rewards.md](./_rewards.md)

---

## What it is

GRPO gives every token in a trajectory the same group-normalized advantage. That is fine when an episode is "one answer per prompt," but it is wrong for **multi-turn streaming** — a video stream where the model repeatedly decides between **respond now** and **keep watching**. All of a trajectory's turns inherit one advantage, so a trajectory full of good silences followed by a bad reply gets the bad reply's low advantage spread backward onto the correct silences (and vice versa). Turn-level credit assigns a *separate* advantage to each decision in the trajectory.

## How it works

**Silence-completed prefix scoring.** For a trajectory of turns $t = 1, \ldots, T$ with per-turn rewards $r_t$ (reward model, verifier, or rule), define each turn's score as the remaining return from that point, with any trailing non-informative actions ("silences") already settled:

$$
s_t \;=\; \sum_{\tau \geq t} \gamma^{\tau - t} \, r_\tau
$$

Then normalize across the group of trajectories sharing a prompt (GRPO-style): $A_t = (s_t - \mu) / \sigma$. Each turn now carries credit for *its own suffix*, not for turns before it.

**Turn-mean policy optimization.** The PPO/GRPO surrogate inside a trajectory is normally a sum or token-mean across all generated tokens. TurnWise computes the token-mean **inside** each turn, then the **turn-mean** across turns:

$$
\mathcal{L}(\theta) \;=\; -\frac{1}{T}\sum_{t=1}^{T} \Bigl(\frac{1}{|o_t|}\sum_{k=1}^{|o_t|} \min\bigl(\rho_{t,k} A_t,\; \text{clip}(\rho_{t,k}) A_t\bigr)\Bigr)
$$

so one-token "wait" decisions and long responses contribute equally to the update.

## Why it matters

Streaming agents (live commentary, phone assistants, driving copilots, real-time translation) make a stream of mostly-short decisions where **saying nothing at the right time** is as valuable as saying the right thing. Trajectory-level advantages silently discourage correct silences because any one bad reply later in the episode drags the whole trajectory's advantage down. Turn-level credit is a drop-in fix — no value network, no extra sampling — and the recipe generalizes to any multi-turn agent RL setup where turns have variable length and value.

## Gotchas & tricks

- Works best when turn rewards are at least *per-turn* observable (verifier, judge, or heuristic). If only trajectory-level reward is available, you can still use the suffix-sum with a single terminal reward, but the per-turn differentiation collapses.
- The turn-mean normalization is the multi-turn analogue of per-sequence length normalization; combining both naively can over-normalize. Pick one.
- Streaming settings with **asymmetric turn costs** (e.g. interrupting is worse than staying silent) should fold that into $r_t$, not into the normalization.

## Sources

- Paper: *Turns Matter: Advancing Streaming Video RL via Turn-Level Credit* — Dong et al., 2026 — https://arxiv.org/abs/2611.06728
