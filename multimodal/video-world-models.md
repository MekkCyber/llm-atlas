# Action-Conditioned Video World Models (ForgeWM)
*Depth — turning a bidirectional action-conditioned video generator into a low-latency few-step causal world model.*

**TL;DR:** Action-conditioned video world models need **low-latency causal generation** and reliable response to game-native controls (keyboard + mouse). Standard causal distillation gets to few-step video generation but breaks when discrete keyboard states and continuous mouse motion must stay aligned with temporally-compressed latent chunks under an autoregressive rollout. **ForgeWM** is a four-stage progressive framework that transforms a bidirectional action-conditioned video generator into an efficient few-step world model without collapsing action controllability.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [identity-preserving-generation.md](identity-preserving-generation.md)

---

## What it is

A pipeline for building an interactive video world model that (a) generates in few forward passes (target: ≤ ~4), (b) is causal (autoregressive over time, no future-view leakage), and (c) faithfully executes both discrete (keyboard) and continuous (mouse) control signals mapped to compressed latent chunks.

## How it works

Four stages, each tightens a different property:

1. **Domain adaptation.** Take a bidirectional action-conditioned base video generator; adapt it to the target interactive-control distribution (game footage + control traces).
2. **Teacher-forced causal training.** Convert the bidirectional model to causal by teacher-forcing prior frames; align the discrete/continuous control channels with the compressed latent chunk boundaries so each control tick lands on the correct latent.
3. **Causal consistency distillation.** Distill the causal teacher into a few-step student that must remain temporally consistent and action-consistent — not just generate a plausible next frame, but the *causally-implied* frame given the control input.
4. **On-policy distribution matching.** Roll out the student autoregressively; match its distribution to a bidirectional teacher's on those rollouts, correcting the drift few-step generators typically accumulate.

## Why it matters

Video world models are the front-line for two important agent applications: **playable neural games** and **embodied policy pretraining** (learn from imagined rollouts). Few-step latency is a hard requirement to make either usable inside a real interactive or RL loop. Getting there without losing causal action-following was the open problem; ForgeWM's progressive discipline — one property per stage, guard each with the next — is a general recipe for high-fidelity causal distillation.

## Gotchas & tricks

- Discrete/continuous control alignment is where subtle bugs live: a keypress that lands between latent chunk boundaries can be silently dropped by a naively-trained causal model.
- Stage 3 (causal consistency distillation) and stage 4 (on-policy distribution matching) interact — skipping stage 4 lets the student drift into out-of-distribution rollouts that the teacher never covered.
- Bidirectional teacher availability is a hard prerequisite; without one, stages 1–2 have no anchor for the causal reformulation.

## Sources

- Paper: *ForgeWM: Progressive Causal Training for Few-Step Action-Conditioned Video World Models* — Li, Lin, Wang et al., 2026 — [arXiv:2608.14022](https://arxiv.org/abs/2608.14022)
