# Hybrid Rule + VLM Reward
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reward function for GRPO-style RL post-training on tasks that mix *structural* correctness (parseability, schema conformance, executability) and *visual/qualitative* correctness (does it look right, does the interactive HTML actually play). Combines a **rule-based term** for the structural side with a **VLM-judge term** for the perceptual side. CogEvol documented a real-world reward-hacking loop where the VLM judge preferred visually convincing but unplayable games, then patched the reward and retrained — a useful public worked example of the failure mode.

**Prereqs:** [grpo.md](grpo.md), [_rewards.md](_rewards.md), [rlvr.md](rlvr.md)
**Related:** [rl-prompt-curation.md](rl-prompt-curation.md) · [rejection-sampling.md](rejection-sampling.md) · [rubric-as-reward.md](rubric-as-reward.md)

---

## What it is

RLVR ([rlvr.md](rlvr.md)) works when a verifier can be written in code (unit tests, math answer match). Rubric-as-reward ([rubric-as-reward.md](rubric-as-reward.md)) works when the target is a curated text corpus. Hybrid rule + VLM reward is for tasks that need *both* — generation of runnable, visually correct artifacts (slides, interactive pages, UI components, diagrams).

Two orthogonal quality axes each get their own scoring:
- **Rule term $r_{\text{rule}}$:** structural checks — does the JSON parse, does the HTML render without error, does the game have a valid move set, is the schema respected.
- **VLM term $r_{\text{VLM}}$:** perceptual quality — a VLM judge scores the rendered output ("does this look like a well-designed slide," "is this game playable and coherent").

Combined reward $r = \lambda_1 r_{\text{rule}} + \lambda_2 r_{\text{VLM}}$ (weighted sum) feeds into GRPO advantages.

## How it works

**Rule term** is written by hand per task family: a linter for the output format, an execution sandbox that runs the generated artifact, a schema validator. Fast, deterministic, cheap.

**VLM term** renders the artifact (slide image, screenshot of the HTML, gameplay recording) and passes it to a VLM judge with a domain-specific prompt. Slower and noisier but captures signal the rule term misses.

**Both terms enter the GRPO advantage** *before* group normalization, so the group mean/std handles balancing $\lambda_1$ vs $\lambda_2$ in absolute scale. Tuning $\lambda_1 : \lambda_2$ decides how much perceptual-quality signal is worth vs how much structural correctness.

## Why it matters

- **Covers the middle ground** between "verifiable" and "text-corpus-only" tasks — production of visual artifacts is a huge class.
- **Two axes catch different failure modes.** Structural errors are caught cheaply by rules; the VLM catches "runs but looks wrong."
- **Scales down.** CogEvol shows a 27B model with this recipe outperforming flagship coding models 26.9× larger on the same benchmarks, and a 4B checkpoint is released Apache-2.0.

## Gotchas & tricks

- **The VLM judge is a reward-model failure surface.** CogEvol's documented reward-hacking episode: the model learned to produce visually convincing but *unplayable* games — a canonical Goodhart failure. The VLM couldn't distinguish "looks like a playable game" from "is a playable game." Patched by adding an execution-check subterm to the rule side. Watch for analogous failures on your task.
- **Rule term is your ground truth.** When rule and VLM disagree strongly, trust rule; the VLM is easier to fool.
- **Judge selection matters.** A too-weak VLM gives noisy rewards; a too-strong (frontier API) VLM makes RL too expensive to run at scale. CogEvol uses a dedicated in-family VLM.
- **Rendering cost is often the bottleneck.** Screenshotting an HTML page or rendering a video adds seconds per rollout — plan the rollout batch size accordingly.
- **Don't skip the reward-audit step.** Sample rewards from mid-training and hand-check them against your intuition. If the VLM is rewarding outputs you'd reject, either patch the prompt or add a rule-side check.

## Sources

- Paper: *CogEvol: Towards Efficient and Reliable Learning Environment Generation* — Tu et al. — Tsinghua / OpenMAIC, 2026 — arxiv.org/abs/2608.30968.
- Code: github.com/CogEvol/CogEvol-4B.
